# -*- coding: utf-8 -*-
"""
AUTONOMOUS LOAN APPROVAL AGENT
Perceive -> Reason -> Act -> Remember -> Repeat (every 60 min)
Credentials loaded from HuggingFace Secrets (environment variables)
"""
import sys
import types

import altair as alt

# Fix for Streamlit + Altair v6 compatibility
vega_module = types.ModuleType("vegalite")
vega_v4 = types.ModuleType("v4")
vega_v4.api = alt
vega_module.v4 = vega_v4

sys.modules["altair.vegalite"] = vega_module
sys.modules["altair.vegalite.v4"] = vega_v4
sys.modules["altair.vegalite.v4.api"] = alt
import os, json, pickle, sqlite3, smtplib, threading, time, logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ── Load credentials from HuggingFace Secrets
GMAIL_USER  = os.environ.get("GMAIL_USER", "")
GMAIL_PASS  = os.environ.get("GMAIL_PASS", "")
SHEET_URL   = os.environ.get("SHEET_URL", "")

DB_FILE        = "agent_memory.db"
LOG_FILE       = "agent_log.txt"
PROCESSED_FILE = "processed_ids.json"
AGENT_INTERVAL = 3600

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

st.set_page_config(page_title="Autonomous Loan Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.hero{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);padding:2rem;border-radius:16px;text-align:center;margin-bottom:1rem;}
.hero h1{color:#00d4aa;font-size:1.8rem;margin:0;font-weight:700;}
.hero p{color:#94a3b8;margin:.3rem 0 0;}
.acard{background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:12px;padding:1rem 1.2rem;margin-bottom:.6rem;}
.acard h4{color:#00d4aa;margin:0 0 .3rem;font-size:.95rem;}
.acard p{color:#94a3b8;margin:0;font-size:.82rem;line-height:1.5;}
.log-box{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:1rem;font-family:"DM Mono",monospace;font-size:.78rem;color:#94a3b8;height:320px;overflow-y:auto;}
.running{background:linear-gradient(135deg,#052e16,#064e3b);border:1px solid #10b981;border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;}
.stopped{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0099ff);color:white;border:none;padding:.6rem 1.5rem;font-weight:600;border-radius:10px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_id TEXT UNIQUE, applicant TEXT, email TEXT,
        decision TEXT, default_prob REAL, creditworth REAL,
        loan_amount REAL, approved_amt REAL, interest_rate REAL,
        emi REAL, email_sent INTEGER DEFAULT 0, timestamp TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS agent_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_time TEXT, apps_found INTEGER, processed INTEGER,
        emails_sent INTEGER, status TEXT)""")
    conn.commit(); conn.close()

def log_decision(app_id, name, email, dec, prob, la, aa, ir, emi):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""INSERT OR IGNORE INTO decisions
            (app_id,applicant,email,decision,default_prob,creditworth,
             loan_amount,approved_amt,interest_rate,emi,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (app_id,name,email,dec,float(prob),float(100-prob),
             float(la),float(aa),float(ir) if ir else 0.0,
             float(emi),datetime.now().isoformat()))
        conn.commit()
    except Exception: pass
    conn.close()

def mark_email_sent(app_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE decisions SET email_sent=1 WHERE app_id=?", (app_id,))
    conn.commit(); conn.close()

def log_run(found, processed, emails, status):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""INSERT INTO agent_runs
        (run_time,apps_found,processed,emails_sent,status) VALUES (?,?,?,?,?)""",
        (datetime.now().isoformat(), found, processed, emails, status))
    conn.commit(); conn.close()

def get_decisions():
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 100", conn)
    except Exception: df = pd.DataFrame()
    conn.close(); return df

def get_runs():
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql("SELECT * FROM agent_runs ORDER BY run_time DESC LIMIT 20", conn)
    except Exception: df = pd.DataFrame()
    conn.close(); return df

def get_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    s = {"total":0,"approved":0,"rejected":0,"review":0,"emails":0}
    try:
        c.execute("SELECT COUNT(*) FROM decisions");              s["total"]    = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM decisions WHERE decision='APPROVED'");      s["approved"] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM decisions WHERE decision='REJECTED'");      s["rejected"] = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM decisions WHERE decision='MANUAL REVIEW'"); s["review"]   = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM decisions WHERE email_sent=1");             s["emails"]   = c.fetchone()[0]
    except Exception: pass
    conn.close(); return s

def get_processed_ids():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f: return set(json.load(f))
    return set()

def save_processed_id(aid):
    ids = get_processed_ids(); ids.add(aid)
    with open(PROCESSED_FILE,"w") as f: json.dump(list(ids), f)


# ══════════════════════════════════════════════
# AGENT LOG
# ══════════════════════════════════════════════

def agent_log(action, msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = "["+ts+"] ["+action+"] "+msg
    logging.info(line)
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = []
    st.session_state.agent_log.append(line)
    if len(st.session_state.agent_log) > 300:
        st.session_state.agent_log = st.session_state.agent_log[-300:]

def get_log_lines():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f: return f.readlines()[-60:]
    return []


# ══════════════════════════════════════════════
# STEP 1: PERCEIVE — Web Scraping
# ══════════════════════════════════════════════

def scrape_applications(sheet_url):
    agent_log("PERCEIVE", "Scraping: "+sheet_url[:60]+"...")
    try:
        if "docs.google.com/spreadsheets" in sheet_url:
            if "/edit" in sheet_url:
                gid = sheet_url.split("/edit")[0]
                sheet_url = gid+"/export?format=csv"
            elif "export?format=csv" not in sheet_url:
                sheet_url = sheet_url+"/export?format=csv"
        r = requests.get(sheet_url, timeout=15)
        if r.status_code != 200:
            agent_log("ERROR","HTTP "+str(r.status_code)); return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text))
        agent_log("PERCEIVE","Found "+str(len(df))+" rows in sheet")
        return df
    except Exception as e:
        agent_log("ERROR","Scrape failed: "+str(e)); return pd.DataFrame()

def parse_row(row):
    def sf(v, d=0.0):
        try: return float(str(v).replace(",","").strip())
        except: return d
    def si(v, d=0):
        try: return int(float(str(v).strip()))
        except: return d
    ts = str(row.get("Timestamp", datetime.now().isoformat()))
    app_id = ts.replace(" ","_").replace(":","").replace("/","")[:20]
    return {
        "app_id"       : app_id,
        "name"         : str(row.get("Full Name", row.get("Name","Applicant"))),
        "email"        : str(row.get("Email Address", row.get("Email",""))),
        "age"          : si(row.get("Age",34)),
        "annual_income": sf(row.get("Annual Income (Rs)", row.get("Annual Income",900000))),
        "loan_amount"  : sf(row.get("Loan Amount (Rs)", row.get("Loan Amount",500000))),
        "existing_emi" : sf(row.get("Existing EMI (Rs/month)", row.get("Existing EMI",0))),
        "employment"   : str(row.get("Employment Status", row.get("Employment","Employed"))),
        "yrs_employed" : sf(row.get("Years Employed",3)),
        "credit_1"     : sf(row.get("Credit Score 1",0.7)),
        "credit_2"     : sf(row.get("Credit Score 2",0.7)),
        "credit_3"     : sf(row.get("Credit Score 3",0.7)),
        "family"       : si(row.get("Family Members",3)),
    }


# ══════════════════════════════════════════════
# STEP 2: REASON — ML + Rules
# ══════════════════════════════════════════════

def decide(app, model, FN, FM):
    inc=app["annual_income"]; la=app["loan_amount"]; age=app["age"]
    emp=app["employment"].lower() in ["employed","yes","true","1"]
    yrs=app["yrs_employed"]; fam=app["family"]; emi0=app["existing_emi"]
    cs1=app["credit_1"]; cs2=app["credit_2"]; cs3=app["credit_3"]

    mr  = 0.12/12
    emi = (la*mr*(1+mr)**60)/((1+mr)**60-1)
    te  = emi0+emi; mi=inc/12
    ac  = (cs1+cs2+cs3)/3
    dti = te/(mi+0.001)

    feat={
        "EXT_SOURCE_1":cs1,"EXT_SOURCE_2":cs2,"EXT_SOURCE_3":cs3,
        "EXT_SOURCE_MEAN":ac,"DEBT_TO_INCOME":te/(inc+1),
        "LOAN_TO_INCOME":la/(inc+1),"ANNUITY_TO_CREDIT":emi/(la+1),
        "AMT_INCOME_TOTAL":inc,"AMT_CREDIT":la,"AMT_ANNUITY":emi,
        "AGE_YEARS":age,"YEARS_EMPLOYED":yrs,"IS_EMPLOYED":int(emp),
        "INCOME_PER_PERSON":inc/(fam+1),"CNT_FAM_MEMBERS":fam,
    }
    vec  = [feat.get(f,FM.get(f,0)) for f in FN]
    df2  = pd.DataFrame([vec],columns=FN)
    prob = float(model.predict_proba(df2)[0][1])
    pp   = round(prob*100,1)

    rules={
        "Credit Score"   : ac>=0.45,
        "DTI Ratio"      : dti<=0.60,
        "Employment"     : emp,
        "Min. Income"    : inc>=150000,
        "Loan-to-Income" : la/inc<=5.0,
        "Age Range"      : 21<=age<=65,
    }
    failed=[k for k,v in rules.items() if not v]
    cf=any(r in failed for r in ["Employment","Min. Income","Age Range"])
    ir=round(6.5+(4.0 if ac>0.7 else 5.5 if ac>0.55 else 7.0),2)

    if cf or len(failed)>=2 or prob>0.55:
        dec="REJECTED";      aa=0;       ir=None
    elif prob>0.30 or len(failed)>=1:
        dec="MANUAL REVIEW"; aa=la*0.75
    else:
        dec="APPROVED";      aa=la

    return {"decision":dec,"default_prob":pp,"avg_credit":round(ac,3),
            "dti":round(dti*100,1),"emi":round(emi,0),
            "approved_amount":aa,"interest_rate":ir,
            "failed_rules":failed,"creditworthiness":round((1-prob)*100,1)}


# ══════════════════════════════════════════════
# STEP 3: ACT — Send Email
# ══════════════════════════════════════════════

def send_email(to_email, name, result, app, guser, gpass):
    dec=result["decision"]; dp=result["default_prob"]
    ac=result["avg_credit"]; dti=result["dti"]
    emi=result["emi"]; aa=result["approved_amount"]
    ir=result["interest_rate"]; fr=result["failed_rules"]
    cw=result["creditworthiness"]
    ts=datetime.now().strftime("%d %B %Y")

    lines_approved = [
        "Dear "+name+",",
        "",
        "Congratulations! Your loan is APPROVED by our AI Agent.",
        "",
        "DECISION SUMMARY:",
        "  Default Probability : "+str(dp)+"%",
        "  Creditworthiness    : "+str(cw)+"/100",
        "  Avg Credit Score    : "+str(ac),
        "  DTI Ratio           : "+str(dti)+"%",
        "",
        "LOAN OFFER:",
        "  Approved Amount : Rs"+"{:,.0f}".format(aa),
        "  Interest Rate   : "+str(ir)+"% p.a. (RBI 6.5% + spread)",
        "  Monthly EMI     : Rs"+"{:,.0f}".format(emi)+"/month (60 months)",
        "",
        "NEXT STEPS: Visit branch within 30 days with Aadhaar, PAN,",
        "last 3 salary slips, 6-month bank statement.",
        "",
        "This was decided autonomously by our AI Loan Agent.",
        "Date: "+ts,
        "",
        "Warm regards,",
        "AI Loan Processing Agent | National Finance Bank",
    ]
    lines_rejected = [
        "Dear "+name+",",
        "",
        "After AI review, your application could not be approved.",
        "",
        "ASSESSMENT:",
        "  Default Probability : "+str(dp)+"% (limit: 55%)",
        "  Avg Credit Score    : "+str(ac)+" (min: 0.45)",
        "  DTI Ratio           : "+str(dti)+"% (max: 60%)",
        "  Issues              : "+(", ".join(fr) if fr else "high default risk"),
        "",
        "TO IMPROVE:",
        "  1. Pay all EMIs on time for 6 months",
        "  2. Reduce DTI below 60%",
        "  3. Apply for lower amount (max 3x income)",
        "",
        "You may reapply after 6 months.",
        "Date: "+ts,
        "",
        "Warm regards,",
        "AI Loan Processing Agent | National Finance Bank",
    ]
    lines_review = [
        "Dear "+name+",",
        "",
        "Your application needs manual review. NOT a rejection.",
        "",
        "  Default Probability : "+str(dp)+"% (borderline)",
        "  Potential Amount    : Rs"+"{:,.0f}".format(aa),
        "",
        "PLEASE SUBMIT:",
        "  1. Last 6 months bank statements",
        "  2. Latest salary slip or Form 16",
        "  3. Details of all existing loans",
        "",
        "Our team will contact you within 5-7 business days.",
        "Date: "+ts,
        "",
        "Warm regards,",
        "AI Loan Processing Agent | National Finance Bank",
    ]

    if dec=="APPROVED":
        subject="Your Loan Application is APPROVED - National Finance Bank"
        body=chr(10).join(lines_approved)
    elif dec=="REJECTED":
        subject="Loan Application Status - National Finance Bank"
        body=chr(10).join(lines_rejected)
    else:
        subject="Loan Application Under Review - National Finance Bank"
        body=chr(10).join(lines_review)

    try:
        msg=MIMEMultipart()
        msg["From"]=guser; msg["To"]=to_email; msg["Subject"]=subject
        msg.attach(MIMEText(body,"plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(guser,gpass); s.send_message(msg)
        agent_log("EMAIL","Sent "+dec+" to "+to_email)
        return True
    except Exception as e:
        agent_log("ERROR","Email failed: "+str(e)); return False

def run_once(sheet_url, guser, gpass, model, FN, FM):
    agent_log("AGENT","="*40)
    agent_log("AGENT","Cycle started at "+datetime.now().strftime("%H:%M:%S"))
    processed=get_processed_ids()
    found=0; new_proc=0; emails=0

    df=scrape_applications(sheet_url)
    if df.empty:
        log_run(0,0,0,"NO_DATA"); return

    found=len(df)
    for _,row in df.iterrows():
        app=parse_row(row)
        aid=app["app_id"]
        if aid in processed:
            continue
        agent_log("PROCESS","Processing: "+app["name"])
        try:
            result=decide(app,model,FN,FM)
            agent_log("DECIDE",app["name"]+" -> "+result["decision"]+" ("+str(result["default_prob"])+"% risk)")
        except Exception as e:
            agent_log("ERROR","Decision failed: "+str(e)); continue

        log_decision(aid,app["name"],app["email"],result["decision"],
                     result["default_prob"],app["loan_amount"],
                     result["approved_amount"],result["interest_rate"],result["emi"])

        if app["email"] and "@" in app["email"] and guser and gpass:
            if send_email(app["email"],app["name"],result,app,guser,gpass):
                mark_email_sent(aid); emails+=1
        else:
            agent_log("EMAIL","No email config -- decision logged only")

        save_processed_id(aid); new_proc+=1

    log_run(found,new_proc,emails,"SUCCESS")
    agent_log("AGENT","Done: "+str(new_proc)+" new | "+str(emails)+" emails | next in 60 min")

def start_loop(sheet_url, guser, gpass, model, FN, FM):
    def loop():
        while st.session_state.get("running",False):
            run_once(sheet_url,guser,gpass,model,FN,FM)
            st.session_state["last_run"]=datetime.now().strftime("%H:%M:%S")
            time.sleep(AGENT_INTERVAL)
    threading.Thread(target=loop,daemon=True).start()


# ══════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════

@st.cache_resource
def load_model():
    for p in [".","/app"]:
        mf=os.path.join(p,"loan_model.json")
        ff=os.path.join(p,"feature_names.pkl")
        fm=os.path.join(p,"feature_medians.pkl")
        if os.path.exists(mf) and os.path.exists(ff) and os.path.exists(fm):
            m=xgb.XGBClassifier(); m.load_model(mf)
            with open(ff,"rb") as f: fn=pickle.load(f)
            with open(fm,"rb") as f: fmed=pickle.load(f)
            return m,fn,fmed
    raise FileNotFoundError("Model files not found! CWD="+os.getcwd()+" Files="+str(os.listdir(".")))

init_db()
try:
    model,FN,FM=load_model(); model_ok=True
except Exception as e:
    model_ok=False; model_err=str(e)

for k,v in [("running",False),("last_run","Never"),("agent_log",[])]:
    if k not in st.session_state: st.session_state[k]=v


# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════

st.markdown("""
<div class="hero">
<h1>🤖 Autonomous Loan Approval Agent</h1>
<p>PERCEIVE &rarr; REASON &rarr; ACT &rarr; REMEMBER &rarr; REPEAT &nbsp;&nbsp;|&nbsp;&nbsp; Every 60 minutes</p>
</div>""", unsafe_allow_html=True)

if not model_ok:
    st.error("Model not loaded: "+model_err)
    st.info("Files: "+str(os.listdir("."))); st.stop()

# Credentials loaded from secrets
creds_ok = bool(GMAIL_USER and GMAIL_PASS and SHEET_URL)
sheet_url  = SHEET_URL
guser      = GMAIL_USER
gpass      = GMAIL_PASS

# ── STATS
stats=get_stats()
s1,s2,s3,s4,s5=st.columns(5)
s1.metric("Total Processed", stats["total"])
s2.metric("Approved",        stats["approved"])
s3.metric("Rejected",        stats["rejected"])
s4.metric("Under Review",    stats["review"])
s5.metric("Emails Sent",     stats["emails"])
st.divider()

# ── MAIN LAYOUT
left,right=st.columns([1,1])

with left:
    st.markdown("### ⚙️ Agent Control Panel")

    if creds_ok:
        st.success("Credentials loaded from Secrets")
        st.info("Sheet: ..."+sheet_url[-40:])
        st.info("Gmail: "+guser)
    else:
        st.warning("Credentials not found in Secrets. Enter manually:")
        sheet_url = st.text_input("Google Sheet CSV URL", SHEET_URL)
        guser     = st.text_input("Gmail", GMAIL_USER)
        gpass     = st.text_input("App Password", GMAIL_PASS, type="password")

    st.markdown("")
    b1,b2=st.columns(2)
    with b1:
        if st.button("▶️ Run Once Now"):
            with st.spinner("Agent running..."):
                run_once(sheet_url,guser,gpass,model,FN,FM)
            st.success("Cycle complete!"); st.rerun()
    with b2:
        if not st.session_state.running:
            if st.button("🚀 Start Auto (60 min)"):
                st.session_state.running=True
                start_loop(sheet_url,guser,gpass,model,FN,FM)
                st.success("Agent started!"); st.rerun()
        else:
            if st.button("⏹️ Stop Agent"):
                st.session_state.running=False
                st.warning("Agent stopped."); st.rerun()

    if st.session_state.running:
        st.markdown('<div class="running"><b style="color:#10b981">🟢 AGENT RUNNING</b> &nbsp;|&nbsp; Last: '+st.session_state.last_run+'</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="stopped"><b style="color:#64748b">⚪ Agent stopped</b></div>',unsafe_allow_html=True)

with right:
    st.markdown("### 🔄 How the Agent Works")
    st.markdown("""
    <div class="acard"><h4>1. 🌐 PERCEIVE — Web Scraping</h4>
    <p>Scrapes loan applications from your Google Sheet every 60 minutes automatically. Each row = one application.</p></div>
    <div class="acard"><h4>2. 🧠 REASON — XGBoost ML + 6 Rules</h4>
    <p>Model predicts default probability. 6 credit rules check income, DTI, age, employment, credit score.</p></div>
    <div class="acard"><h4>3. 📧 ACT — Send Decision Email</h4>
    <p>Automatically emails each applicant their full decision letter: APPROVED / REJECTED / MANUAL REVIEW.</p></div>
    <div class="acard"><h4>4. 💾 REMEMBER — SQLite Database</h4>
    <p>All decisions stored permanently. Processed IDs tracked so no application is processed twice.</p></div>
    <div class="acard"><h4>5. 😴 REPEAT — Every 60 Minutes</h4>
    <p>Sleeps 60 minutes then loops back to Step 1. Fully autonomous — zero human intervention needed.</p></div>
    """, unsafe_allow_html=True)

st.divider()

# ── ACTIVITY LOG
st.markdown("### 📋 Agent Activity Log")
lines=get_log_lines()
if lines:
    html=""
    for l in reversed(lines):
        l=l.strip()
        if   "[DECIDE]" in l and "APPROVED"      in l: c="#10b981"
        elif "[DECIDE]" in l and "REJECTED"      in l: c="#ef4444"
        elif "[DECIDE]" in l and "MANUAL REVIEW" in l: c="#f59e0b"
        elif "[ERROR]"  in l: c="#ef4444"
        elif "[EMAIL]"  in l: c="#60a5fa"
        elif "[AGENT]"  in l: c="#00d4aa"
        else: c="#94a3b8"
        html+="<span style='color:"+c+"'>"+l+"</span><br>"
    st.markdown('<div class="log-box">'+html+'</div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="log-box">No activity yet. Click Run Once Now to test.</div>',unsafe_allow_html=True)

if st.button("🔄 Refresh"): st.rerun()
st.divider()

# ── DECISIONS TABLE
st.markdown("### 📊 All Decisions (Agent Memory)")
df_dec=get_decisions()
if not df_dec.empty:
    def col_dec(v):
        if v=="APPROVED":     return "color:#10b981;font-weight:bold"
        elif v=="REJECTED":   return "color:#ef4444;font-weight:bold"
        return "color:#f59e0b;font-weight:bold"
    cols=[c for c in ["app_id","applicant","email","decision","default_prob",
                      "loan_amount","approved_amt","email_sent","timestamp"] if c in df_dec.columns]
    st.dataframe(df_dec[cols].style.applymap(col_dec,subset=["decision"]),use_container_width=True)
    st.download_button("📥 Download CSV",data=df_dec.to_csv(index=False),
                       file_name="decisions.csv",mime="text/csv")
else:
    st.info("No decisions yet. Run the agent to start.")

st.divider()
st.markdown("### 🏃 Agent Run History")
df_r=get_runs()
if not df_r.empty: st.dataframe(df_r,use_container_width=True)
else: st.info("No runs yet.")
st.divider()
st.caption("Autonomous Loan Agent v3.0 | "+datetime.now().strftime("%Y-%m-%d %H:%M")+" | Perceive->Reason->Act->Remember->Repeat")
