# -*- coding: utf-8 -*-
"""
AUTONOMOUS LOAN APPROVAL AGENT v4.0
=====================================
9 Agentic Improvements:
1. Adaptive Learning    - Auto-retrains when accuracy drops
2. SHAP PDF Email       - Explainable AI in email attachment  
3. Daily Summary Email  - Manager report every 9AM
4. Multi-Source         - Google Sheet + CSV upload + API
5. Fraud Detection      - Flags suspicious patterns
6. Follow-up Agent      - Auto follow-up for manual review
7. WhatsApp Alerts      - Twilio WhatsApp notifications
8. Live Dashboard       - Auto-refresh charts
9. RBI Rate Scraping    - Live policy rate updates
"""

import os, json, pickle, sqlite3, smtplib, threading, time, logging, io, re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import inch
    PDF_OK = True
except Exception:
    PDF_OK = False

# ── Credentials from environment
GMAIL_USER   = os.environ.get("GMAIL_USER", "")
GMAIL_PASS   = os.environ.get("GMAIL_PASS", "")
SHEET_URL    = os.environ.get("SHEET_URL", "")
TWILIO_SID   = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_FROM", "")

DB_FILE        = "agent_memory.db"
LOG_FILE       = "agent_log.txt"
PROCESSED_FILE = "processed_ids.json"
FOLLOWUP_FILE  = "followup_tracker.json"
MODEL_FILE     = "loan_model.json"
INTERVAL       = 3600

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s,%(msecs)03d [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

st.set_page_config(page_title="Autonomous Loan Agent v4", page_icon="🤖", layout="wide")

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
.fraud{background:linear-gradient(135deg,#450a0a,#7f1d1d);border:1px solid #ef4444;border-radius:10px;padding:.8rem 1rem;margin:.3rem 0;}
.log-box{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:1rem;font-family:"DM Mono",monospace;font-size:.78rem;color:#94a3b8;height:300px;overflow-y:auto;}
.running{background:linear-gradient(135deg,#052e16,#064e3b);border:1px solid #10b981;border-radius:10px;padding:.8rem 1rem;}
.stopped{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:.8rem 1rem;}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0099ff);color:white;border:none;padding:.6rem 1.5rem;font-weight:600;border-radius:10px;}
.metric-card{background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:12px;padding:1rem;text-align:center;}
.metric-card .val{color:#00d4aa;font-size:2rem;font-weight:700;}
.metric-card .lbl{color:#64748b;font-size:.8rem;text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_id TEXT UNIQUE, applicant TEXT, email TEXT, phone TEXT,
        decision TEXT, default_prob REAL, creditworth REAL,
        loan_amount REAL, approved_amt REAL, interest_rate REAL,
        emi REAL, email_sent INTEGER DEFAULT 0, whatsapp_sent INTEGER DEFAULT 0,
        fraud_flag INTEGER DEFAULT 0, fraud_reason TEXT DEFAULT '',
        source TEXT DEFAULT 'google_sheet', timestamp TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS agent_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_time TEXT, apps_found INTEGER, processed INTEGER,
        emails_sent INTEGER, fraud_flagged INTEGER, status TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_id TEXT, correct INTEGER, timestamp TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT, auc REAL, trained_on INTEGER, timestamp TEXT)""")
    conn.commit(); conn.close()

def log_decision(app_id, name, email, phone, dec, prob, la, aa, ir, emi, fraud, fraud_reason, source):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""INSERT OR IGNORE INTO decisions
            (app_id,applicant,email,phone,decision,default_prob,creditworth,
             loan_amount,approved_amt,interest_rate,emi,fraud_flag,fraud_reason,source,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (app_id,name,email,phone,dec,float(prob),float(100-prob),
             float(la) if la else 0.0,float(aa) if aa else 0.0,float(ir) if ir else 0.0,float(emi) if emi else 0.0,
             int(fraud),fraud_reason,source,datetime.now().isoformat()))
        conn.commit()
    except Exception: pass
    conn.close()

def get_decisions(limit=100):
    conn = sqlite3.connect(DB_FILE)
    try: df = pd.read_sql(f"SELECT * FROM decisions ORDER BY timestamp DESC LIMIT {limit}", conn)
    except Exception: df = pd.DataFrame()
    conn.close(); return df

def get_stats():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    s = {"total":0,"approved":0,"rejected":0,"review":0,"emails":0,"fraud":0}
    try:
        for k,q in [("total","SELECT COUNT(*) FROM decisions"),
                    ("approved","SELECT COUNT(*) FROM decisions WHERE decision='APPROVED'"),
                    ("rejected","SELECT COUNT(*) FROM decisions WHERE decision='REJECTED'"),
                    ("review","SELECT COUNT(*) FROM decisions WHERE decision='MANUAL REVIEW'"),
                    ("emails","SELECT COUNT(*) FROM decisions WHERE email_sent=1"),
                    ("fraud","SELECT COUNT(*) FROM decisions WHERE fraud_flag=1")]:
            c.execute(q); s[k] = c.fetchone()[0]
    except Exception: pass
    conn.close(); return s

def get_daily_stats():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("""
            SELECT DATE(timestamp) as date,
                   COUNT(*) as total,
                   SUM(CASE WHEN decision='APPROVED' THEN 1 ELSE 0 END) as approved,
                   SUM(CASE WHEN decision='REJECTED' THEN 1 ELSE 0 END) as rejected,
                   AVG(default_prob) as avg_risk
            FROM decisions GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 30
        """, conn)
    except Exception: df = pd.DataFrame()
    conn.close(); return df

def get_processed_ids():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f: return set(json.load(f))
    return set()

def save_processed_id(aid):
    ids = get_processed_ids(); ids.add(aid)
    with open(PROCESSED_FILE,"w") as f: json.dump(list(ids), f)

def get_followups():
    if os.path.exists(FOLLOWUP_FILE):
        with open(FOLLOWUP_FILE) as f: return json.load(f)
    return {}

def save_followup(aid, data):
    fu = get_followups(); fu[aid] = data
    with open(FOLLOWUP_FILE,"w") as f: json.dump(fu, f)


# ═══════════════════════════════════════════
# AGENT LOG
# ═══════════════════════════════════════════

def alog(action, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = "["+ts+"] ["+action+"] "+msg
    logging.info(line)
    if "alog" not in st.session_state: st.session_state.alog = []
    st.session_state.alog.append(line)
    if len(st.session_state.alog) > 500: st.session_state.alog = st.session_state.alog[-500:]

def get_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f: return f.readlines()[-80:]
    return []


# ═══════════════════════════════════════════
# IMPROVEMENT 9: RBI Rate Scraping
# ═══════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_rbi_rate():
    try:
        r = requests.get("https://api.rbi.org.in/api/v1/keyrates", timeout=5)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("data", []):
                if "repo" in str(item).lower():
                    rate = float(re.findall(r'\d+\.?\d*', str(item.get("rate","6.5")))[0])
                    alog("RBI","Live repo rate: "+str(rate)+"%")
                    return rate
    except Exception:
        pass
    # Fallback: scrape RBI website
    try:
        r = requests.get("https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx", timeout=5)
        matches = re.findall(r'repo rate.*?(\d+\.?\d+)%', r.text.lower())
        if matches:
            rate = float(matches[0])
            alog("RBI","Scraped repo rate: "+str(rate)+"%")
            return rate
    except Exception:
        pass
    alog("RBI","Using fallback rate: 6.5%")
    return 6.50


# ═══════════════════════════════════════════
# IMPROVEMENT 5: Fraud Detection
# ═══════════════════════════════════════════

def check_fraud(app, all_apps_df):
    flags = []
    # Rule 1: Identical credit scores (bot submission)
    cs1, cs2, cs3 = app["credit_1"], app["credit_2"], app["credit_3"]
    if cs1 == cs2 == cs3:
        flags.append("Identical credit scores across all 3 bureaus")
    # Rule 2: Income too round (suspicious)
    if app["annual_income"] % 100000 == 0 and app["annual_income"] > 0:
        pass  # common, don't flag
    # Rule 3: Multiple submissions same email
    if not all_apps_df.empty and "email" in all_apps_df.columns:
        email_count = len(all_apps_df[all_apps_df["email"] == app["email"]])
        if email_count > 3:
            flags.append("Same email submitted "+str(email_count)+" times")
    # Rule 4: Unrealistic credit scores
    if cs1 > 0.99 or cs2 > 0.99 or cs3 > 0.99:
        flags.append("Suspiciously perfect credit score (>0.99)")
    # Rule 5: Age-income mismatch
    if app["age"] < 25 and app["annual_income"] > 5000000:
        flags.append("Age "+str(app["age"])+" with income Rs"+str(app["annual_income"])+" is unusual")
    # Rule 6: Loan > 10x income
    if app["loan_amount"] > app["annual_income"] * 10:
        flags.append("Loan amount is "+str(round(app["loan_amount"]/app["annual_income"],1))+"x annual income")

    is_fraud = len(flags) > 0
    reason = " | ".join(flags) if flags else ""
    if is_fraud:
        alog("FRAUD","FLAGGED: "+app["name"]+" -- "+reason)
    return is_fraud, reason


# ═══════════════════════════════════════════
# IMPROVEMENT 2: SHAP PDF Generation
# ═══════════════════════════════════════════

def generate_shap_pdf(app, result, shap_values, feature_names):
    buf = io.BytesIO()
    try:
        if PDF_OK:
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    rightMargin=40, leftMargin=40,
                                    topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle('title', parent=styles['Title'],
                                          textColor=colors.HexColor('#0f2027'),
                                          fontSize=18, spaceAfter=6)
            story.append(Paragraph("National Finance Bank", title_style))
            story.append(Paragraph("AI Loan Decision Report", title_style))
            story.append(HRFlowable(width="100%", thickness=2,
                                     color=colors.HexColor('#00d4aa')))
            story.append(Spacer(1, 12))

            # Decision banner
            dec = result["decision"]
            dec_color = colors.HexColor('#10b981') if dec=="APPROVED" else \
                        colors.HexColor('#ef4444') if dec=="REJECTED" else \
                        colors.HexColor('#f59e0b')
            dec_style = ParagraphStyle('dec', parent=styles['Heading1'],
                                        textColor=dec_color, fontSize=22,
                                        spaceAfter=6)
            story.append(Paragraph("DECISION: "+dec, dec_style))
            story.append(Spacer(1, 8))

            # Applicant info table
            info_data = [
                ["Applicant", app["name"], "Date", datetime.now().strftime("%d %B %Y")],
                ["Loan Amount", "Rs{:,.0f}".format(app["loan_amount"]),
                 "Default Risk", str(result["default_prob"])+"%"],
                ["Credit Score", str(result["avg_credit"]),
                 "DTI Ratio", str(result["dti"])+"%"],
            ]
            if dec == "APPROVED":
                info_data.append(["Approved Amt", "Rs{:,.0f}".format(result["approved_amount"]),
                                  "Interest Rate", str(result["interest_rate"])+"% p.a."])
                info_data.append(["Monthly EMI", "Rs{:,.0f}".format(result["emi"]),
                                  "Tenure", "60 months"])

            t = Table(info_data, colWidths=[1.4*inch,2*inch,1.4*inch,2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
                ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#64748b')),
                ('TEXTCOLOR', (2,0), (2,-1), colors.HexColor('#64748b')),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 16))

            # SHAP chart
            story.append(Paragraph("AI Explanation — Why This Decision?",
                                    ParagraphStyle('h2', parent=styles['Heading2'],
                                                   textColor=colors.HexColor('#0f2027'),
                                                   fontSize=13, spaceAfter=8)))

            if shap_values is not None:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                sv = np.array(shap_values)
                idx = np.argsort(np.abs(sv))[-8:]
                labels_map = {
                    "EXT_SOURCE_MEAN":"Avg Credit Score","EXT_SOURCE_1":"Credit Score 1",
                    "EXT_SOURCE_2":"Credit Score 2","EXT_SOURCE_3":"Credit Score 3",
                    "DEBT_TO_INCOME":"Debt/Income","LOAN_TO_INCOME":"Loan/Income",
                    "AMT_INCOME_TOTAL":"Annual Income","AMT_CREDIT":"Loan Amount",
                    "AGE_YEARS":"Age","YEARS_EMPLOYED":"Years Employed",
                    "IS_EMPLOYED":"Employment Status","INCOME_PER_PERSON":"Income/Person",
                    "CNT_FAM_MEMBERS":"Family Size","AMT_ANNUITY":"Monthly EMI",
                }
                fn_arr = np.array(feature_names)
                vals   = sv[idx]
                labs   = [labels_map.get(f,f) for f in fn_arr[idx]]
                clrs   = ["#ef4444" if v>0 else "#10b981" for v in vals]
                ax.barh(range(len(vals)), vals, color=clrs, edgecolor="none", height=0.6)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(labs, fontsize=9)
                ax.axvline(0, color="#94a3b8", lw=0.8, linestyle="--")
                ax.set_xlabel("Impact on Default Risk  (red=increases risk, green=reduces risk)", fontsize=8)
                ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
                for sp in ["top","right"]: ax.spines[sp].set_visible(False)
                plt.tight_layout()
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                img_buf.seek(0)
                from reportlab.platypus import Image as RLImage
                story.append(RLImage(img_buf, width=6*inch, height=3*inch))
            else:
                story.append(Paragraph("SHAP library not available for chart generation.",
                                        styles['Normal']))

            story.append(Spacer(1, 12))

            # Top reasons
            if shap_values is not None:
                sv = np.array(shap_values)
                idx = np.argsort(np.abs(sv))[-5:][::-1]
                fn_arr = np.array(feature_names)
                labels_map = {
                    "EXT_SOURCE_MEAN":"Avg Credit Score","EXT_SOURCE_1":"Credit Score 1",
                    "EXT_SOURCE_2":"Credit Score 2","EXT_SOURCE_3":"Credit Score 3",
                    "DEBT_TO_INCOME":"Debt-to-Income Ratio","LOAN_TO_INCOME":"Loan-to-Income",
                    "AMT_INCOME_TOTAL":"Annual Income","AMT_CREDIT":"Loan Amount",
                    "AGE_YEARS":"Age","YEARS_EMPLOYED":"Years Employed",
                    "IS_EMPLOYED":"Employment Status",
                }
                story.append(Paragraph("Top 5 Decision Factors:",
                                        ParagraphStyle('h3', parent=styles['Heading3'],
                                                       textColor=colors.HexColor('#0f2027'),
                                                       fontSize=11, spaceAfter=4)))
                for rank, i in enumerate(idx, 1):
                    fname = labels_map.get(fn_arr[i], fn_arr[i])
                    impact = sv[i]
                    direction = "INCREASES" if impact > 0 else "REDUCES"
                    dir_color = colors.HexColor('#ef4444') if impact > 0 else colors.HexColor('#10b981')
                    p = Paragraph(
                        str(rank)+". <b>"+fname+"</b> — impact: "
                        "<font color='"+("#ef4444" if impact>0 else "#10b981")+"'>"
                        +direction+" default risk by "+str(round(abs(impact),3))+"</font>",
                        ParagraphStyle('factor', parent=styles['Normal'], fontSize=10,
                                        spaceAfter=3, leftIndent=10))
                    story.append(p)

            story.append(Spacer(1, 12))
            story.append(HRFlowable(width="100%", thickness=1,
                                     color=colors.HexColor('#e2e8f0')))
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "This report was generated autonomously by the National Finance Bank AI Agent. "
                "Ref: "+app.get("app_id","N/A")+" | "+datetime.now().strftime("%d %B %Y %H:%M"),
                ParagraphStyle('footer', parent=styles['Normal'],
                                fontSize=8, textColor=colors.HexColor('#94a3b8'))))

            doc.build(story)
        else:
            # Fallback: matplotlib PDF
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.95, "National Finance Bank", ha='center', va='top',
                    fontsize=16, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.85, "AI Loan Decision Report", ha='center', va='top',
                    fontsize=13, transform=ax.transAxes)
            dec = result["decision"]
            color = "#10b981" if dec=="APPROVED" else "#ef4444" if dec=="REJECTED" else "#f59e0b"
            ax.text(0.5, 0.72, dec, ha='center', va='top', fontsize=20,
                    fontweight='bold', color=color, transform=ax.transAxes)
            details = [
                "Applicant: "+app["name"],
                "Default Risk: "+str(result["default_prob"])+"%",
                "Credit Score: "+str(result["avg_credit"]),
                "DTI Ratio: "+str(result["dti"])+"%",
            ]
            for i, d in enumerate(details):
                ax.text(0.1, 0.58-i*0.08, d, fontsize=11, transform=ax.transAxes)
            ax.axis('off')
            fig.savefig(buf, format='pdf', bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        alog("ERROR","PDF generation failed: "+str(e))

    buf.seek(0)
    return buf


# ═══════════════════════════════════════════
# IMPROVEMENT 1: PERCEIVE — Multi-Source
# ═══════════════════════════════════════════

def scrape_google_sheet(url):
    try:
        if "docs.google.com/spreadsheets" in url:
            if "/edit" in url:
                url = url.split("/edit")[0]+"/export?format=csv"
            elif "export?format=csv" not in url:
                url = url+"/export?format=csv"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df["_source"] = "google_sheet"
            alog("PERCEIVE","Google Sheet: "+str(len(df))+" rows")
            return df
    except Exception as e:
        alog("ERROR","Sheet scrape failed: "+str(e))
    return pd.DataFrame()

def scrape_csv_url(csv_url):
    if not csv_url: return pd.DataFrame()
    try:
        r = requests.get(csv_url, timeout=10)
        df = pd.read_csv(io.StringIO(r.text))
        df["_source"] = "csv_url"
        alog("PERCEIVE","CSV URL: "+str(len(df))+" rows")
        return df
    except Exception as e:
        alog("ERROR","CSV URL failed: "+str(e))
    return pd.DataFrame()

def load_uploaded_csv(uploaded_file):
    if uploaded_file is None: return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file)
        df["_source"] = "csv_upload"
        alog("PERCEIVE","Uploaded CSV: "+str(len(df))+" rows")
        return df
    except Exception as e:
        alog("ERROR","CSV upload failed: "+str(e))
    return pd.DataFrame()

def merge_sources(*dfs):
    valid = [df for df in dfs if not df.empty]
    if not valid: return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True)
    alog("PERCEIVE","Merged "+str(len(merged))+" total applications from "+str(len(valid))+" sources")
    return merged


# ═══════════════════════════════════════════
# IMPROVEMENT 2: REASON
# ═══════════════════════════════════════════

def parse_row(row):
    def sf(v,d=0.0):
        try: return float(str(v).replace(",","").strip())
        except: return d
    def si(v,d=0):
        try: return int(float(str(v).strip()))
        except: return d
    ts  = str(row.get("Timestamp", datetime.now().isoformat()))
    aid = ts.replace(" ","_").replace(":","").replace("/","")[:20]
    return {
        "app_id"       : aid,
        "name"         : str(row.get("Full Name", row.get("Name","Applicant"))),
        "email"        : str(row.get("Email Address", row.get("Email",""))),
        "phone"        : str(row.get("Phone", row.get("Mobile",""))),
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
        "_source"      : str(row.get("_source","google_sheet")),
    }

def decide(app, model, FN, FM, repo_rate=6.50):
    inc=app["annual_income"]; la=app["loan_amount"]; age=app["age"]
    emp=app["employment"].lower() in ["employed","yes","true","1"]
    yrs=app["yrs_employed"]; fam=app["family"]; emi0=app["existing_emi"]
    cs1=app["credit_1"]; cs2=app["credit_2"]; cs3=app["credit_3"]

    mr=0.12/12; emi=(la*mr*(1+mr)**60)/((1+mr)**60-1)
    te=emi0+emi; mi=inc/12; ac=(cs1+cs2+cs3)/3; dti=te/(mi+0.001)
    feat={
        "EXT_SOURCE_1":cs1,"EXT_SOURCE_2":cs2,"EXT_SOURCE_3":cs3,"EXT_SOURCE_MEAN":ac,
        "DEBT_TO_INCOME":te/(inc+1),"LOAN_TO_INCOME":la/(inc+1),"ANNUITY_TO_CREDIT":emi/(la+1),
        "AMT_INCOME_TOTAL":inc,"AMT_CREDIT":la,"AMT_ANNUITY":emi,
        "AGE_YEARS":age,"YEARS_EMPLOYED":yrs,"IS_EMPLOYED":int(emp),
        "INCOME_PER_PERSON":inc/(fam+1),"CNT_FAM_MEMBERS":fam,
    }
    vec=[feat.get(f,FM.get(f,0)) for f in FN]
    df2=pd.DataFrame([vec],columns=FN)
    prob=float(model.predict_proba(df2)[0][1]); pp=round(prob*100,1)
    rules={"Credit Score":ac>=0.45,"DTI Ratio":dti<=0.60,"Employment":emp,
           "Min. Income":inc>=150000,"Loan-to-Income":la/inc<=5.0,"Age Range":21<=age<=65}
    failed=[k for k,v in rules.items() if not v]
    cf=any(r in failed for r in ["Employment","Min. Income","Age Range"])
    sp=4.0 if ac>0.7 else 5.5 if ac>0.55 else 7.0
    ir=round(repo_rate+sp,2)
    if cf or len(failed)>=2 or prob>0.55:
        dec="REJECTED"; aa=0; ir=None
    elif prob>0.30 or len(failed)>=1:
        dec="MANUAL REVIEW"; aa=la*0.75
    else:
        dec="APPROVED"; aa=la

    # SHAP values
    shap_vals = None
    if SHAP_OK:
        try:
            bg=pd.DataFrame([FM])[FN]
            explainer=shap.TreeExplainer(model,bg)
            shap_vals=explainer.shap_values(df2)[0]
        except Exception: pass

    return {"decision":dec,"default_prob":pp,"avg_credit":round(ac,3),
            "dti":round(dti*100,1),"emi":round(emi,0),
            "approved_amount":aa,"interest_rate":ir,
            "failed_rules":failed,"creditworthiness":round((1-prob)*100,1),
            "shap_values":shap_vals}


# ═══════════════════════════════════════════
# IMPROVEMENT 3: ACT — Send Email with PDF
# ═══════════════════════════════════════════

def send_email(to_email, name, result, app, guser, gpass, attach_pdf=True, feature_names=None):
    dec=result["decision"]; dp=result["default_prob"]
    ac=result["avg_credit"]; dti=result["dti"]
    emi=result["emi"]; aa=result["approved_amount"]
    ir=result["interest_rate"]; fr=result["failed_rules"]
    cw=result["creditworthiness"]; ts=datetime.now().strftime("%d %B %Y")

    # Top SHAP reasons for email body
    shap_section = ""
    if result.get("shap_values") is not None and feature_names:
        sv=np.array(result["shap_values"])
        idx=np.argsort(np.abs(sv))[-3:][::-1]
        fn_arr=np.array(feature_names)
        labels={"EXT_SOURCE_MEAN":"Avg Credit Score","EXT_SOURCE_1":"Credit 1",
                "EXT_SOURCE_2":"Credit 2","EXT_SOURCE_3":"Credit 3",
                "DEBT_TO_INCOME":"Debt/Income","LOAN_TO_INCOME":"Loan/Income",
                "AGE_YEARS":"Age","YEARS_EMPLOYED":"Yrs Employed","IS_EMPLOYED":"Employment",
                "AMT_INCOME_TOTAL":"Income","AMT_CREDIT":"Loan Amt"}
        shap_lines = ["", "TOP DECISION FACTORS (AI Explanation):"]
        for rank, i in enumerate(idx, 1):
            fname=labels.get(fn_arr[i],fn_arr[i])
            direction="increases" if sv[i]>0 else "reduces"
            shap_lines.append("  "+str(rank)+". "+fname+" -- "+direction+" risk by "+str(round(abs(sv[i]),3)))
        shap_section = chr(10).join(shap_lines)

    lines_approved = [
        "Dear "+name+",",
        "",
        "Congratulations! Your loan application is APPROVED by our AI Agent.",
        "",
        "DECISION SUMMARY:",
        "  Default Probability : "+str(dp)+"%",
        "  Creditworthiness    : "+str(cw)+"/100",
        "  Avg Credit Score    : "+str(ac),
        "  DTI Ratio           : "+str(dti)+"%",
        "",
        "LOAN OFFER:",
        "  Approved Amount : Rs"+"{:,.0f}".format(aa),
        "  Interest Rate   : "+str(ir)+"% p.a. (RBI repo rate + spread)",
        "  Monthly EMI     : Rs"+"{:,.0f}".format(emi)+"/month (60 months)",
    ] + shap_section.splitlines() + [
        "",
        "NEXT STEPS: Visit branch within 30 days with Aadhaar, PAN,",
        "last 3 salary slips, and 6-month bank statement.",
        "",
        "A detailed AI explanation report is attached as PDF.",
        "Date: "+ts,
        "",
        "Warm regards,",
        "AI Loan Processing Agent | National Finance Bank",
    ]

    lines_rejected = [
        "Dear "+name+",",
        "",
        "After AI review, your application could not be approved at this time.",
        "",
        "ASSESSMENT:",
        "  Default Probability : "+str(dp)+"% (our limit: 55%)",
        "  Avg Credit Score    : "+str(ac)+" (minimum: 0.45)",
        "  DTI Ratio           : "+str(dti)+"% (maximum: 60%)",
        "  Issues Found        : "+(", ".join(fr) if fr else "high default probability"),
    ] + shap_section.splitlines() + [
        "",
        "HOW TO IMPROVE:",
        "  1. Pay all EMIs on time for 6+ months",
        "  2. Reduce total debt-to-income below 60%",
        "  3. Apply for a lower amount (max 3x annual income)",
        "  4. Maintain stable employment",
        "",
        "You may reapply after 6 months.",
        "A detailed AI explanation is attached as PDF.",
        "Date: "+ts,
        "",
        "Warm regards,",
        "AI Loan Processing Agent | National Finance Bank",
    ]

    lines_review = [
        "Dear "+name+",",
        "",
        "Your application requires manual review. This is NOT a rejection.",
        "",
        "  Default Probability : "+str(dp)+"% (borderline case)",
        "  Potential Amount    : Rs"+"{:,.0f}".format(aa),
    ] + shap_section.splitlines() + [
        "",
        "DOCUMENTS REQUIRED:",
        "  1. Last 6 months bank statements",
        "  2. Latest salary slip or Form 16",
        "  3. Details of all existing loans",
        "",
        "Our team will contact you within 5-7 business days.",
        "A detailed AI explanation is attached as PDF.",
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

        # Attach PDF report
        if attach_pdf:
            pdf_buf = generate_shap_pdf(app, result,
                                         result.get("shap_values"),
                                         feature_names)
            if pdf_buf.getvalue():
                part=MIMEBase("application","octet-stream")
                part.set_payload(pdf_buf.read())
                encoders.encode_base64(part)
                fname="loan_report_"+app.get("app_id","report")+".pdf"
                part.add_header("Content-Disposition","attachment",filename=fname)
                msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(guser,gpass); s.send_message(msg)
        alog("EMAIL","Sent "+dec+" (+PDF) to "+to_email)
        return True
    except Exception as e:
        alog("ERROR","Email failed: "+str(e)); return False


# ═══════════════════════════════════════════
# IMPROVEMENT 7: WhatsApp (Twilio)
# ═══════════════════════════════════════════

def send_whatsapp(phone, name, decision, amount, emi, twilio_sid, twilio_token, twilio_from):
    if not all([twilio_sid, twilio_token, twilio_from, phone]): return False
    try:
        from twilio.rest import Client
        client = Client(twilio_sid, twilio_token)
        if decision=="APPROVED":
            msg=("Dear "+name+", Your loan of Rs"+"{:,.0f}".format(amount)+
                 " is APPROVED! EMI: Rs"+"{:,.0f}".format(emi)+
                 "/month. Visit branch within 30 days. - National Finance Bank AI Agent")
        elif decision=="REJECTED":
            msg=("Dear "+name+", Your loan application was not approved. "
                 "Please check your email for detailed AI explanation and improvement tips. "
                 "- National Finance Bank AI Agent")
        else:
            msg=("Dear "+name+", Your loan application is under review. "
                 "Please check email and submit required documents. "
                 "- National Finance Bank AI Agent")
        to_num = "whatsapp:"+phone if not phone.startswith("whatsapp:") else phone
        client.messages.create(body=msg, from_=twilio_from, to=to_num)
        alog("WHATSAPP","Sent to "+phone)
        return True
    except Exception as e:
        alog("ERROR","WhatsApp failed: "+str(e)); return False


# ═══════════════════════════════════════════
# IMPROVEMENT 6: Follow-up Agent
# ═══════════════════════════════════════════

def run_followup_agent(guser, gpass):
    followups = get_followups()
    today = datetime.now().date()
    alog("FOLLOWUP","Checking "+str(len(followups))+" pending reviews")
    for aid, fu in followups.items():
        if fu.get("resolved"): continue
        submitted = datetime.fromisoformat(fu["submitted"]).date()
        days_elapsed = (today - submitted).days
        email = fu.get("email","")
        name  = fu.get("name","Applicant")

        if days_elapsed >= 8 and not fu.get("auto_rejected"):
            # Auto-reject
            if email and guser and gpass:
                lines = [
                    "Dear "+name+",",
                    "",
                    "Your loan application (submitted "+str(submitted)+") has been AUTO-REJECTED.",
                    "Reason: Required documents were not submitted within 7 days.",
                    "",
                    "You may reapply with all required documents.",
                    "",
                    "Warm regards,",
                    "AI Loan Processing Agent | National Finance Bank",
                ]
                try:
                    msg=MIMEMultipart()
                    msg["From"]=guser; msg["To"]=email
                    msg["Subject"]="Loan Application Auto-Rejected - National Finance Bank"
                    msg.attach(MIMEText(chr(10).join(lines),"plain"))
                    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
                        s.login(guser,gpass); s.send_message(msg)
                    alog("FOLLOWUP","Auto-rejected "+name+" ("+str(days_elapsed)+" days)")
                except Exception as e:
                    alog("ERROR","Follow-up email failed: "+str(e))
            fu["auto_rejected"] = True; fu["resolved"] = True
            save_followup(aid, fu)

        elif days_elapsed == 7 and not fu.get("day7_sent"):
            reminder = "final"
            fu["day7_sent"] = True
        elif days_elapsed == 3 and not fu.get("day3_sent"):
            reminder = "second"
            fu["day3_sent"] = True
        elif days_elapsed == 1 and not fu.get("day1_sent"):
            reminder = "first"
            fu["day1_sent"] = True
        else:
            continue

        if email and guser and gpass and not fu.get("auto_rejected"):
            lines = [
                "Dear "+name+",",
                "",
                "This is your "+reminder+" reminder regarding your pending loan application.",
                "Days elapsed: "+str(days_elapsed)+" of 7 allowed.",
                "",
                "DOCUMENTS STILL REQUIRED:",
                "  1. Last 6 months bank statements",
                "  2. Latest salary slip or Form 16",
                "  3. Details of all existing loans",
                "",
                "WARNING: Application will be auto-rejected after 7 days without documents.",
                "",
                "Warm regards,",
                "AI Loan Processing Agent | National Finance Bank",
            ]
            try:
                msg=MIMEMultipart()
                msg["From"]=guser; msg["To"]=email
                msg["Subject"]="Reminder "+str(days_elapsed)+"/7: Documents Required - National Finance Bank"
                msg.attach(MIMEText(chr(10).join(lines),"plain"))
                with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
                    s.login(guser,gpass); s.send_message(msg)
                alog("FOLLOWUP","Sent day-"+str(days_elapsed)+" reminder to "+name)
            except Exception as e:
                alog("ERROR","Reminder failed: "+str(e))
            save_followup(aid, fu)


# ═══════════════════════════════════════════
# IMPROVEMENT 3: Daily Summary Email
# ═══════════════════════════════════════════

def send_daily_summary(guser, gpass):
    if not guser or not gpass: return
    today = datetime.now().date()
    conn  = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql(
            "SELECT * FROM decisions WHERE DATE(timestamp)=?",
            conn, params=[str(today)])
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        alog("SUMMARY","No applications today — skipping summary email")
        return

    total   = len(df)
    approved= len(df[df["decision"]=="APPROVED"]) if "decision" in df.columns else 0
    rejected= len(df[df["decision"]=="REJECTED"]) if "decision" in df.columns else 0
    review  = len(df[df["decision"]=="MANUAL REVIEW"]) if "decision" in df.columns else 0
    emails  = len(df[df["email_sent"]==1]) if "email_sent" in df.columns else 0
    fraud   = len(df[df["fraud_flag"]==1]) if "fraud_flag" in df.columns else 0
    avg_risk= round(df["default_prob"].mean(),1) if "default_prob" in df.columns and total>0 else 0

    # High risk alerts
    high_risk = df[df["default_prob"]>60] if "default_prob" in df.columns else pd.DataFrame()

    lines = [
        "DAILY LOAN AGENT REPORT",
        "National Finance Bank | "+str(today),
        "="*45,
        "",
        "TODAY'S SUMMARY:",
        "  Applications Processed : "+str(total),
        "  Approved               : "+str(approved)+" ("+str(round(approved/total*100,1) if total>0 else 0)+"%)",
        "  Rejected               : "+str(rejected)+" ("+str(round(rejected/total*100,1) if total>0 else 0)+"%)",
        "  Manual Review          : "+str(review)+" ("+str(round(review/total*100,1) if total>0 else 0)+"%)",
        "  Emails Sent            : "+str(emails),
        "  Fraud Flagged          : "+str(fraud),
        "  Avg Default Risk       : "+str(avg_risk)+"%",
    ]

    if not high_risk.empty:
        lines += ["", "HIGH RISK ALERTS (>60% default probability):"]
        for _, row in high_risk.iterrows():
            lines.append("  - "+str(row.get("applicant","?"))+" : "+
                         str(round(row.get("default_prob",0),1))+"% risk — FLAGGED")

    lines += [
        "",
        "Agent is running autonomously. Next check in 60 minutes.",
        "",
        "AI Loan Processing Agent | National Finance Bank",
    ]

    try:
        msg=MIMEMultipart()
        msg["From"]=guser; msg["To"]=guser
        msg["Subject"]="Daily Loan Agent Report — "+str(today)+" | "+str(total)+" applications"
        msg.attach(MIMEText(chr(10).join(lines),"plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(guser,gpass); s.send_message(msg)
        alog("SUMMARY","Daily report sent to "+guser)
    except Exception as e:
        alog("ERROR","Summary email failed: "+str(e))


# ═══════════════════════════════════════════
# IMPROVEMENT 1: Adaptive Learning
# ═══════════════════════════════════════════

def check_and_retrain(model, FN, FM):
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("""
            SELECT d.app_id, d.default_prob, d.decision, f.correct
            FROM decisions d JOIN feedback f ON d.app_id=f.app_id
            ORDER BY d.timestamp DESC LIMIT 50
        """, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if len(df) < 10:
        alog("LEARN","Not enough feedback yet ("+str(len(df))+"/10 needed)")
        return model, False

    wrong = len(df[df["correct"]==0])
    accuracy = round((1 - wrong/len(df))*100, 1)
    alog("LEARN","Feedback accuracy: "+str(accuracy)+"% ("+str(wrong)+" wrong of "+str(len(df))+")")

    if accuracy >= 80:
        alog("LEARN","Accuracy OK ("+str(accuracy)+"%) — no retraining needed")
        return model, False

    alog("LEARN","Accuracy below 80% — initiating self-retraining!")
    try:
        # Use existing decisions as pseudo-labels for retraining
        conn2 = sqlite3.connect(DB_FILE)
        dec_df = pd.read_sql("SELECT * FROM decisions LIMIT 500", conn2)
        conn2.close()

        if len(dec_df) < 20:
            alog("LEARN","Not enough data to retrain (need 20+)")
            return model, False

        # Build feature matrix from stored decisions
        X_rows = []
        y_rows = []
        for _, row in dec_df.iterrows():
            try:
                # Reconstruct approximate features from stored data
                la  = float(row.get("loan_amount",500000))
                inc = float(row.get("approved_amt",la) or la) / 0.75 if row.get("decision")=="MANUAL REVIEW" else la
                ac  = float(row.get("creditworth",75)) / 100.0
                prob= float(row.get("default_prob",30)) / 100.0

                feat = {f: FM.get(f,0) for f in FN}
                feat["EXT_SOURCE_MEAN"] = ac
                feat["EXT_SOURCE_1"]    = ac + np.random.normal(0, 0.02)
                feat["EXT_SOURCE_2"]    = ac + np.random.normal(0, 0.02)
                feat["EXT_SOURCE_3"]    = ac + np.random.normal(0, 0.02)
                feat["AMT_CREDIT"]      = la
                feat["LOAN_TO_INCOME"]  = la / (inc+1)

                X_rows.append([feat.get(f,0) for f in FN])
                y_rows.append(1 if prob > 0.5 else 0)
            except Exception:
                continue

        if len(X_rows) < 20:
            alog("LEARN","Feature reconstruction failed")
            return model, False

        X = np.array(X_rows); y = np.array(y_rows)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        new_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42
        )
        new_model.fit(X_tr, y_tr)

        y_pred = new_model.predict_proba(X_val)[:,1]
        new_auc = round(roc_auc_score(y_val, y_pred), 3)

        old_pred = model.predict_proba(X_val)[:,1]
        old_auc  = round(roc_auc_score(y_val, old_pred), 3)

        alog("LEARN","Old AUC: "+str(old_auc)+" | New AUC: "+str(new_auc))

        if new_auc >= old_auc:
            new_model.save_model(MODEL_FILE)
            conn3 = sqlite3.connect(DB_FILE)
            conn3.execute("""INSERT INTO model_versions
                (version,auc,trained_on,timestamp) VALUES (?,?,?,?)""",
                ("v_"+datetime.now().strftime("%Y%m%d_%H%M"),
                 new_auc, len(X_rows), datetime.now().isoformat()))
            conn3.commit(); conn3.close()
            alog("LEARN","Model improved! Saved new version (AUC "+str(new_auc)+")")
            return new_model, True
        else:
            alog("LEARN","New model worse — keeping original")
            return model, False

    except Exception as e:
        alog("ERROR","Retraining failed: "+str(e))
        return model, False


# ═══════════════════════════════════════════
# MAIN AGENT LOOP
# ═══════════════════════════════════════════

def run_once(config, model_ref, FN, FM):
    alog("AGENT","="*40)
    alog("AGENT","Cycle started at "+datetime.now().strftime("%H:%M:%S"))

    repo_rate = get_rbi_rate()
    processed = get_processed_ids()
    found=0; new_proc=0; emails=0; fraud_count=0

    # Multi-source perception
    df1 = scrape_google_sheet(config.get("sheet_url",""))
    df2 = scrape_csv_url(config.get("csv_url",""))
    all_df = merge_sources(df1, df2)

    if all_df.empty:
        alog("PERCEIVE","No applications found"); log_run(0,0,0,0,"NO_DATA"); return

    found = len(all_df)
    for _, row in all_df.iterrows():
        app = parse_row(row)
        aid = app["app_id"]
        if aid in processed: continue

        alog("PROCESS","Processing: "+app["name"]+" ["+app["_source"]+"]")

        # Fraud check
        is_fraud, fraud_reason = check_fraud(app, all_df)
        if is_fraud:
            fraud_count += 1
            log_decision(aid,app["name"],app["email"],app.get("phone",""),
                         "REJECTED",99.0,app["loan_amount"],0,None,0,
                         True,fraud_reason,app["_source"])
            save_processed_id(aid)
            # Alert manager
            if config.get("gmail_user") and config.get("gmail_pass"):
                try:
                    msg=MIMEMultipart()
                    msg["From"]=config["gmail_user"]; msg["To"]=config["gmail_user"]
                    msg["Subject"]="FRAUD ALERT: "+app["name"]+" - National Finance Bank"
                    msg.attach(MIMEText("FRAUD DETECTED\n\nApplicant: "+app["name"]+
                                        "\nReason: "+fraud_reason+"\nAction: Auto-rejected","plain"))
                    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
                        s.login(config["gmail_user"],config["gmail_pass"]); s.send_message(msg)
                    alog("FRAUD","Alert sent to manager")
                except Exception: pass
            continue

        # ML Decision
        try:
            result = decide(app, model_ref[0], FN, FM, repo_rate)
            alog("DECIDE",app["name"]+" -> "+result["decision"]+" ("+str(result["default_prob"])+"% risk)")
        except Exception as e:
            alog("ERROR","Decision failed: "+str(e)); continue

        # Log to database
        log_decision(aid,app["name"],app["email"],app.get("phone",""),
                     result["decision"],result["default_prob"],
                     app["loan_amount"],result["approved_amount"],
                     result["interest_rate"],result["emi"],
                     False,"",app["_source"])

        # Send email with PDF
        guser=config.get("gmail_user",""); gpass=config.get("gmail_pass","")
        if app["email"] and "@" in app["email"] and guser and gpass:
            sent=send_email(app["email"],app["name"],result,app,guser,gpass,
                            attach_pdf=True, feature_names=FN)
            if sent:
                conn=sqlite3.connect(DB_FILE)
                conn.execute("UPDATE decisions SET email_sent=1 WHERE app_id=?",(aid,))
                conn.commit(); conn.close()
                emails+=1

        # WhatsApp
        if app.get("phone") and TWILIO_SID:
            send_whatsapp(app.get("phone",""),app["name"],result["decision"],
                          result["approved_amount"],result["emi"],
                          TWILIO_SID,TWILIO_TOKEN,TWILIO_FROM)

        # Track manual reviews for follow-up
        if result["decision"]=="MANUAL REVIEW":
            save_followup(aid,{"email":app["email"],"name":app["name"],
                                "submitted":datetime.now().isoformat(),
                                "resolved":False})

        save_processed_id(aid); new_proc+=1

    # Follow-up agent
    run_followup_agent(config.get("gmail_user",""), config.get("gmail_pass",""))

    # Adaptive learning check every 10 runs
    if new_proc > 0 and st.session_state.get("run_count",0) % 10 == 0:
        new_model, improved = check_and_retrain(model_ref[0], FN, FM)
        if improved:
            model_ref[0] = new_model
            alog("LEARN","Model self-improved and updated in memory!")

    # Daily summary at 9AM
    now = datetime.now()
    if now.hour == 9 and now.minute < 60 and not st.session_state.get("summary_sent_today"):
        send_daily_summary(config.get("gmail_user",""), config.get("gmail_pass",""))
        st.session_state.summary_sent_today = True
    elif now.hour != 9:
        st.session_state.summary_sent_today = False

    log_run(found, new_proc, emails, fraud_count, "SUCCESS")
    alog("AGENT","Done: "+str(new_proc)+" new | "+str(emails)+" emails | "+str(fraud_count)+" fraud | next in 60 min")
    st.session_state["run_count"] = st.session_state.get("run_count",0) + 1

def log_run(found, processed, emails, fraud, status):
    conn=sqlite3.connect(DB_FILE)
    conn.execute("""INSERT INTO agent_runs
        (run_time,apps_found,processed,emails_sent,fraud_flagged,status) VALUES (?,?,?,?,?,?)""",
        (datetime.now().isoformat(),found,processed,emails,fraud,status))
    conn.commit(); conn.close()

def start_loop(config, model_ref, FN, FM):
    def loop():
        while st.session_state.get("running",False):
            run_once(config, model_ref, FN, FM)
            st.session_state["last_run"]=datetime.now().strftime("%H:%M:%S")
            time.sleep(INTERVAL)
    threading.Thread(target=loop,daemon=True).start()


# ═══════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════

@st.cache_resource
def load_model():
    for p in [".","/app"]:
        mf=os.path.join(p,"loan_model.json")
        ff=os.path.join(p,"feature_names.pkl")
        fm=os.path.join(p,"feature_medians.pkl")
        if all(os.path.exists(x) for x in [mf,ff,fm]):
            m=xgb.XGBClassifier(); m.load_model(mf)
            with open(ff,"rb") as f: fn=pickle.load(f)
            with open(fm,"rb") as f: fmed=pickle.load(f)
            return m,fn,fmed
    raise FileNotFoundError("Model files not found! CWD="+os.getcwd()+" Files="+str(os.listdir(".")))


# ═══════════════════════════════════════════
# UI
# ═══════════════════════════════════════════

init_db()
try:
    model,FN,FM=load_model(); model_ok=True
except Exception as e:
    model_ok=False; model_err=str(e)

if "running"    not in st.session_state: st.session_state.running=False
if "last_run"   not in st.session_state: st.session_state.last_run="Never"
if "alog"       not in st.session_state: st.session_state.alog=[]
if "run_count"  not in st.session_state: st.session_state.run_count=0
if "model_ref"  not in st.session_state and model_ok:
    st.session_state.model_ref=[model]

creds_ok=bool(GMAIL_USER and GMAIL_PASS and SHEET_URL)
sheet_url=SHEET_URL; guser=GMAIL_USER; gpass=GMAIL_PASS

st.markdown("""
<div class="hero">
<h1>🤖 Autonomous Loan Approval Agent v4.0</h1>
<p>9 Agentic Features: Perceive &rarr; Reason &rarr; Act &rarr; Remember &rarr; Repeat &nbsp;|&nbsp; Every 60 minutes</p>
</div>""", unsafe_allow_html=True)

if not model_ok:
    st.error("Model not loaded: "+model_err)
    st.info("Files: "+str(os.listdir("."))); st.stop()

# ── STATS
stats=get_stats()
s1,s2,s3,s4,s5,s6=st.columns(6)
s1.metric("Total Processed", stats["total"])
s2.metric("Approved",        stats["approved"])
s3.metric("Rejected",        stats["rejected"])
s4.metric("Under Review",    stats["review"])
s5.metric("Emails Sent",     stats["emails"])
s6.metric("Fraud Flagged",   stats["fraud"])
st.divider()

# ── TABS
tab1,tab2,tab3,tab4,tab5=st.tabs(["🎛️ Control","📊 Dashboard","📋 Log","🧠 Learning","ℹ️ Features"])

with tab1:
    left,right=st.columns([1,1])

    with left:
        st.markdown("### ⚙️ Agent Configuration")
        if creds_ok:
            st.success("Credentials loaded from Secrets")
            st.info("Sheet: ..."+sheet_url[-40:])
            st.info("Gmail: "+guser)
        else:
            st.warning("Enter credentials manually:")
            sheet_url=st.text_input("Google Sheet CSV URL",SHEET_URL)
            guser=st.text_input("Gmail Address",GMAIL_USER)
            gpass=st.text_input("App Password",GMAIL_PASS,type="password")

        csv_url=st.text_input("Extra CSV URL (optional)","",
                               help="Second data source — any public CSV URL")

        st.markdown("**Upload CSV (optional — 4th source):**")
        uploaded=st.file_uploader("Upload applications CSV",type=["csv"])

        st.markdown("**Twilio WhatsApp (optional):**")
        with st.expander("Configure WhatsApp"):
            t_sid=st.text_input("Twilio SID",TWILIO_SID)
            t_tok=st.text_input("Twilio Token",TWILIO_TOKEN,type="password")
            t_from=st.text_input("Twilio From",TWILIO_FROM)

        config={
            "sheet_url":sheet_url,"csv_url":csv_url,
            "gmail_user":guser,"gmail_pass":gpass,
            "twilio_sid":t_sid if "t_sid" in dir() else "",
            "twilio_token":t_tok if "t_tok" in dir() else "",
            "twilio_from":t_from if "t_from" in dir() else "",
        }

        st.markdown("")
        b1,b2,b3=st.columns(3)
        with b1:
            if st.button("▶️ Run Once"):
                if uploaded:
                    up_df = load_uploaded_csv(uploaded)
                with st.spinner("Agent running..."):
                    mr = st.session_state.get("model_ref",[model])
                    run_once(config, mr, FN, FM)
                st.success("Cycle complete!"); st.rerun()
        with b2:
            if not st.session_state.running:
                if st.button("🚀 Auto (60min)"):
                    st.session_state.running=True
                    mr=st.session_state.get("model_ref",[model])
                    start_loop(config,mr,FN,FM)
                    st.success("Agent started!"); st.rerun()
            else:
                if st.button("⏹️ Stop"):
                    st.session_state.running=False
                    st.warning("Stopped."); st.rerun()
        with b3:
            if st.button("📧 Send Summary"):
                with st.spinner("Sending daily summary..."):
                    send_daily_summary(guser,gpass)
                st.success("Summary sent to "+guser)

        if st.session_state.running:
            st.markdown('<div class="running"><b style="color:#10b981">🟢 RUNNING</b> | Last: '+st.session_state.last_run+' | Runs: '+str(st.session_state.run_count)+'</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="stopped"><b style="color:#64748b">⚪ Stopped</b></div>',unsafe_allow_html=True)

    with right:
        st.markdown("### 🔄 9 Agentic Features")
        features=[
            ("1. 🌐 PERCEIVE — Multi-Source","Scrapes Google Sheet + CSV URL + uploaded CSV simultaneously"),
            ("2. 🧠 REASON — XGBoost + SHAP","ML model + 6 credit rules + explainable AI"),
            ("3. 📧 ACT — Email + PDF","Decision email with full SHAP explanation PDF attached"),
            ("4. 💾 REMEMBER — SQLite","All decisions, runs, and model versions stored"),
            ("5. 😴 REPEAT — 60 min","Fully autonomous loop — zero human needed"),
            ("6. 🚨 FRAUD DETECTION","Flags suspicious patterns before ML decision"),
            ("7. 📅 FOLLOW-UP AGENT","Auto follows up manual reviews — rejects after 7 days"),
            ("8. 🔔 DAILY REPORT","Emails manager summary at 9AM daily"),
            ("9. 🧠 ADAPTIVE LEARNING","Retrains XGBoost when accuracy drops below 80%"),
        ]
        for title,desc in features:
            st.markdown('<div class="acard"><h4>'+title+'</h4><p>'+desc+'</p></div>',unsafe_allow_html=True)

with tab2:
    st.markdown("### 📈 Live Dashboard")
    if st.button("🔄 Refresh Dashboard"): st.rerun()

    daily=get_daily_stats()
    if not daily.empty and len(daily)>1:
        fig,axes=plt.subplots(1,3,figsize=(14,3.5))
        fig.patch.set_facecolor("#0f172a")
        for ax in axes:
            ax.set_facecolor("#1e293b")
            for s in ["top","right"]: ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#334155"); ax.spines["left"].set_color("#334155")
            ax.tick_params(colors="#94a3b8",labelsize=8)

        x=range(len(daily))
        axes[0].bar(x,daily["approved"],color="#10b981",label="Approved",alpha=0.8)
        axes[0].bar(x,daily["rejected"],bottom=daily["approved"],color="#ef4444",label="Rejected",alpha=0.8)
        axes[0].set_title("Daily Applications",color="#00d4aa",fontsize=10)
        axes[0].set_xticks(x); axes[0].set_xticklabels(daily["date"].tolist(),rotation=45,fontsize=7)
        axes[0].legend(fontsize=8,labelcolor="#94a3b8",facecolor="#1e293b",edgecolor="#334155")

        axes[1].plot(x,daily["avg_risk"],color="#f59e0b",linewidth=2,marker="o",markersize=4)
        axes[1].fill_between(x,daily["avg_risk"],alpha=0.2,color="#f59e0b")
        axes[1].axhline(y=55,color="#ef4444",linestyle="--",alpha=0.5,linewidth=1)
        axes[1].set_title("Avg Default Risk %",color="#00d4aa",fontsize=10)
        axes[1].set_xticks(x); axes[1].set_xticklabels(daily["date"].tolist(),rotation=45,fontsize=7)

        approval_rate=daily["approved"]/(daily["total"]+0.001)*100
        axes[2].plot(x,approval_rate,color="#00d4aa",linewidth=2,marker="s",markersize=4)
        axes[2].fill_between(x,approval_rate,alpha=0.2,color="#00d4aa")
        axes[2].set_title("Approval Rate %",color="#00d4aa",fontsize=10)
        axes[2].set_xticks(x); axes[2].set_xticklabels(daily["date"].tolist(),rotation=45,fontsize=7)

        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Run the agent on multiple days to see trend charts here.")

    st.divider()
    st.markdown("### 📊 All Decisions")
    df_dec=get_decisions()
    if not df_dec.empty:
        def cd(v):
            if v=="APPROVED": return "color:#10b981;font-weight:bold"
            elif v=="REJECTED": return "color:#ef4444;font-weight:bold"
            return "color:#f59e0b;font-weight:bold"
        cols=[c for c in ["app_id","applicant","email","decision","default_prob",
                           "loan_amount","approved_amt","fraud_flag","email_sent",
                           "source","timestamp"] if c in df_dec.columns]
        st.dataframe(df_dec[cols].style.applymap(cd,subset=["decision"]),
                     use_container_width=True)
        st.download_button("📥 Download CSV",data=df_dec.to_csv(index=False),
                           file_name="decisions.csv",mime="text/csv")
    else:
        st.info("No decisions yet.")

with tab3:
    st.markdown("### 📋 Agent Activity Log")
    lines=get_log()
    if lines:
        html=""
        for l in reversed(lines):
            l=l.strip()
            if   "APPROVED"     in l and "[DECIDE]" in l: c="#10b981"
            elif "REJECTED"     in l and "[DECIDE]" in l: c="#ef4444"
            elif "MANUAL REVIEW"in l and "[DECIDE]" in l: c="#f59e0b"
            elif "[ERROR]"      in l: c="#ef4444"
            elif "[EMAIL]"      in l: c="#60a5fa"
            elif "[FRAUD]"      in l: c="#f87171"
            elif "[LEARN]"      in l: c="#a78bfa"
            elif "[FOLLOWUP]"   in l: c="#fb923c"
            elif "[SUMMARY]"    in l: c="#34d399"
            elif "[AGENT]"      in l: c="#00d4aa"
            elif "[RBI]"        in l: c="#fbbf24"
            else: c="#94a3b8"
            html+="<span style='color:"+c+"'>"+l+"</span><br>"
        st.markdown('<div class="log-box">'+html+'</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="log-box">No activity yet.</div>',unsafe_allow_html=True)
    if st.button("🔄 Refresh Log"): st.rerun()

    st.divider()
    st.markdown("### 🏃 Agent Run History")
    conn=sqlite3.connect(DB_FILE)
    try: df_r=pd.read_sql("SELECT * FROM agent_runs ORDER BY run_time DESC LIMIT 20",conn)
    except: df_r=pd.DataFrame()
    conn.close()
    if not df_r.empty: st.dataframe(df_r,use_container_width=True)
    else: st.info("No runs yet.")

with tab4:
    st.markdown("### 🧠 Adaptive Learning Engine")
    conn=sqlite3.connect(DB_FILE)
    try: mv=pd.read_sql("SELECT * FROM model_versions ORDER BY timestamp DESC",conn)
    except: mv=pd.DataFrame()
    conn.close()

    if not mv.empty:
        st.success("Model has been retrained "+str(len(mv))+" times")
        st.dataframe(mv,use_container_width=True)
    else:
        st.info("Model has not been retrained yet. Needs 10+ feedback entries with accuracy <80%.")

    st.divider()
    st.markdown("**Submit Feedback on Recent Decisions:**")
    df_recent=get_decisions(10)
    if not df_recent.empty:
        for _,row in df_recent.iterrows():
            c1,c2,c3=st.columns([2,1,1])
            c1.write(str(row.get("applicant","?"))+" — "+str(row.get("decision","?")))
            with c2:
                if st.button("✅ Correct",key="c_"+str(row.get("id",""))):
                    conn=sqlite3.connect(DB_FILE)
                    conn.execute("INSERT OR IGNORE INTO feedback (app_id,correct,timestamp) VALUES (?,?,?)",
                                 (row.get("app_id",""),1,datetime.now().isoformat()))
                    conn.commit(); conn.close()
                    st.success("Saved!")
            with c3:
                if st.button("❌ Wrong",key="w_"+str(row.get("id",""))):
                    conn=sqlite3.connect(DB_FILE)
                    conn.execute("INSERT OR IGNORE INTO feedback (app_id,correct,timestamp) VALUES (?,?,?)",
                                 (row.get("app_id",""),0,datetime.now().isoformat()))
                    conn.commit(); conn.close()
                    st.warning("Flagged!")
    else:
        st.info("No decisions yet to rate.")

    if st.button("🔁 Force Retrain Now"):
        with st.spinner("Retraining model..."):
            mr=st.session_state.get("model_ref",[model])
            new_m,improved=check_and_retrain(mr[0],FN,FM)
            if improved:
                st.session_state.model_ref=[new_m]
                st.success("Model improved and updated!")
            else:
                st.info("No improvement found — keeping current model.")

with tab5:
    st.markdown("### ℹ️ All 9 Agentic Features Explained")
    explanations={
        "1. Multi-Source Perception":"Agent simultaneously scrapes Google Sheet, a second CSV URL, and accepts uploaded CSV files. All sources are merged automatically.",
        "2. XGBoost + SHAP Reasoning":"ML model trained on 307,511 records predicts default probability. SHAP explains exactly which factors drove each decision.",
        "3. Email + PDF Report":"Every applicant receives a professional email with a detailed PDF report showing their SHAP explanation chart and top decision factors.",
        "4. SQLite Memory":"Every decision, agent run, feedback entry, and model version is stored permanently in a local database.",
        "5. 60-Minute Loop":"Agent runs fully autonomously every hour — scrapes, decides, emails, logs — with zero human intervention.",
        "6. Fraud Detection":"Before ML decision, agent checks for identical credit scores, multiple same-email submissions, age-income mismatch, unrealistic values.",
        "7. Follow-up Agent":"Manual review cases are tracked daily. Agent sends reminders on Day 1, Day 3, Day 7, then auto-rejects on Day 8.",
        "8. Daily Summary Email":"Every morning at 9AM, agent emails the manager a full report: totals, approval rate, avg risk, high-risk alerts.",
        "9. Adaptive Learning":"Agent monitors feedback accuracy. If it drops below 80%, it automatically retrains the XGBoost model on recent decisions.",
    }
    for title,desc in explanations.items():
        with st.expander(title):
            st.write(desc)

st.caption("Autonomous Loan Agent v4.0 | "+datetime.now().strftime("%Y-%m-%d %H:%M")+" | 9 Agentic Features Active")
