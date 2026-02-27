# -*- coding: utf-8 -*-
"""
AUTONOMOUS LOAN APPROVAL AGENT v4.0
=====================================
FIXES APPLIED:
  1. bootstrap_train_model() added — auto-trains XGBoost on first launch if
     loan_model.json / feature_names.pkl / feature_medians.pkl are missing.
     This is the root cause of the app never starting on Streamlit Cloud.
  2. Removed deprecated use_label_encoder=False (dropped in XGBoost 2.0+).
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
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.units import inch
    PDF_OK = True
except Exception:
    PDF_OK = False

# ── Credentials from environment / Streamlit Secrets
GMAIL_USER   = os.environ.get("GMAIL_USER",   "")
GMAIL_PASS   = os.environ.get("GMAIL_PASS",   "")
SHEET_URL    = os.environ.get("SHEET_URL",    "")
TWILIO_SID   = os.environ.get("TWILIO_SID",   "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_FROM",  "")

DB_FILE        = "agent_memory.db"
LOG_FILE       = "agent_log.txt"
PROCESSED_FILE = "processed_ids.json"
FOLLOWUP_FILE  = "followup_tracker.json"
MODEL_FILE     = "loan_model.json"
INTERVAL       = 3600

# Thread-safe globals — st.session_state is inaccessible from background threads
# Using a mutable dict avoids "global" keyword issues inside nested scopes
_G = {
    "log":          [],      # agent activity log
    "running":      False,   # loop on/off
    "last_run":     "Never", # last cycle timestamp
    "run_count":    0,       # cycle counter
    "summary_sent": False,   # daily summary sent today
}
_AGENT_LOG = _G["log"]  # convenience alias for alog()

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s,%(msecs)03d [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

st.set_page_config(page_title="Autonomous Loan Agent v4",
                   page_icon="🤖", layout="wide")

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
.log-box{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:1rem;font-family:"DM Mono",monospace;font-size:.78rem;color:#94a3b8;height:300px;overflow-y:auto;}
.running{background:linear-gradient(135deg,#052e16,#064e3b);border:1px solid #10b981;border-radius:10px;padding:.8rem 1rem;}
.stopped{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:.8rem 1rem;}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0099ff);color:white;border:none;padding:.6rem 1.5rem;font-weight:600;border-radius:10px;}
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

def log_decision(app_id, name, email, phone, dec, prob, la, aa, ir, emi,
                 fraud, fraud_reason, source, _extra=None):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""INSERT OR IGNORE INTO decisions
            (app_id,applicant,email,phone,decision,default_prob,creditworth,
             loan_amount,approved_amt,interest_rate,emi,fraud_flag,fraud_reason,source,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (app_id, name, email, phone, dec,
             float(prob), float(100 - prob),
             float(la)  if la  else 0.0,
             float(aa)  if aa  else 0.0,
             float(ir)  if ir  else 0.0,
             float(emi) if emi else 0.0,
             int(fraud), fraud_reason, source, datetime.now().isoformat()))
        conn.commit()
    except Exception:
        pass
    conn.close()

def get_decisions(limit=100):
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql(
            f"SELECT * FROM decisions ORDER BY timestamp DESC LIMIT {limit}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close(); return df

def get_stats():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    s = {"total":0,"approved":0,"rejected":0,"review":0,"emails":0,"fraud":0}
    try:
        for k, q in [
            ("total",    "SELECT COUNT(*) FROM decisions"),
            ("approved", "SELECT COUNT(*) FROM decisions WHERE decision='APPROVED'"),
            ("rejected", "SELECT COUNT(*) FROM decisions WHERE decision='REJECTED'"),
            ("review",   "SELECT COUNT(*) FROM decisions WHERE decision='MANUAL REVIEW'"),
            ("emails",   "SELECT COUNT(*) FROM decisions WHERE email_sent=1"),
            ("fraud",    "SELECT COUNT(*) FROM decisions WHERE fraud_flag=1"),
        ]:
            c.execute(q); s[k] = c.fetchone()[0]
    except Exception:
        pass
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
    except Exception:
        df = pd.DataFrame()
    conn.close(); return df

def get_processed_ids():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f: return set(json.load(f))
    return set()

def save_processed_id(aid):
    ids = get_processed_ids(); ids.add(aid)
    with open(PROCESSED_FILE, "w") as f: json.dump(list(ids), f)

def get_followups():
    if os.path.exists(FOLLOWUP_FILE):
        with open(FOLLOWUP_FILE) as f: return json.load(f)
    return {}

def save_followup(aid, data):
    fu = get_followups(); fu[aid] = data
    with open(FOLLOWUP_FILE, "w") as f: json.dump(fu, f)


# ═══════════════════════════════════════════
# AGENT LOG
# ═══════════════════════════════════════════

def alog(action, msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = "[" + ts + "] [" + action + "] " + msg
    logging.info(line)
    # Use global list — st.session_state is NOT accessible from background threads
    _G["log"].append(line)
    if len(_G["log"]) > 500:
        _G["log"] = _G["log"][-500:]

def get_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f: return f.readlines()[-80:]
    return []


# ═══════════════════════════════════════════
# RBI Rate Scraping
# ═══════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_rbi_rate():
    try:
        r = requests.get("https://api.rbi.org.in/api/v1/keyrates", timeout=5)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("data", []):
                if "repo" in str(item).lower():
                    rate = float(re.findall(r'\d+\.?\d*',
                                            str(item.get("rate", "6.5")))[0])
                    alog("RBI", "Live repo rate: " + str(rate) + "%")
                    return rate
    except Exception:
        pass
    try:
        r = requests.get(
            "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
            timeout=5)
        matches = re.findall(r'repo rate.*?(\d+\.?\d+)%', r.text.lower())
        if matches:
            rate = float(matches[0])
            alog("RBI", "Scraped repo rate: " + str(rate) + "%")
            return rate
    except Exception:
        pass
    alog("RBI", "Using fallback rate: 6.5%")
    return 6.50


# ═══════════════════════════════════════════
# Fraud Detection
# ═══════════════════════════════════════════

def check_fraud(app, all_apps_df):
    flags = []
    cs1, cs2, cs3 = app["credit_1"], app["credit_2"], app["credit_3"]

    # Rule 1: Identical bureau scores — only when 3 separate columns exist
    if app.get("has_3_credit_cols", False) and cs1 == cs2 == cs3:
        flags.append("Identical credit scores across all 3 bureaus")

    # Rule 2: Duplicate email (same person applying multiple times)
    if not all_apps_df.empty and "email" in all_apps_df.columns:
        email_count = len(all_apps_df[all_apps_df["email"] == app["email"]])
        if email_count > 1:
            flags.append("Duplicate email — submitted " + str(email_count) + " times")

    # Rule 3: Perfect / near-perfect credit score
    if cs1 > 0.95 or cs2 > 0.95 or cs3 > 0.95:
        flags.append("Suspiciously high credit score (>" + str(round(max(cs1,cs2,cs3),2)) + ")")

    # Rule 4: Extremely low credit score (possible data entry error or fabricated)
    if cs1 < 0.10 or cs2 < 0.10 or cs3 < 0.10:
        flags.append("Abnormally low credit score (<0.10)")

    # Rule 5: Age-income mismatch
    if app["age"] < 25 and app["annual_income"] > 2000000:
        flags.append("Age " + str(app["age"]) +
                     " with income Rs" + "{:,.0f}".format(app["annual_income"]) + " is unusual")

    # Rule 6: Loan > 7x annual income (relaxed from 10x)
    ratio = app["loan_amount"] / (app["annual_income"] + 1)
    if ratio > 7:
        flags.append("Loan is " + str(round(ratio, 1)) + "x annual income (max 7x)")

    # Rule 7: Zero or missing income
    if app["annual_income"] <= 0:
        flags.append("Annual income is zero or missing")

    # Rule 8: Loan amount suspiciously round (e.g. exactly 5000000, 10000000)
    if app["loan_amount"] >= 1000000 and app["loan_amount"] % 1000000 == 0:
        if app["loan_amount"] > app["annual_income"] * 3:
            flags.append("Suspiciously round loan amount Rs" +
                         "{:,.0f}".format(app["loan_amount"]) + " exceeds 3x income")

    # Rule 9: Invalid email format
    email = app.get("email", "")
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        flags.append("Invalid or missing email address")

    is_fraud = len(flags) > 0
    reason   = " | ".join(flags) if flags else ""
    if is_fraud:
        alog("FRAUD", "FLAGGED: " + app["name"] + " -- " + reason)
    return is_fraud, reason


# ═══════════════════════════════════════════
# SHAP PDF Generation
# ═══════════════════════════════════════════

def generate_shap_pdf(app, result, shap_values, feature_names):
    buf = io.BytesIO()
    try:
        if PDF_OK:
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    rightMargin=40, leftMargin=40,
                                    topMargin=40, bottomMargin=40)
            styles  = getSampleStyleSheet()
            story   = []

            title_style = ParagraphStyle(
                'title', parent=styles['Title'],
                textColor=colors.HexColor('#0f2027'),
                fontSize=18, spaceAfter=6)
            story.append(Paragraph("National Finance Bank", title_style))
            story.append(Paragraph("AI Loan Decision Report", title_style))
            story.append(HRFlowable(width="100%", thickness=2,
                                    color=colors.HexColor('#00d4aa')))
            story.append(Spacer(1, 12))

            dec = result["decision"]
            dec_color = (colors.HexColor('#10b981') if dec == "APPROVED"
                         else colors.HexColor('#ef4444') if dec == "REJECTED"
                         else colors.HexColor('#f59e0b'))
            story.append(Paragraph(
                "DECISION: " + dec,
                ParagraphStyle('dec', parent=styles['Heading1'],
                               textColor=dec_color, fontSize=22, spaceAfter=6)))
            story.append(Spacer(1, 8))

            info_data = [
                ["Applicant", app["name"],
                 "Date", datetime.now().strftime("%d %B %Y")],
                ["Loan Amount", "Rs{:,.0f}".format(app["loan_amount"]),
                 "Default Risk", str(result["default_prob"]) + "%"],
                ["Credit Score", str(result["avg_credit"]),
                 "DTI Ratio", str(result["dti"]) + "%"],
            ]
            if dec == "APPROVED":
                info_data.append([
                    "Approved Amt", "Rs{:,.0f}".format(result["approved_amount"]),
                    "Interest Rate", str(result["interest_rate"]) + "% p.a."])
                info_data.append([
                    "Monthly EMI", "Rs{:,.0f}".format(result["emi"]),
                    "Tenure", "60 months"])

            t = Table(info_data,
                      colWidths=[1.4*inch, 2*inch, 1.4*inch, 2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
                ('TEXTCOLOR',  (0,0), (0,-1),  colors.HexColor('#64748b')),
                ('TEXTCOLOR',  (2,0), (2,-1),  colors.HexColor('#64748b')),
                ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE',   (0,0), (-1,-1), 10),
                ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING',    (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 16))

            story.append(Paragraph(
                "AI Explanation — Why This Decision?",
                ParagraphStyle('h2', parent=styles['Heading2'],
                               textColor=colors.HexColor('#0f2027'),
                               fontSize=13, spaceAfter=8)))

            labels_map = {
                "EXT_SOURCE_MEAN": "Avg Credit Score",
                "EXT_SOURCE_1":    "Credit Score 1",
                "EXT_SOURCE_2":    "Credit Score 2",
                "EXT_SOURCE_3":    "Credit Score 3",
                "DEBT_TO_INCOME":  "Debt/Income",
                "LOAN_TO_INCOME":  "Loan/Income",
                "AMT_INCOME_TOTAL":"Annual Income",
                "AMT_CREDIT":      "Loan Amount",
                "AGE_YEARS":       "Age",
                "YEARS_EMPLOYED":  "Years Employed",
                "IS_EMPLOYED":     "Employment Status",
                "INCOME_PER_PERSON":"Income/Person",
                "CNT_FAM_MEMBERS": "Family Size",
                "AMT_ANNUITY":     "Monthly EMI",
            }

            if shap_values is not None:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                sv   = np.array(shap_values)
                idx  = np.argsort(np.abs(sv))[-8:]
                fn_a = np.array(feature_names)
                vals = sv[idx]
                labs = [labels_map.get(f, f) for f in fn_a[idx]]
                clrs = ["#ef4444" if v > 0 else "#10b981" for v in vals]
                ax.barh(range(len(vals)), vals, color=clrs,
                        edgecolor="none", height=0.6)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(labs, fontsize=9)
                ax.axvline(0, color="#94a3b8", lw=0.8, linestyle="--")
                ax.set_xlabel(
                    "Impact on Default Risk  "
                    "(red=increases risk, green=reduces risk)", fontsize=8)
                ax.set_facecolor("#f8fafc")
                fig.patch.set_facecolor("#f8fafc")
                for sp in ["top", "right"]:
                    ax.spines[sp].set_visible(False)
                plt.tight_layout()
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                img_buf.seek(0)
                from reportlab.platypus import Image as RLImage
                story.append(RLImage(img_buf, width=6*inch, height=3*inch))
            else:
                story.append(Paragraph(
                    "SHAP library not available.", styles['Normal']))

            story.append(Spacer(1, 12))

            if shap_values is not None:
                sv   = np.array(shap_values)
                idx  = np.argsort(np.abs(sv))[-5:][::-1]
                fn_a = np.array(feature_names)
                story.append(Paragraph(
                    "Top 5 Decision Factors:",
                    ParagraphStyle('h3', parent=styles['Heading3'],
                                   textColor=colors.HexColor('#0f2027'),
                                   fontSize=11, spaceAfter=4)))
                for rank, i in enumerate(idx, 1):
                    fname  = labels_map.get(fn_a[i], fn_a[i])
                    impact = sv[i]
                    dirn   = "INCREASES" if impact > 0 else "REDUCES"
                    clr    = "#ef4444" if impact > 0 else "#10b981"
                    story.append(Paragraph(
                        str(rank) + ". <b>" + fname + "</b> — "
                        "<font color='" + clr + "'>" + dirn +
                        " default risk by " +
                        str(round(abs(impact), 3)) + "</font>",
                        ParagraphStyle('factor', parent=styles['Normal'],
                                       fontSize=10, spaceAfter=3,
                                       leftIndent=10)))

            story.append(Spacer(1, 12))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor('#e2e8f0')))
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "Ref: " + app.get("app_id", "N/A") + " | " +
                datetime.now().strftime("%d %B %Y %H:%M"),
                ParagraphStyle('footer', parent=styles['Normal'],
                               fontSize=8,
                               textColor=colors.HexColor('#94a3b8'))))
            doc.build(story)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.95, "National Finance Bank",
                    ha='center', va='top', fontsize=16,
                    fontweight='bold', transform=ax.transAxes)
            dec = result["decision"]
            color = ("#10b981" if dec == "APPROVED"
                     else "#ef4444" if dec == "REJECTED"
                     else "#f59e0b")
            ax.text(0.5, 0.72, dec, ha='center', va='top',
                    fontsize=20, fontweight='bold',
                    color=color, transform=ax.transAxes)
            for i, d in enumerate([
                "Applicant: " + app["name"],
                "Default Risk: " + str(result["default_prob"]) + "%",
                "Credit Score: " + str(result["avg_credit"]),
                "DTI Ratio: " + str(result["dti"]) + "%",
            ]):
                ax.text(0.1, 0.58 - i*0.08, d,
                        fontsize=11, transform=ax.transAxes)
            ax.axis('off')
            fig.savefig(buf, format='pdf', bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        alog("ERROR", "PDF generation failed: " + str(e))

    buf.seek(0); return buf


# ═══════════════════════════════════════════
# Multi-Source Perception
# ═══════════════════════════════════════════

def scrape_google_sheet(url):
    try:
        if "docs.google.com/spreadsheets" in url:
            if "/edit" in url:
                url = url.split("/edit")[0] + "/export?format=csv"
            elif "export?format=csv" not in url:
                url = url + "/export?format=csv"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df["_source"] = "google_sheet"
            alog("PERCEIVE", "Google Sheet: " + str(len(df)) + " rows")
            return df
    except Exception as e:
        alog("ERROR", "Sheet scrape failed: " + str(e))
    return pd.DataFrame()

def scrape_csv_url(csv_url):
    if not csv_url: return pd.DataFrame()
    try:
        r  = requests.get(csv_url, timeout=10)
        df = pd.read_csv(io.StringIO(r.text))
        df["_source"] = "csv_url"
        alog("PERCEIVE", "CSV URL: " + str(len(df)) + " rows")
        return df
    except Exception as e:
        alog("ERROR", "CSV URL failed: " + str(e))
    return pd.DataFrame()

def load_uploaded_csv(uploaded_file):
    if uploaded_file is None: return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file)
        df["_source"] = "csv_upload"
        alog("PERCEIVE", "Uploaded CSV: " + str(len(df)) + " rows")
        return df
    except Exception as e:
        alog("ERROR", "CSV upload failed: " + str(e))
    return pd.DataFrame()

def merge_sources(*dfs):
    valid = [df for df in dfs if not df.empty]
    if not valid: return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True)
    alog("PERCEIVE", "Merged " + str(len(merged)) +
         " total applications from " + str(len(valid)) + " sources")
    return merged


# ═══════════════════════════════════════════
# Parse Row & Make Decision
# ═══════════════════════════════════════════

def _fuzzy_get(row, keywords, default=None):
    """
    Case-insensitive, partial-match column lookup.
    Tries exact match first, then checks if any keyword appears
    anywhere in the lowercased column name.
    keywords: list of strings ordered by priority.
    Returns the first matching value, or default.

    Examples of what this handles automatically:
      'full_name', 'Full Name', 'FULLNAME', 'customer name', 'applicant_name'
      'ann_income', 'Annual Income (Rs)', 'yearly income', 'income_pa'
      'loan_amt', 'Loan Amount', 'credit amount', 'requested_loan'
      'cibil', 'credit score', 'Credit_Score', 'score'
    """
    # Build a lowercase lookup map once
    low_map = {str(k).lower().replace(" ","_").replace("-","_"): (k, v)
               for k, v in row.items()}
    for kw in keywords:
        kw_low = kw.lower().replace(" ","_").replace("-","_")
        # 1) exact match (case-insensitive)
        if kw_low in low_map:
            val = low_map[kw_low][1]
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                return val
        # 2) partial match — keyword contained anywhere in column name
        for col_low, (col_orig, col_val) in low_map.items():
            if kw_low in col_low:
                if col_val is not None and str(col_val).strip() not in ("", "nan", "None"):
                    return col_val
    return default


def parse_row(row):
    def sf(v, d=0.0):
        try:    return float(str(v).replace(",", "").replace("₹","").replace("Rs","").strip())
        except: return d
    def si(v, d=0):
        try:    return int(float(str(v).strip()))
        except: return d

    # ── Fuzzy column extraction — works with ANY reasonable column naming ──
    # Name: 'full name', 'name', 'applicant', 'customer', 'customer_name' etc.
    name_raw  = _fuzzy_get(row, ["full_name","full name","applicant name","customer name",
                                  "name","applicant","customer"], "Applicant")

    # Email: 'email', 'email address', 'e-mail', 'mail' etc.
    email_raw = _fuzzy_get(row, ["email address","email_address","e-mail",
                                  "email","mail"], "")

    # Phone: 'phone', 'mobile', 'contact', 'phone number', 'mobile number'
    phone_raw = _fuzzy_get(row, ["phone number","mobile number","phone","mobile",
                                  "contact","cell"], "")

    # Age: 'age', 'age_years', 'applicant age'
    age_raw   = _fuzzy_get(row, ["age_years","applicant age","age"], 34)

    # Income: 'annual income', 'income', 'salary', 'yearly income', 'ann_income'
    income_raw= _fuzzy_get(row, ["annual income (rs)","annual income","yearly income",
                                  "ann_income","income_pa","salary","income"], 0.0)

    # Loan: 'loan amount', 'loan', 'credit amount', 'requested amount', 'loan_amt'
    loan_raw  = _fuzzy_get(row, ["loan amount (rs)","loan amount","loan_amount",
                                  "credit amount","requested amount","loan_amt",
                                  "loan"], 0.0)

    # Existing EMI: 'existing emi', 'current emi', 'monthly emi', 'emi'
    emi_raw   = _fuzzy_get(row, ["existing emi (rs/month)","existing emi","current emi",
                                  "monthly emi","emi"], 0.0)

    # Employment: 'employment status', 'employment', 'job status', 'employed'
    emp_raw   = _fuzzy_get(row, ["employment status","employment_status","job status",
                                  "employment","job","employed","work status"], "Employed")

    # Years employed: 'years employed', 'work experience', 'experience', 'tenure'
    yrs_raw   = _fuzzy_get(row, ["years employed","years_employed","work experience",
                                  "experience","job tenure","tenure"], 3.0)

    # Family: 'family members', 'dependents', 'family size', 'household'
    fam_raw   = _fuzzy_get(row, ["family members","family_members","dependents",
                                  "family size","household size","family"], 3)

    # Credit scores — check for 3 separate bureau columns first
    cs1_raw   = _fuzzy_get(row, ["credit score 1","credit_score_1","bureau_1",
                                  "cibil_1","score_1","credit1"], None)
    cs2_raw   = _fuzzy_get(row, ["credit score 2","credit_score_2","bureau_2",
                                  "cibil_2","score_2","credit2"], None)
    cs3_raw   = _fuzzy_get(row, ["credit score 3","credit_score_3","bureau_3",
                                  "cibil_3","score_3","credit3"], None)

    # Single credit score column: 'credit score', 'cibil', 'credit', 'score'
    cs_single_raw = _fuzzy_get(row, ["credit score","cibil score","credit_score",
                                      "cibil","credit","score"], None)

    has_3_credit_cols = all(x is not None for x in [cs1_raw, cs2_raw, cs3_raw])

    if has_3_credit_cols:
        cs1 = sf(cs1_raw, 0.7); cs2 = sf(cs2_raw, 0.7); cs3 = sf(cs3_raw, 0.7)
    elif cs_single_raw is not None:
        cs_val = sf(cs_single_raw, 0.7)
        # Auto-normalise: if score looks like CIBIL (300-900), convert to 0-1 scale
        if cs_val > 1.0:
            cs_val = round((cs_val - 300) / 600, 4)
            cs_val = max(0.01, min(0.99, cs_val))
        cs1 = cs2 = cs3 = cs_val
    else:
        cs1 = cs2 = cs3 = 0.7  # safe default

    income = sf(income_raw, 0.0)
    loan   = sf(loan_raw,   0.0)
    age    = si(age_raw,    34)
    emp    = str(emp_raw)

    # ── Stable unique ID (unchanged logic) ──
    import hashlib
    _ts = str(_fuzzy_get(row, ["timestamp","submitted_at","date","created_at"], ""))
    if _ts and _ts not in ("", "nan", "None"):
        raw = (_ts + "_" + str(name_raw)).replace(" ","_").replace(":","").replace("/","")
    elif str(name_raw).strip() and str(email_raw).strip():
        raw = hashlib.md5((str(name_raw)+str(email_raw)+str(loan_raw)).encode()).hexdigest()[:16]
    else:
        raw = datetime.now().strftime("%Y%m%d%H%M%S") + str(abs(hash(str(row.values.tolist()))))[:6]
    aid = raw[:24].replace(" ", "_")

    # ── All values now come from fuzzy-matched variables — no hardcoded column names ──
    return {
        "app_id"            : aid,
        "name"              : str(name_raw).strip()  or "Applicant",
        "email"             : str(email_raw).strip() or "",
        "phone"             : str(phone_raw).strip() or "",
        "age"               : si(age_raw,    34),
        "annual_income"     : sf(income_raw, 900000),
        "loan_amount"       : sf(loan_raw,   500000),
        "existing_emi"      : sf(emi_raw,    0.0),
        "employment"        : str(emp_raw),
        "yrs_employed"      : sf(yrs_raw,    3.0),
        "credit_1"          : cs1 if cs1 > 0 else 0.7,
        "credit_2"          : cs2 if cs2 > 0 else 0.7,
        "credit_3"          : cs3 if cs3 > 0 else 0.7,
        "has_3_credit_cols" : has_3_credit_cols,
        "family"            : si(fam_raw,    3),
        "_source"           : str(row.get("_source", "google_sheet")),
    }


def decide(app, model, FN, FM, repo_rate=6.50):
    inc  = app["annual_income"]; la  = app["loan_amount"]
    age  = app["age"]
    emp  = app["employment"].lower() in ["employed","yes","true","1"]
    yrs  = app["yrs_employed"]; fam = app["family"]
    emi0 = app["existing_emi"]
    cs1  = app["credit_1"]; cs2 = app["credit_2"]; cs3 = app["credit_3"]

    mr  = 0.12 / 12
    emi = (la * mr * (1+mr)**60) / ((1+mr)**60 - 1)
    te  = emi0 + emi; mi = inc / 12
    ac  = (cs1 + cs2 + cs3) / 3
    dti = te / (mi + 0.001)

    feat = {
        "EXT_SOURCE_1":cs1,"EXT_SOURCE_2":cs2,"EXT_SOURCE_3":cs3,
        "EXT_SOURCE_MEAN":ac,
        "DEBT_TO_INCOME":te/(inc+1),"LOAN_TO_INCOME":la/(inc+1),
        "ANNUITY_TO_CREDIT":emi/(la+1),
        "AMT_INCOME_TOTAL":inc,"AMT_CREDIT":la,"AMT_ANNUITY":emi,
        "AGE_YEARS":age,"YEARS_EMPLOYED":yrs,"IS_EMPLOYED":int(emp),
        "INCOME_PER_PERSON":inc/(fam+1),"CNT_FAM_MEMBERS":fam,
    }
    vec = [feat.get(f, FM.get(f, 0)) for f in FN]
    df2 = pd.DataFrame([vec], columns=FN)
    prob = float(model.predict_proba(df2)[0][1])
    pp   = round(prob * 100, 1)

    rules = {
        "Credit Score":     ac  >= 0.45,
        "DTI Ratio":        dti <= 0.60,
        "Employment":       emp,
        "Min. Income":      inc >= 150000,
        "Loan-to-Income":   la / inc <= 5.0,
        "Age Range":        21 <= age <= 65,
    }
    failed = [k for k, v in rules.items() if not v]
    cf  = any(r in failed for r in ["Employment","Min. Income","Age Range"])
    sp  = 4.0 if ac > 0.7 else 5.5 if ac > 0.55 else 7.0
    ir  = round(repo_rate + sp, 2)

    # Decision — 3 tiers:
    # REJECTED:      very high ML risk (>55%)  OR  3+ rule failures  OR  critical failure + high risk (>45%)
    # MANUAL REVIEW: moderate risk (>15%)  OR  any single rule failure  OR  any critical failure
    # APPROVED:      passes all rules with low ML risk (≤15%)
    if prob > 0.55 or len(failed) >= 3 or (cf and prob > 0.45):
        dec = "REJECTED";      aa = 0;       ir = None
    elif prob > 0.15 or len(failed) >= 1 or cf:
        dec = "MANUAL REVIEW"; aa = la * 0.75
    else:
        dec = "APPROVED";      aa = la

    shap_vals = None
    if SHAP_OK:
        try:
            bg        = pd.DataFrame([FM])[FN]
            explainer = shap.TreeExplainer(model, bg)
            shap_vals = explainer.shap_values(df2)[0]
        except Exception:
            pass

    return {
        "decision":dec,"default_prob":pp,"avg_credit":round(ac,3),
        "dti":round(dti*100,1),"emi":round(emi,0),
        "approved_amount":aa,"interest_rate":ir,
        "failed_rules":failed,"creditworthiness":round((1-prob)*100,1),
        "shap_values":shap_vals,
    }


# ═══════════════════════════════════════════
# Email + PDF
# ═══════════════════════════════════════════

def send_email(to_email, name, result, app, guser, gpass,
               attach_pdf=True, feature_names=None):
    dec = result["decision"]; dp = result["default_prob"]
    ac  = result["avg_credit"]; dti = result["dti"]
    emi = result["emi"]; aa = result["approved_amount"]
    ir  = result["interest_rate"]; fr = result["failed_rules"]
    cw  = result["creditworthiness"]
    ts  = datetime.now().strftime("%d %B %Y")

    shap_section = ""
    if result.get("shap_values") is not None and feature_names:
        sv   = np.array(result["shap_values"])
        idx  = np.argsort(np.abs(sv))[-3:][::-1]
        fn_a = np.array(feature_names)
        lbls = {"EXT_SOURCE_MEAN":"Avg Credit Score",
                "DEBT_TO_INCOME":"Debt/Income","LOAN_TO_INCOME":"Loan/Income",
                "AGE_YEARS":"Age","YEARS_EMPLOYED":"Yrs Employed",
                "IS_EMPLOYED":"Employment","AMT_INCOME_TOTAL":"Income",
                "AMT_CREDIT":"Loan Amt"}
        lines_shap = ["", "TOP DECISION FACTORS (AI Explanation):"]
        for rank, i in enumerate(idx, 1):
            fname = lbls.get(fn_a[i], fn_a[i])
            dirn  = "increases" if sv[i] > 0 else "reduces"
            lines_shap.append("  " + str(rank) + ". " + fname +
                               " -- " + dirn + " risk by " +
                               str(round(abs(sv[i]), 3)))
        shap_section = "\n".join(lines_shap)

    if dec == "APPROVED":
        subject = "Your Loan Application is APPROVED - National Finance Bank"
        body = "\n".join([
            "Dear " + name + ",", "",
            "Congratulations! Your loan application is APPROVED.", "",
            "DECISION SUMMARY:",
            "  Default Probability : " + str(dp) + "%",
            "  Creditworthiness    : " + str(cw) + "/100",
            "  Avg Credit Score    : " + str(ac),
            "  DTI Ratio           : " + str(dti) + "%", "",
            "LOAN OFFER:",
            "  Approved Amount : Rs" + "{:,.0f}".format(aa),
            "  Interest Rate   : " + str(ir) + "% p.a.",
            "  Monthly EMI     : Rs" + "{:,.0f}".format(emi) + "/month (60 months)",
        ]) + shap_section + "\n\nDate: " + ts + "\n\nWarm regards,\nNational Finance Bank AI Agent"
    elif dec == "REJECTED":
        subject = "Loan Application Status - National Finance Bank"
        body = "\n".join([
            "Dear " + name + ",", "",
            "After AI review, your application could not be approved.", "",
            "ASSESSMENT:",
            "  Default Probability : " + str(dp) + "%",
            "  Avg Credit Score    : " + str(ac),
            "  DTI Ratio           : " + str(dti) + "%",
            "  Issues Found        : " + (", ".join(fr) if fr else "high default probability"),
        ]) + shap_section + "\n\nDate: " + ts + "\n\nWarm regards,\nNational Finance Bank AI Agent"
    else:
        subject = "Loan Application Under Review - National Finance Bank"
        body = "\n".join([
            "Dear " + name + ",", "",
            "Your application requires manual review. This is NOT a rejection.", "",
            "  Default Probability : " + str(dp) + "%",
            "  Potential Amount    : Rs" + "{:,.0f}".format(aa),
        ]) + shap_section + "\n\nDate: " + ts + "\n\nWarm regards,\nNational Finance Bank AI Agent"

    try:
        msg = MIMEMultipart()
        msg["From"] = guser; msg["To"] = to_email; msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if attach_pdf:
            pdf_buf = generate_shap_pdf(
                app, result, result.get("shap_values"), feature_names)
            if pdf_buf.getvalue():
                part = MIMEBase("application", "octet-stream")
                part.set_payload(pdf_buf.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", "attachment",
                                filename="loan_report_" +
                                app.get("app_id", "report") + ".pdf")
                msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(guser, gpass); s.send_message(msg)
        alog("EMAIL", "Sent " + dec + " (+PDF) to " + to_email)
        return True
    except Exception as e:
        alog("ERROR", "Email failed: " + str(e)); return False


# ═══════════════════════════════════════════
# WhatsApp (Twilio)
# ═══════════════════════════════════════════

def send_whatsapp(phone, name, decision, amount, emi,
                  twilio_sid, twilio_token, twilio_from):
    if not all([twilio_sid, twilio_token, twilio_from, phone]): return False
    try:
        from twilio.rest import Client
        client = Client(twilio_sid, twilio_token)
        if decision == "APPROVED":
            msg = ("Dear " + name + ", Your loan of Rs" +
                   "{:,.0f}".format(amount) + " is APPROVED! EMI: Rs" +
                   "{:,.0f}".format(emi) + "/month. - National Finance Bank")
        elif decision == "REJECTED":
            msg = ("Dear " + name +
                   ", Your loan application was not approved. "
                   "Check email for details. - National Finance Bank")
        else:
            msg = ("Dear " + name +
                   ", Your loan is under review. "
                   "Please submit required documents. - National Finance Bank")
        to_num = ("whatsapp:" + phone
                  if not phone.startswith("whatsapp:") else phone)
        client.messages.create(body=msg, from_=twilio_from, to=to_num)
        alog("WHATSAPP", "Sent to " + phone)
        return True
    except Exception as e:
        alog("ERROR", "WhatsApp failed: " + str(e)); return False


# ═══════════════════════════════════════════
# Follow-up Agent
# ═══════════════════════════════════════════

def run_followup_agent(guser, gpass):
    followups = get_followups()
    today     = datetime.now().date()
    alog("FOLLOWUP", "Checking " + str(len(followups)) + " pending reviews")
    for aid, fu in followups.items():
        if fu.get("resolved"): continue
        submitted    = datetime.fromisoformat(fu["submitted"]).date()
        days_elapsed = (today - submitted).days
        email        = fu.get("email", "")
        name         = fu.get("name", "Applicant")

        if days_elapsed >= 8 and not fu.get("auto_rejected"):
            if email and guser and gpass:
                try:
                    msg = MIMEMultipart()
                    msg["From"] = guser; msg["To"] = email
                    msg["Subject"] = "Loan Application Auto-Rejected - National Finance Bank"
                    msg.attach(MIMEText(
                        "Dear " + name + ",\n\nYour application has been "
                        "auto-rejected (documents not submitted within 7 days).\n\n"
                        "Warm regards,\nNational Finance Bank AI Agent", "plain"))
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                        s.login(guser, gpass); s.send_message(msg)
                    alog("FOLLOWUP", "Auto-rejected " + name)
                except Exception as e:
                    alog("ERROR", "Follow-up email failed: " + str(e))
            fu["auto_rejected"] = True; fu["resolved"] = True
            save_followup(aid, fu)
            continue

        reminder = None
        if   days_elapsed == 7 and not fu.get("day7_sent"):
            reminder = "final"; fu["day7_sent"] = True
        elif days_elapsed == 3 and not fu.get("day3_sent"):
            reminder = "second"; fu["day3_sent"] = True
        elif days_elapsed == 1 and not fu.get("day1_sent"):
            reminder = "first"; fu["day1_sent"] = True

        if reminder and email and guser and gpass:
            try:
                msg = MIMEMultipart()
                msg["From"] = guser; msg["To"] = email
                msg["Subject"] = ("Reminder " + str(days_elapsed) +
                                  "/7: Documents Required - National Finance Bank")
                msg.attach(MIMEText(
                    "Dear " + name + ",\n\nThis is your " + reminder +
                    " reminder. Please submit required documents.\n\n"
                    "Warm regards,\nNational Finance Bank AI Agent", "plain"))
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                    s.login(guser, gpass); s.send_message(msg)
                alog("FOLLOWUP", "Sent day-" + str(days_elapsed) +
                     " reminder to " + name)
            except Exception as e:
                alog("ERROR", "Reminder failed: " + str(e))
            save_followup(aid, fu)


# ═══════════════════════════════════════════
# Daily Summary Email
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
        alog("SUMMARY", "No applications today — skipping summary")
        return

    total    = len(df)
    approved = (len(df[df["decision"]=="APPROVED"])
                if "decision" in df.columns else 0)
    rejected = (len(df[df["decision"]=="REJECTED"])
                if "decision" in df.columns else 0)
    review   = (len(df[df["decision"]=="MANUAL REVIEW"])
                if "decision" in df.columns else 0)
    fraud    = (len(df[df["fraud_flag"]==1])
                if "fraud_flag" in df.columns else 0)
    avg_risk = (round(df["default_prob"].mean(), 1)
                if "default_prob" in df.columns and total > 0 else 0)

    body = "\n".join([
        "DAILY LOAN AGENT REPORT — " + str(today),
        "="*45, "",
        "Applications Processed : " + str(total),
        "Approved               : " + str(approved),
        "Rejected               : " + str(rejected),
        "Manual Review          : " + str(review),
        "Fraud Flagged          : " + str(fraud),
        "Avg Default Risk       : " + str(avg_risk) + "%",
        "", "AI Loan Processing Agent | National Finance Bank",
    ])

    try:
        msg = MIMEMultipart()
        msg["From"] = guser; msg["To"] = guser
        msg["Subject"] = ("Daily Loan Agent Report — " + str(today) +
                          " | " + str(total) + " applications")
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(guser, gpass); s.send_message(msg)
        alog("SUMMARY", "Daily report sent to " + guser)
    except Exception as e:
        alog("ERROR", "Summary email failed: " + str(e))


# ═══════════════════════════════════════════
# Adaptive Learning
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
        alog("LEARN", "Not enough feedback yet (" + str(len(df)) + "/10)")
        return model, False

    wrong    = len(df[df["correct"] == 0])
    accuracy = round((1 - wrong / len(df)) * 100, 1)
    alog("LEARN", "Feedback accuracy: " + str(accuracy) + "%")

    if accuracy >= 80:
        alog("LEARN", "Accuracy OK — no retraining needed")
        return model, False

    alog("LEARN", "Accuracy below 80% — retraining!")
    try:
        conn2  = sqlite3.connect(DB_FILE)
        dec_df = pd.read_sql("SELECT * FROM decisions LIMIT 500", conn2)
        conn2.close()

        if len(dec_df) < 20:
            alog("LEARN", "Not enough data to retrain")
            return model, False

        X_rows, y_rows = [], []
        for _, row in dec_df.iterrows():
            try:
                la   = float(row.get("loan_amount", 500000))
                ac   = float(row.get("creditworth", 75)) / 100.0
                prob = float(row.get("default_prob", 30)) / 100.0
                feat = {f: FM.get(f, 0) for f in FN}
                feat["EXT_SOURCE_MEAN"] = ac
                feat["EXT_SOURCE_1"]    = ac + np.random.normal(0, 0.02)
                feat["EXT_SOURCE_2"]    = ac + np.random.normal(0, 0.02)
                feat["EXT_SOURCE_3"]    = ac + np.random.normal(0, 0.02)
                feat["AMT_CREDIT"]      = la
                X_rows.append([feat.get(f, 0) for f in FN])
                y_rows.append(1 if prob > 0.5 else 0)
            except Exception:
                continue

        if len(X_rows) < 20:
            return model, False

        X = np.array(X_rows); y = np.array(y_rows)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # FIX: use_label_encoder removed in XGBoost 2.0+
        new_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric='logloss', random_state=42)
        new_model.fit(X_tr, y_tr)

        new_auc = round(roc_auc_score(y_val, new_model.predict_proba(X_val)[:,1]), 3)
        old_auc = round(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]), 3)
        alog("LEARN", "Old AUC: " + str(old_auc) + " | New AUC: " + str(new_auc))

        if new_auc >= old_auc:
            new_model.save_model(MODEL_FILE)
            conn3 = sqlite3.connect(DB_FILE)
            conn3.execute("""INSERT INTO model_versions
                (version,auc,trained_on,timestamp) VALUES (?,?,?,?)""",
                ("v_" + datetime.now().strftime("%Y%m%d_%H%M"),
                 new_auc, len(X_rows), datetime.now().isoformat()))
            conn3.commit(); conn3.close()
            alog("LEARN", "Model improved! AUC " + str(new_auc))
            return new_model, True
        else:
            alog("LEARN", "New model worse — keeping original")
            return model, False

    except Exception as e:
        alog("ERROR", "Retraining failed: " + str(e))
        return model, False


# ═══════════════════════════════════════════
# MAIN AGENT LOOP
# ═══════════════════════════════════════════

def run_once(config, model_ref, FN, FM):
    alog("AGENT", "=" * 40)
    alog("AGENT", "Cycle started at " + datetime.now().strftime("%H:%M:%S"))

    repo_rate = get_rbi_rate()
    processed = get_processed_ids()
    found = new_proc = emails = fraud_count = 0

    df1    = scrape_google_sheet(config.get("sheet_url", ""))
    df2    = scrape_csv_url(config.get("csv_url", ""))
    all_df = merge_sources(df1, df2)

    if all_df.empty:
        alog("PERCEIVE", "No applications found")
        log_run(0, 0, 0, 0, "NO_DATA"); return

    found = len(all_df)
    for _, row in all_df.iterrows():
        app = parse_row(row); aid = app["app_id"]
        if aid in processed: continue

        alog("PROCESS", "Processing: " + app["name"] +
             " [" + app["_source"] + "]")

        is_fraud, fraud_reason = check_fraud(app, all_df)
        if is_fraud:
            fraud_count += 1
            log_decision(aid, app["name"], app["email"], app.get("phone",""),
                         "REJECTED", 99.0, app["loan_amount"], 0, None, 0,
                         True, fraud_reason, app["_source"])
            save_processed_id(aid)
            gu = config.get("gmail_user",""); gp = config.get("gmail_pass","")
            if gu and gp:
                try:
                    msg = MIMEMultipart()
                    msg["From"] = gu; msg["To"] = gu
                    msg["Subject"] = "FRAUD ALERT: " + app["name"]
                    msg.attach(MIMEText(
                        "FRAUD DETECTED\nApplicant: " + app["name"] +
                        "\nReason: " + fraud_reason, "plain"))
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                        s.login(gu, gp); s.send_message(msg)
                except Exception:
                    pass
            continue

        try:
            result = decide(app, model_ref[0], FN, FM, repo_rate)
            alog("DECIDE", app["name"] + " -> " + result["decision"] +
                 " (" + str(result["default_prob"]) + "% risk)")
        except Exception as e:
            alog("ERROR", "Decision failed: " + str(e)); continue

        log_decision(aid, app["name"], app["email"], app.get("phone",""),
                     result["decision"], result["default_prob"],
                     app["loan_amount"], result["approved_amount"],
                     result["interest_rate"], result["emi"],
                     False, "", app["_source"])

        gu = config.get("gmail_user",""); gp = config.get("gmail_pass","")
        if app["email"] and "@" in app["email"] and gu and gp:
            sent = send_email(app["email"], app["name"], result, app,
                              gu, gp, attach_pdf=True, feature_names=FN)
            if sent:
                conn = sqlite3.connect(DB_FILE)
                conn.execute(
                    "UPDATE decisions SET email_sent=1 WHERE app_id=?", (aid,))
                conn.commit(); conn.close()
                emails += 1

        if app.get("phone") and TWILIO_SID:
            send_whatsapp(app.get("phone",""), app["name"],
                          result["decision"], result["approved_amount"],
                          result["emi"], TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM)

        if result["decision"] == "MANUAL REVIEW":
            save_followup(aid, {
                "email": app["email"], "name": app["name"],
                "submitted": datetime.now().isoformat(), "resolved": False})

        save_processed_id(aid); new_proc += 1

    run_followup_agent(config.get("gmail_user",""), config.get("gmail_pass",""))

    if new_proc > 0 and _G["run_count"] % 10 == 0:
        new_model, improved = check_and_retrain(model_ref[0], FN, FM)
        if improved:
            model_ref[0] = new_model
            alog("LEARN", "Model self-improved!")

    now = datetime.now()
    if now.hour == 9 and not _G["summary_sent"]:
        send_daily_summary(config.get("gmail_user",""), config.get("gmail_pass",""))
        _G["summary_sent"] = True
    elif now.hour != 9:
        _G["summary_sent"] = False

    log_run(found, new_proc, emails, fraud_count, "SUCCESS")
    alog("AGENT", "Done: " + str(new_proc) + " new | " +
         str(emails) + " emails | " + str(fraud_count) + " fraud")
    _G["run_count"] += 1


def log_run(found, processed, emails, fraud, status):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""INSERT INTO agent_runs
        (run_time,apps_found,processed,emails_sent,fraud_flagged,status)
        VALUES (?,?,?,?,?,?)""",
        (datetime.now().isoformat(), found, processed, emails, fraud, status))
    conn.commit(); conn.close()


def start_loop(config, model_ref, FN, FM):
    _G["running"] = True
    def loop():
        while _G["running"]:
            run_once(config, model_ref, FN, FM)
            _G["last_run"] = datetime.now().strftime("%H:%M:%S")
            time.sleep(INTERVAL)
    threading.Thread(target=loop, daemon=True).start()


# ═══════════════════════════════════════════
# ✅ FIX 1: BOOTSTRAP MODEL (auto-trains on first deploy)
# ═══════════════════════════════════════════

FEATURE_NAMES = [
    "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","EXT_SOURCE_MEAN",
    "DEBT_TO_INCOME","LOAN_TO_INCOME","ANNUITY_TO_CREDIT",
    "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
    "AGE_YEARS","YEARS_EMPLOYED","IS_EMPLOYED",
    "INCOME_PER_PERSON","CNT_FAM_MEMBERS",
]


def bootstrap_train_model(save_path="."):
    """
    Train a synthetic XGBoost model and save the three required artefacts:
      - loan_model.json
      - feature_names.pkl
      - feature_medians.pkl
    Called automatically on first deploy when files are absent.
    """
    np.random.seed(42)
    n = 6000

    cs1     = np.clip(np.random.beta(8, 2, n), 0.01, 0.99)
    cs2     = np.clip(cs1 + np.random.normal(0, 0.03, n), 0.01, 0.99)
    cs3     = np.clip(cs1 + np.random.normal(0, 0.03, n), 0.01, 0.99)
    cs_mean = (cs1 + cs2 + cs3) / 3
    income  = np.random.lognormal(13.5, 0.7, n)
    loan    = income * np.random.uniform(0.5, 5, n)
    annuity = loan * 0.02
    age     = np.random.randint(22, 63, n).astype(float)
    yrs_emp = np.clip(np.random.exponential(4, n), 0, 40)
    employed= np.random.binomial(1, 0.82, n).astype(float)
    family  = np.random.randint(1, 7, n).astype(float)

    X = np.column_stack([
        cs1, cs2, cs3, cs_mean,
        annuity / (income + 1),
        loan    / (income + 1),
        annuity / (loan   + 1),
        income, loan, annuity,
        age, yrs_emp, employed,
        income / (family + 1), family,
    ])

    risk = ((1 - cs_mean) * 0.55
            + (loan / (income + 1)) * 0.15
            + (1 - employed) * 0.25
            + np.random.normal(0, 0.04, n))
    y = (risk > 0.38).astype(int)

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ FIX 2: No use_label_encoder (removed in XGBoost 2.0+)
    mdl = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        eval_metric="logloss", random_state=42)
    mdl.fit(X_tr, y_tr)

    mf   = os.path.join(save_path, "loan_model.json")
    ff   = os.path.join(save_path, "feature_names.pkl")
    fm_p = os.path.join(save_path, "feature_medians.pkl")

    mdl.save_model(mf)
    with open(ff,   "wb") as f: pickle.dump(FEATURE_NAMES, f)
    medians = {n: float(np.median(X[:, i]))
               for i, n in enumerate(FEATURE_NAMES)}
    with open(fm_p, "wb") as f: pickle.dump(medians, f)

    return mdl, FEATURE_NAMES, medians


@st.cache_resource
def load_model():
    """
    Try to load pre-trained artefacts.
    If missing (fresh Streamlit Cloud deploy), auto-train a bootstrap model.
    """
    for p in [".", "/app", "/mount/src/loan-approval-agent"]:
        mf   = os.path.join(p, "loan_model.json")
        ff   = os.path.join(p, "feature_names.pkl")
        fm_p = os.path.join(p, "feature_medians.pkl")
        if all(os.path.exists(x) for x in [mf, ff, fm_p]):
            m = xgb.XGBClassifier()
            m.load_model(mf)
            with open(ff,   "rb") as f: fn   = pickle.load(f)
            with open(fm_p, "rb") as f: fmed = pickle.load(f)
            return m, fn, fmed

    # ← no pre-trained files found: train from scratch (~8 seconds)
    st.toast("⚙️ First launch — training bootstrap model (~8s)…", icon="🤖")
    mdl, fn, fmed = bootstrap_train_model(".")
    return mdl, fn, fmed


# ═══════════════════════════════════════════
# UI
# ═══════════════════════════════════════════

init_db()
try:
    model, FN, FM = load_model()
    model_ok = True
except Exception as e:
    model_ok  = False
    model_err = str(e)

for key, val in [("running",False),("last_run","Never"),
                 ("alog",[]),("run_count",0)]:
    if key not in st.session_state:
        st.session_state[key] = val
if "model_ref" not in st.session_state and model_ok:
    st.session_state.model_ref = [model]

creds_ok  = bool(GMAIL_USER and GMAIL_PASS and SHEET_URL)
sheet_url = SHEET_URL; guser = GMAIL_USER; gpass = GMAIL_PASS

st.markdown("""
<div class="hero">
<h1>🤖 Autonomous Loan Approval Agent v4.0</h1>
<p>9 Agentic Features: Perceive &rarr; Reason &rarr; Act &rarr; Remember &rarr; Repeat
&nbsp;|&nbsp; Every 60 minutes</p>
</div>""", unsafe_allow_html=True)

if not model_ok:
    st.error("❌ Model failed to load: " + model_err)
    st.info("CWD files: " + str(os.listdir(".")))
    st.stop()

# Auto-refresh stats every 30 seconds when agent is running
if _G["running"]:
    import time as _time
    refresh_interval = 30
    st_autorefresh_placeholder = st.empty()
    # Use st.rerun via a time-based query param trick
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = _time.time()
    if _time.time() - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = _time.time()
        st.rerun()

col_refresh, col_empty = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Stats"):
        st.rerun()

stats = get_stats()
s1,s2,s3,s4,s5,s6 = st.columns(6)
s1.metric("Total Processed", stats["total"])
s2.metric("Approved",        stats["approved"])
s3.metric("Rejected",        stats["rejected"])
s4.metric("Under Review",    stats["review"])
s5.metric("Emails Sent",     stats["emails"])
s6.metric("Fraud Flagged",   stats["fraud"])
st.divider()

tab1,tab2,tab3,tab4,tab5 = st.tabs(
    ["🎛️ Control","📊 Dashboard","📋 Log","🧠 Learning","ℹ️ Features"])

# ── Control Tab ──────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### ⚙️ Agent Configuration")
        if creds_ok:
            st.success("Credentials loaded from Secrets")
            st.info("Sheet: ..." + sheet_url[-40:])
            st.info("Gmail: " + guser)
        else:
            st.warning("Enter credentials manually:")
            sheet_url = st.text_input("Google Sheet CSV URL", SHEET_URL)
            guser     = st.text_input("Gmail Address", GMAIL_USER)
            gpass     = st.text_input("App Password",  GMAIL_PASS, type="password")

        csv_url  = st.text_input("Extra CSV URL (optional)", "",
                                  help="Any public CSV URL as second source")
        st.markdown("**Upload CSV (optional):**")
        uploaded = st.file_uploader("Upload applications CSV", type=["csv"])

        st.markdown("**Twilio WhatsApp (optional):**")
        with st.expander("Configure WhatsApp"):
            t_sid  = st.text_input("Twilio SID",   TWILIO_SID)
            t_tok  = st.text_input("Twilio Token", TWILIO_TOKEN, type="password")
            t_from = st.text_input("Twilio From",  TWILIO_FROM)

        config = {
            "sheet_url":   sheet_url,
            "csv_url":     csv_url,
            "gmail_user":  guser,
            "gmail_pass":  gpass,
            "twilio_sid":  t_sid  if "t_sid"  in dir() else "",
            "twilio_token":t_tok  if "t_tok"  in dir() else "",
            "twilio_from": t_from if "t_from" in dir() else "",
        }

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶️ Run Once"):
                if uploaded: load_uploaded_csv(uploaded)
                with st.spinner("Agent running..."):
                    mr = st.session_state.get("model_ref", [model])
                    run_once(config, mr, FN, FM)
                st.success("Cycle complete!"); st.rerun()
        with b2:
            if not _G["running"]:
                if st.button("🚀 Auto (60min)"):
                    mr = st.session_state.get("model_ref", [model])
                    start_loop(config, mr, FN, FM)
                    st.success("Agent started!"); st.rerun()
            else:
                if st.button("⏹️ Stop"):
                    _G["running"] = False
                    st.warning("Stopped."); st.rerun()
        with b3:
            if st.button("📧 Send Summary"):
                with st.spinner("Sending daily summary..."):
                    send_daily_summary(guser, gpass)
                st.success("Summary sent to " + guser)

        status_class = "running" if _G["running"] else "stopped"
        status_icon  = "🟢 RUNNING" if _G["running"] else "⚪ Stopped"
        st.markdown(
            '<div class="' + status_class + '"><b>' + status_icon +
            '</b> | Last: ' + _G["last_run"] +
            ' | Runs: ' + str(_G["run_count"]) + '</div>',
            unsafe_allow_html=True)

    with right:
        st.markdown("### 🔄 9 Agentic Features")
        for title, desc in [
            ("1. 🌐 PERCEIVE — Multi-Source",
             "Scrapes Google Sheet + CSV URL + uploaded CSV simultaneously"),
            ("2. 🧠 REASON — XGBoost + SHAP",
             "ML model + 6 credit rules + explainable AI"),
            ("3. 📧 ACT — Email + PDF",
             "Decision email with full SHAP explanation PDF attached"),
            ("4. 💾 REMEMBER — SQLite",
             "All decisions, runs, and model versions stored"),
            ("5. 😴 REPEAT — 60 min",
             "Fully autonomous loop — zero human needed"),
            ("6. 🚨 FRAUD DETECTION",
             "Flags suspicious patterns before ML decision"),
            ("7. 📅 FOLLOW-UP AGENT",
             "Auto follows up manual reviews — rejects after 7 days"),
            ("8. 🔔 DAILY REPORT",
             "Emails manager summary at 9AM daily"),
            ("9. 🧠 ADAPTIVE LEARNING",
             "Retrains XGBoost when accuracy drops below 80%"),
        ]:
            st.markdown(
                '<div class="acard"><h4>' + title + '</h4>'
                '<p>' + desc + '</p></div>',
                unsafe_allow_html=True)

# ── Dashboard Tab ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📈 Live Dashboard")
    if st.button("🔄 Refresh Dashboard"): st.rerun()

    daily = get_daily_stats()
    if not daily.empty and len(daily) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        fig.patch.set_facecolor("#0f172a")
        for ax in axes:
            ax.set_facecolor("#1e293b")
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#334155")
            ax.spines["left"].set_color("#334155")
            ax.tick_params(colors="#94a3b8", labelsize=8)

        x = range(len(daily))
        axes[0].bar(x, daily["approved"], color="#10b981",
                    label="Approved", alpha=0.8)
        axes[0].bar(x, daily["rejected"], bottom=daily["approved"],
                    color="#ef4444", label="Rejected", alpha=0.8)
        axes[0].set_title("Daily Applications", color="#00d4aa", fontsize=10)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(daily["date"].tolist(),
                                rotation=45, fontsize=7)
        axes[0].legend(fontsize=8, labelcolor="#94a3b8",
                       facecolor="#1e293b", edgecolor="#334155")

        axes[1].plot(x, daily["avg_risk"], color="#f59e0b",
                     linewidth=2, marker="o", markersize=4)
        axes[1].fill_between(x, daily["avg_risk"], alpha=0.2, color="#f59e0b")
        axes[1].axhline(y=55, color="#ef4444", linestyle="--",
                        alpha=0.5, linewidth=1)
        axes[1].set_title("Avg Default Risk %", color="#00d4aa", fontsize=10)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(daily["date"].tolist(),
                                rotation=45, fontsize=7)

        ap_rate = daily["approved"] / (daily["total"] + 0.001) * 100
        axes[2].plot(x, ap_rate, color="#00d4aa",
                     linewidth=2, marker="s", markersize=4)
        axes[2].fill_between(x, ap_rate, alpha=0.2, color="#00d4aa")
        axes[2].set_title("Approval Rate %", color="#00d4aa", fontsize=10)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(daily["date"].tolist(),
                                rotation=45, fontsize=7)

        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Run the agent on multiple days to see trend charts.")

    st.divider()
    st.markdown("### 📊 All Decisions")
    df_dec = get_decisions()
    if not df_dec.empty:
        def cd(v):
            if v == "APPROVED":     return "color:#10b981;font-weight:bold"
            elif v == "REJECTED":   return "color:#ef4444;font-weight:bold"
            return "color:#f59e0b;font-weight:bold"
        cols = [c for c in ["app_id","applicant","email","decision",
                             "default_prob","loan_amount","approved_amt",
                             "fraud_flag","email_sent","source","timestamp"]
                if c in df_dec.columns]
        st.dataframe(
            df_dec[cols].style.map(cd, subset=["decision"]),
            width="stretch")
        st.download_button("📥 Download CSV",
                           data=df_dec.to_csv(index=False),
                           file_name="decisions.csv", mime="text/csv")
    else:
        st.info("No decisions yet.")

# ── Log Tab ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Agent Activity Log")
    log_lines = get_log()
    if log_lines:
        html = ""
        for l in reversed(log_lines):
            l = l.strip()
            if   "APPROVED"      in l and "[DECIDE]"  in l: c = "#10b981"
            elif "REJECTED"      in l and "[DECIDE]"  in l: c = "#ef4444"
            elif "MANUAL REVIEW" in l and "[DECIDE]"  in l: c = "#f59e0b"
            elif "[ERROR]"       in l: c = "#ef4444"
            elif "[EMAIL]"       in l: c = "#60a5fa"
            elif "[FRAUD]"       in l: c = "#f87171"
            elif "[LEARN]"       in l: c = "#a78bfa"
            elif "[FOLLOWUP]"    in l: c = "#fb923c"
            elif "[SUMMARY]"     in l: c = "#34d399"
            elif "[AGENT]"       in l: c = "#00d4aa"
            elif "[RBI]"         in l: c = "#fbbf24"
            else:                      c = "#94a3b8"
            html += "<span style='color:" + c + "'>" + l + "</span><br>"
        st.markdown('<div class="log-box">' + html + '</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="log-box">No activity yet.</div>',
                    unsafe_allow_html=True)

    if st.button("🔄 Refresh Log"): st.rerun()
    st.divider()
    st.markdown("### 🏃 Agent Run History")
    conn = sqlite3.connect(DB_FILE)
    try:
        df_r = pd.read_sql(
            "SELECT * FROM agent_runs ORDER BY run_time DESC LIMIT 20", conn)
    except Exception:
        df_r = pd.DataFrame()
    conn.close()
    if not df_r.empty: st.dataframe(df_r, width="stretch")
    else:              st.info("No runs yet.")

# ── Learning Tab ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🧠 Adaptive Learning Engine")
    conn = sqlite3.connect(DB_FILE)
    try:
        mv = pd.read_sql(
            "SELECT * FROM model_versions ORDER BY timestamp DESC", conn)
    except Exception:
        mv = pd.DataFrame()
    conn.close()

    if not mv.empty:
        st.success("Model has been retrained " + str(len(mv)) + " times")
        st.dataframe(mv, width="stretch")
    else:
        st.info("Model not yet retrained. Needs 10+ feedback entries < 80% accuracy.")

    st.divider()
    st.markdown("**Submit Feedback on Recent Decisions:**")
    df_recent = get_decisions(10)
    if not df_recent.empty:
        for _, row in df_recent.iterrows():
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.write(str(row.get("applicant","?")) + " — " +
                     str(row.get("decision","?")))
            with c2:
                if st.button("✅ Correct", key="c_" + str(row.get("id",""))):
                    conn = sqlite3.connect(DB_FILE)
                    conn.execute(
                        "INSERT OR IGNORE INTO feedback "
                        "(app_id,correct,timestamp) VALUES (?,?,?)",
                        (row.get("app_id",""), 1, datetime.now().isoformat()))
                    conn.commit(); conn.close()
                    st.success("Saved!")
            with c3:
                if st.button("❌ Wrong", key="w_" + str(row.get("id",""))):
                    conn = sqlite3.connect(DB_FILE)
                    conn.execute(
                        "INSERT OR IGNORE INTO feedback "
                        "(app_id,correct,timestamp) VALUES (?,?,?)",
                        (row.get("app_id",""), 0, datetime.now().isoformat()))
                    conn.commit(); conn.close()
                    st.warning("Flagged!")
    else:
        st.info("No decisions yet to rate.")

    if st.button("🔁 Force Retrain Now"):
        with st.spinner("Retraining model..."):
            mr = st.session_state.get("model_ref", [model])
            new_m, improved = check_and_retrain(mr[0], FN, FM)
            if improved:
                st.session_state["model_ref"] = [new_m]
                st.success("Model improved and updated!")
            else:
                st.info("No improvement found — keeping current model.")

# ── Features Tab ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### ℹ️ All 9 Agentic Features Explained")
    for title, desc in {
        "1. Multi-Source Perception":
            "Agent simultaneously scrapes Google Sheet, a second CSV URL, "
            "and accepts uploaded CSV files. All sources are merged automatically.",
        "2. XGBoost + SHAP Reasoning":
            "ML model predicts default probability. SHAP explains exactly "
            "which factors drove each decision.",
        "3. Email + PDF Report":
            "Every applicant receives a professional email with a detailed PDF "
            "report showing their SHAP explanation chart and top decision factors.",
        "4. SQLite Memory":
            "Every decision, agent run, feedback entry, and model version "
            "is stored permanently in a local database.",
        "5. 60-Minute Loop":
            "Agent runs fully autonomously every hour — scrapes, decides, "
            "emails, logs — with zero human intervention.",
        "6. Fraud Detection":
            "Before ML decision, agent checks for identical credit scores, "
            "multiple same-email submissions, age-income mismatch, "
            "and unrealistic values.",
        "7. Follow-up Agent":
            "Manual review cases are tracked daily. Agent sends reminders on "
            "Day 1, Day 3, Day 7, then auto-rejects on Day 8.",
        "8. Daily Summary Email":
            "Every morning at 9AM, agent emails the manager a full report: "
            "totals, approval rate, avg risk, high-risk alerts.",
        "9. Adaptive Learning":
            "Agent monitors feedback accuracy. If it drops below 80%, "
            "it automatically retrains the XGBoost model on recent decisions.",
    }.items():
        with st.expander(title):
            st.write(desc)

st.caption("Autonomous Loan Agent v4.0 | " +
           datetime.now().strftime("%Y-%m-%d %H:%M") +
           " | 9 Agentic Features Active")
