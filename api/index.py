import os
import joblib
import pandas as pd
import pytz
import sqlite3
import pdfplumber
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# --- 1. LOCAL MODULE IMPORTS ---
# These must be in your repository folders
from database.db_connection import get_db
from database.init_db import init_database
from utils.risk_score import classify_risk
from utils.pdf_generator import create_visual_report
from config import UPLOAD_FOLDER, REPORT_FOLDER

app = Flask(__name__)
app.secret_key = "bnpl_ultra_secure_key_telangana_2026"

# --- 2. VERCEL SERVERLESS PATH ADJUSTMENTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On Vercel, we must use /tmp for any file writing (Uploads/Reports)
if os.environ.get('VERCEL'):
    REPORT_FOLDER = "/tmp/reports"
    UPLOAD_FOLDER = "/tmp/uploads"
    PROFILE_PIC_DIR = "/tmp/profile_pics"
else:
    # Local development paths
    FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
    PROFILE_PIC_DIR = os.path.join(FRONTEND_DIR, "assets", "profile_pics")

os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROFILE_PIC_DIR, exist_ok=True)

# --- 3. GLOBAL CONFIGURATION ---
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024
)

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
IST = pytz.timezone("Asia/Kolkata")

# --- 4. DATABASE AUTO-MIGRATOR ---
def apply_migrations():
    db = get_db()
    schema_map = {
        "predictions": [
            ("filename", "TEXT"), ("high_risk_count", "INTEGER"), 
            ("low_risk_count", "INTEGER"), ("total_count", "INTEGER"),
            ("probability", "REAL"), ("report_path", "TEXT"),
            ("income", "REAL"), ("loan", "REAL"), ("emi", "REAL"),
            ("tenure", "INTEGER"), ("ontime", "INTEGER"), ("delays", "INTEGER"),
            ("credit", "INTEGER"), ("risk", "TEXT"), ("source", "TEXT")
        ],
        "users": [
            ("age", "INTEGER"), ("address", "TEXT"), 
            ("profile_photo", "TEXT"), ("mobile", "TEXT"),
            ("last_login", "TIMESTAMP")
        ]
    }
    for table, columns in schema_map.items():
        for col_name, col_type in columns:
            try:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass 
    db.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, name TEXT, email TEXT, 
        category TEXT, details TEXT, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    db.commit()
    db.close()

# Initialize System
init_database()
apply_migrations()

# --- 5. ML MODEL LOADING ---
try:
    # Path logic to find models whether running locally or in /api/
    MODEL_DIR = os.path.join(BASE_DIR, "models") if not os.path.exists(os.path.join(BASE_DIR, "..", "models")) else os.path.join(BASE_DIR, "..", "models")
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("✅ System Ready: ML Random Forest Model Loaded")
except Exception as e:
    print(f"❌ Critical Error: ML Assets could not be loaded: {e}")

# --- 6. CORE UTILITY FUNCTIONS ---

def format_time(ts):
    try:
        dt = datetime.fromisoformat(ts.replace(" ", "T"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=pytz.UTC)
        ist_dt = dt.astimezone(IST)
        return ist_dt.strftime("%d-%m-%Y %I:%M %p")
    except Exception:
        return ts

def generate_smart_suggestion(row):
    try:
        income = float(row.get('income', 0) or 0)
        loan = float(row.get('loan', 0) or 0)
        emi = float(row.get('emi', 0) or 0)
        tenure = float(row.get('tenure', 1) or 1)
        credit = float(row.get('credit', 0) or 0)
        delays = float(row.get('delays', 0) or 0)

        if income > 0 and loan > (income * tenure):
            return f"Capacity exceeded. Max recommended: ₹{int(income * tenure * 0.7)}."
        if income > 0 and emi > (income * 0.5):
            return "High EMI burden (>50%). Consider a longer tenure."
        if credit < 650:
            return "CIBIL score low. Avoid fresh credit inquiries for 6 months."
        if delays > 3:
            return "Frequent delays detected. Focus on consistent repayment history."
        return "Financial profile appears healthy. Proceeding with standard verification."
    except Exception:
        return "Manual review required."

# --- 7. AUTHENTICATION ---

@app.route("/register", methods=["POST"])
def register():
    data, db = request.json, get_db()
    hashed_pw = generate_password_hash(data.get("password"))
    try:
        db.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                   (data.get("name"), data.get("email"), hashed_pw))
        db.commit()
        return jsonify(success=True)
    except Exception:
        return jsonify(success=False, message="Email already registered"), 400

@app.route("/login", methods=["POST"])
def login():
    data, db = request.json, get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (data.get("email"),)).fetchone()
    if user and check_password_hash(user["password"], data.get("password")):
        session.permanent = True
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        db.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(IST), user["id"]))
        db.commit()
        return jsonify(success=True, name=user["name"])
    return jsonify(success=False, message="Incorrect email or password"), 401

@app.route("/logout")
def logout():
    session.clear()
    return jsonify(success=True)

# --- 8. USER PROFILE ---

@app.route("/user/profile")
def get_profile():
    if "user_id" not in session: return jsonify(error="Unauthorized"), 401
    db = get_db()
    u = db.execute("SELECT name, email, age, address, profile_photo, mobile FROM users WHERE id=?", 
                   (session["user_id"],)).fetchone()
    return jsonify(name=u[0], email=u[1], age=u[2] or "", address=u[3] or "", photo=u[4] or None, mobile=u[5] or "")

@app.route("/user/update", methods=["POST"])
def update_profile():
    if "user_id" not in session: return jsonify(success=False, error="Unauthorized"), 401
    name, age, address, mobile = request.form.get('name'), request.form.get('age'), request.form.get('address'), request.form.get('mobile')
    photo, db = request.files.get('photo'), get_db()
    try:
        if photo:
            filename = secure_filename(f"user_{session['user_id']}_{photo.filename}")
            photo.save(os.path.join(PROFILE_PIC_DIR, filename))
            db.execute("UPDATE users SET profile_photo = ? WHERE id = ?", (filename, session['user_id']))
        db.execute("UPDATE users SET name=?, age=?, address=?, mobile=? WHERE id=?", (name, age, address, mobile, session['user_id']))
        db.commit()
        session["user_name"] = name
        return jsonify(success=True)
    except Exception as e: return jsonify(success=False, error=str(e)), 500

# --- 9. ML PREDICTIONS ---

@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session: return jsonify(error="Unauthorized"), 401
    data = request.json
    try:
        data['name'] = session.get("user_name", "User")
        vals = {k: float(data.get(k, 0) or 0) for k in ["income", "loan", "emi", "tenure", "ontime", "delays", "credit"]}
        income_s = vals["income"] if vals["income"] > 0 else 1
        total_p = (vals["ontime"] + vals["delays"]) if (vals["ontime"] + vals["delays"]) > 0 else 1
        input_df = pd.DataFrame([{**vals, "emi_income_ratio": vals["emi"] / income_s, "delay_ratio": vals["delays"] / total_p}])
        
        feature_order = ['income', 'loan', 'emi', 'tenure', 'ontime', 'delays', 'credit', 'emi_income_ratio', 'delay_ratio']
        scaled_input = scaler.transform(input_df[feature_order])
        prob = float(model.predict_proba(scaled_input)[0][1])
        risk = classify_risk(prob)
        suggestion = generate_smart_suggestion(vals)

        if (vals["emi"]/income_s > 0.60 and vals["credit"] < 600) or (vals["ontime"] == 0 and vals["delays"] >= 6):
            risk = "HIGH RISK"

        pdf_name = f"report_{session['user_id']}_{os.urandom(3).hex()}.pdf"
        data['timestamp'] = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p")
        create_visual_report(data, prob, risk, os.path.join(REPORT_FOLDER, pdf_name))

        db = get_db()
        db.execute("""INSERT INTO predictions (user_id, source, income, loan, emi, tenure, ontime, delays, credit, risk, probability, report_path, created_at) 
                      VALUES (?, 'manual', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (session["user_id"], vals["income"], vals["loan"], vals["emi"], vals["tenure"], vals["ontime"], vals["delays"], vals["credit"], risk, prob, pdf_name, datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')))
        db.commit()
        return jsonify(success=True, risk=risk, probability=prob, report_url=pdf_name, suggestion=suggestion)
    except Exception as e: return jsonify(success=False, error=str(e)), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" not in session: return jsonify(error="Unauthorized"), 401
    file = request.files.get("file")
    if not file: return jsonify(error="No file uploaded"), 400
    filename = file.filename.lower()
    try:
        if filename.endswith('.csv'): df = pd.read_csv(file)
        elif filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                table = pdf.pages[0].extract_table({"horizontal_strategy": "text", "vertical_strategy": "text"})
                if not table: return jsonify(success=False, error="No table found"), 400
                df = pd.DataFrame(table[1:], columns=table[0])
        else: return jsonify(error="Unsupported format"), 400

        df.columns = [str(c).lower().replace('\n', ' ').strip() for c in df.columns]
        numeric_cols = ['income', 'loan', 'emi', 'tenure', 'ontime', 'delays', 'credit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,\s]', '', regex=True), errors='coerce').fillna(0)
            else: df[col] = 0 

        df['emi_income_ratio'] = df['emi'] / (df['income'] + 1)
        df['delay_ratio'] = df['delays'] / (df['ontime'] + df['delays'] + 1)
        features = ['income', 'loan', 'emi', 'tenure', 'ontime', 'delays', 'credit', 'emi_income_ratio', 'delay_ratio']
        probs = model.predict_proba(scaler.transform(df[features]))[:, 1]
        
        e_mask, p_mask, d_mask = probs < 0.3, (probs >= 0.3) & (probs < 0.7), probs >= 0.7
        db = get_db()
        db.execute("INSERT INTO predictions (user_id, source, filename, high_risk_count, low_risk_count, total_count, created_at) VALUES (?, 'batch', ?, ?, ?, ?, ?)", 
                   (session["user_id"], file.filename, int(sum(d_mask)), int(sum(e_mask)), len(df), datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')))
        db.commit()
        return jsonify(success=True, total=len(df), low_risk=int(sum(e_mask)), high_risk=int(sum(d_mask)))
    except Exception as e: return jsonify(success=False, error=str(e)), 500

# --- 10. HISTORY & FEEDBACK ---

@app.route("/user/prediction-history")
def history():
    if "user_id" not in session: return jsonify(error="Unauthorized"), 401
    db = get_db()
    rows = db.execute("SELECT id, created_at, source, income, filename, risk, high_risk_count, low_risk_count, total_count, probability, report_path FROM predictions WHERE user_id=? ORDER BY created_at DESC", (session["user_id"],)).fetchall()
    return jsonify([{"id": r[0], "created_at": format_time(r[1]), "type": r[2], "risk": r[5], "report": r[10]} for r in rows])

@app.route("/user/feedback", methods=["POST"])
def submit_feedback():
    if "user_id" not in session: return jsonify(success=False, error="Unauthorized"), 401
    data, db = request.json, get_db()
    db.execute("INSERT INTO feedback (user_id, name, email, category, details) VALUES (?, ?, ?, ?, ?)", (session["user_id"], data.get('name'), data.get('email'), data.get('category'), data.get('details')))
    db.commit()
    return jsonify(success=True)

# --- 11. VERCEL / STATIC ---

@app.route("/download-report/<filename>")
def download(filename):
    return send_from_directory(REPORT_FOLDER, filename)

@app.route("/")
def index():
    return jsonify(status="online", message="BNPL Risk API IST 2026")

# EXPORT FOR VERCEL
app = app

if __name__ == "__main__":
    app.run(debug=True)