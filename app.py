# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, render_template_string
from functools import wraps
import pickle
import re
from datetime import datetime
import pytesseract
from PIL import Image
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import smtplib
from email.message import EmailMessage
import os

app = Flask(__name__)

# -------------------- CONFIG --------------------
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_with_a_strong_secret")
SECURITY_PASSWORD_SALT = os.environ.get("SECURITY_PASSWORD_SALT", "replace_with_random_salt")

EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.example.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USER = os.environ.get("EMAIL_USER", "you@example.com")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "your-email-password")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)

serializer = URLSafeTimedSerializer(app.secret_key)

# ---------- TESSERACT PATH ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- LOAD MODEL ----------
with open("model_artifact.pkl", "rb") as f:
    ART = pickle.load(f)
PIPELINE = ART["pipeline"]

# -------------------- DATABASE HELPERS --------------------
DB_PATH = "database.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT,
            cleaned TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Add user_id column for existing databases that pre-date this change
    cursor.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")

    conn.commit()
    conn.close()

init_db()

# -------------------- UTILITIES --------------------
def send_reset_email(to_email, token):
    reset_link = url_for('reset_password', token=token, _external=True)

    msg = EmailMessage()
    msg['Subject'] = 'Password reset for Fake News Detection'
    msg['From'] = EMAIL_FROM
    msg['To'] = to_email
    msg.set_content(f"Click here to reset your password:\n{reset_link}")

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def save_history(original, cleaned, prediction, confidence, timestamp, user_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (original, cleaned, prediction, confidence, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?)",
        (original, cleaned, prediction, confidence, timestamp, user_id)
    )
    conn.commit()
    conn.close()

# -------------------- AUTH HELPERS --------------------
def create_user(name, email, password):
    password_hash = generate_password_hash(password)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, created_at)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return row

def update_user_password(email, new_password):
    new_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email))
    conn.commit()
    conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


def process_registration(name, email, password):
    if not name or not email or not password:
        return False, "Please fill all fields."

    existing = get_user_by_email(email)
    if existing:
        return False, "Email already registered."

    uid = create_user(name, email, password)
    if uid:
        return True, "Registration successful! Please log in."

    return False, "Registration failed. Try again."


# ============================================================
# REGISTER ROUTES
# ============================================================

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password")

    if confirm_password is not None and password != confirm_password:
        return jsonify({"status": "error", "message": "Passwords do not match."})

    success, message = process_registration(name, email, password)
    status = "success" if success else "error"
    return jsonify({"status": status, "message": message})


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", form_data={})

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if password != confirm_password:
        return render_template("signup.html",
                               error="Passwords do not match.",
                               form_data={"name": name, "email": email})

    success, message = process_registration(name, email, password)
    if success:
        return redirect(url_for("login", registered=1))

    return render_template("signup.html", error=message, form_data={"name": name, "email": email})


# ============================================================
# LOGIN ROUTE (Normal submit)
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        success_message = None
        if request.args.get("registered") == "1":
            success_message = "Account created successfully. Please sign in to continue."
        return render_template("login.html", form_data={}, success_message=success_message)

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password")

    user = get_user_by_email(email)

    if user and check_password_hash(user["password_hash"], password):
        session['user_id'] = user["id"]
        session['user_name'] = user["name"]
        session['user_email'] = user["email"]
        return redirect(url_for('index'))

    return render_template("login.html",
                           error="Invalid email or password.",
                           form_data={"email": email},
                           success_message=None)


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('landing_page'))


# ============================================================
# PASSWORD RESET (unchanged)
# ============================================================

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email", "").strip().lower()

    user = get_user_by_email(email)
    if not user:
        return render_template("forgot_password.html",
                               info="If the email exists, a reset link has been sent.")

    token = serializer.dumps(email, salt=SECURITY_PASSWORD_SALT)
    sent, error = send_reset_email(email, token)
    if sent:
        return render_template("forgot_password.html",
                               success="Reset link sent! Check your inbox.")

    return render_template("forgot_password.html",
                           error=f"Unable to send reset email: {error}")


@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        email = serializer.loads(token, salt=SECURITY_PASSWORD_SALT, max_age=3600)
    except Exception:
        return "Invalid or expired token"

    if request.method == "POST":
        pw = request.form.get("password")
        pw2 = request.form.get("password2")

        if pw != pw2:
            return "Passwords do not match"

        update_user_password(email, pw)
        return "Password reset successful. <a href='/'>Login</a>"

    return render_template_string("""
        <form method="POST">
            <input name="password" type="password" placeholder="New password"><br>
            <input name="password2" type="password" placeholder="Confirm password"><br>
            <button>Reset</button>
        </form>
    """)


# ============================================================
# ML PREDICTION ROUTES
# ============================================================

@app.route("/predict_text", methods=["POST"])
@login_required
def predict_text():
    text = request.form.get("news_text")

    if not text or not text.strip():
        return render_template("result.html",
                               prediction="No text",
                               confidence=0,
                               original="",
                               cleaned="",
                               timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    cleaned = clean_text(text)
    pred = PIPELINE.predict([cleaned])[0]
    prob = round(max(PIPELINE.predict_proba([cleaned])[0]) * 100, 2)

    timestamp_value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_history(text, cleaned, pred, prob, timestamp_value, session.get("user_id"))

    return render_template("result.html",
                           prediction=pred,
                           confidence=prob,
                           original=text,
                           cleaned=cleaned,
                           timestamp=timestamp_value)


@app.route("/predict_image", methods=["POST"])
@login_required
def predict_image():
    img = request.files.get("news_image")

    if not img:
        return render_template("result.html",
                               prediction="No image",
                               confidence=0,
                               original="",
                               cleaned="",
                               timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    img.save("uploaded.png")
    text = pytesseract.image_to_string(Image.open("uploaded.png"))

    cleaned = clean_text(text)
    pred = PIPELINE.predict([cleaned])[0]
    prob = round(max(PIPELINE.predict_proba([cleaned])[0]) * 100, 2)

    timestamp_value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_history(text, cleaned, pred, prob, timestamp_value, session.get("user_id"))

    return render_template("result.html",
                           prediction=pred,
                           confidence=prob,
                           original=text,
                           cleaned=cleaned,
                           timestamp=timestamp_value)


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def landing_page():
    return render_template("landing.html", logged_in="user_id" in session, user_name=session.get("user_name"))


@app.route("/index")
@login_required
def index():
    return render_template("index.html", user_name=session.get("user_name"))


@app.route("/chart-data")
def chart_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT prediction, COUNT(*) FROM history GROUP BY prediction")
    rows = cursor.fetchall()

    fake = real = 0
    for label, count in rows:
        if label == "FAKE":
            fake = count
        else:
            real = count

    monthly = [0]*12
    cursor.execute("SELECT strftime('%m',timestamp), COUNT(*) FROM history GROUP BY 1")
    for month, count in cursor.fetchall():
        monthly[int(month)-1] = count

    conn.close()
    return jsonify({"fake": fake, "real": real, "monthly": monthly})


@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE user_id = ? ORDER BY id DESC", (session["user_id"],))
    rows = cursor.fetchall()
    conn.close()
    return render_template("history.html", records=rows, user_name=session.get("user_name"))


# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
