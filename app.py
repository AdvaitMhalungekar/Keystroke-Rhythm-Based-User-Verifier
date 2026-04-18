from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import random, time, joblib, pandas as pd
from features_flattener import flatten_features
from data_creator.para_generator import generate_text

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this in production


# --- Static user credentials ---
users = {
    "advait": "1234",
    "chetan": "1234",
    "gayatri": "1234",
    "purva": "1234",
    "siddhant": "1234"
}

model = joblib.load("models/xgb_keystroke_model.pkl")
le = joblib.load("models/label_encoder.pkl")
top_digraphs = joblib.load("models/top_digraphs.pkl")
feature_names = joblib.load("models/feature_names.pkl")

keystroke_buffer = []


# ====================
# LOGIN PAGE (Primary Auth)
# ====================
@app.route("/", methods=["GET"])
def login_page():
    if session.get("logged_in_user"):
        return redirect(url_for("verify_page"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    if username in users and users[username] == password:
        session["logged_in_user"] = username
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Invalid credentials"})

# ====================
# VERIFY PAGE (Keystroke 2FA)
# ====================
@app.route("/verify", methods=["GET"])
def verify_page():
    if not session.get("logged_in_user"):
        return redirect(url_for("login_page"))
    try:
        text = generate_text()
    except:
        text = """ability able about above across act action active actual add address admit adult\nadvance advice affect after again against age agency agent agree ahead air\nall allow almost alone along already also although always among amount analysis\nand animal answer any anyone anything appear apply approach area argue arm\naround arrive art article artist as ask assume at attack attention attorney\naudience author available avoid away baby back bad bag ball bank bar base\nbasic basis be beat beautiful because become bed before begin behavior behind\nbelieve benefit best better between beyond big bill billion bit black blood\nblue board body book born both box build building business but buy by"""
    return render_template("verify.html", text=text, username=session["logged_in_user"])

# ====================
# DASHBOARD PAGE
# ====================
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in_user"):
        return redirect(url_for("login_page"))
    return render_template("dashboard.html", username=session["logged_in_user"])


# ====================
# Keystroke Logging (Unchanged)
# ====================
@app.route("/log", methods=["POST"])
def log_keystroke():
    global keystroke_buffer
    data = request.json
    keystroke_buffer.append([
        "unknown", data["key"], data["event"], data["time"]
    ])
    return jsonify({"status": "ok"})


# ====================
# Predict (2FA logic)
# ====================
@app.route("/predict", methods=["POST"])
def predict():
    global keystroke_buffer
    if not keystroke_buffer:
        return jsonify({"result": "no_data"})

    df = pd.DataFrame(keystroke_buffer, columns=["user_id", "key", "event_type", "timestamp"])
    feats = flatten_features(df, label=None, top_digraphs=top_digraphs)
    df_feat = pd.DataFrame([feats]).reindex(columns=feature_names, fill_value=0)
    X = df_feat.astype(float).to_numpy()

    probs = model.predict_proba(X)[0]
    max_prob = probs.max()
    pred_idx = probs.argmax()
    threshold = 0.6
    if max_prob < threshold:
        predicted_user = "unknown"
    else:
        predicted_user = le.inverse_transform([pred_idx])[0]

    print(f"\n\n\n\n{predicted_user}")
    logged_in_user = session.get("logged_in_user")
    access_granted = bool(predicted_user == logged_in_user and max_prob >= threshold)

    keystroke_buffer = []

    return jsonify({
        "predicted_user": predicted_user,
        "confidence": float(max_prob),
        "access_granted": access_granted,
        "logged_in_user": str(logged_in_user) if logged_in_user is not None else None
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
