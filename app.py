from flask import Flask, render_template, request, jsonify
import random, time, joblib, pandas as pd
from features_flattener import flatten_features
from data_creator.para_generator import generate_text
app = Flask(__name__)

model = joblib.load("models/xgb_keystroke_model.pkl")
le = joblib.load("models/label_encoder.pkl")
top_digraphs = joblib.load("models/top_digraphs.pkl")
feature_names = joblib.load("models/feature_names.pkl")

keystroke_buffer = [] 

@app.route("/")
def home():
    try:
        text = generate_text()
    except:
        text = """ability able about above across act action active actual add address admit adult
    advance advice affect after again against age agency agent agree ahead air
    all allow almost alone along already also although always among amount analysis
    and animal answer any anyone anything appear apply approach area argue arm
    around arrive art article artist as ask assume at attack attention attorney
    audience author available avoid away baby back bad bag ball bank bar base
    basic basis be beat beautiful because become bed before begin behavior behind
    believe benefit best better between beyond big bill billion bit black blood
    blue board body book born both box build building business but buy by"""
    return render_template("index.html", text=text)

@app.route("/log", methods=["POST"])
def log_keystroke():
    global keystroke_buffer
    data = request.json
    keystroke_buffer.append([
        "unknown", data["key"], data["event"], data["time"]
    ])
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    global keystroke_buffer
    if not keystroke_buffer:
        return jsonify({"user": "No data"})

    df = pd.DataFrame(keystroke_buffer, columns=["user_id", "key", "event_type", "timestamp"])
    feats = flatten_features(df, label=None, top_digraphs=top_digraphs)
    df_feat = pd.DataFrame([feats]).reindex(columns=feature_names, fill_value=0)

    X = df_feat.astype(float).to_numpy()

    # ✅ Get probabilities instead of just labels
    probs = model.predict_proba(X)[0]
    max_prob = probs.max()
    pred_idx = probs.argmax()

    # ✅ Threshold-based rejection
    threshold = 0.6  # you can tune this based on validation
    print("\n\n\n\nProbability: ")
    print(max_prob) 
    if max_prob < threshold:
        user = "unknown"
    else:
        user = le.inverse_transform([pred_idx])[0]

    keystroke_buffer = []

    return jsonify({"user": user, "confidence": float(max_prob)})

if __name__ == "__main__":
    app.run(debug=True)
