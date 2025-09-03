from pynput import keyboard
import pandas as pd
import time
import joblib
from features_flattener import flatten_features
import webbrowser

# Load model + encoder + top digraphs
model = joblib.load("models/xgb_keystroke_model.pkl")
le = joblib.load("models/label_encoder.pkl")
top_digraphs = joblib.load("models/top_digraphs.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Buffer for all keystrokes
fieldnames = ["user_id", "key", "event_type", "timestamp"]
buffer = []

def process_all(buffer):
    """Process the entire keystroke buffer once and predict the user."""
    df = pd.DataFrame(buffer, columns=fieldnames)
    feats = flatten_features(df, label=None, top_digraphs=top_digraphs)
    df_feat = pd.DataFrame([feats]).fillna(0).infer_objects(copy=False)

    if "label" in df_feat.columns:
        df_feat = df_feat.drop("label", axis=1)

    # Reindex to match training order
    df_feat = df_feat.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(df_feat)
    user = le.inverse_transform(pred)[0]
    print(f"\n[FINAL PREDICTION] User identified as: {user}")
    return user

def on_press(key):
    try:
        key_str = key.char
    except AttributeError:
        key_str = str(key)

    buffer.append(["unknown", key_str, "down", time.time()])

def on_release(key):
    try:
        key_str = key.char
    except AttributeError:
        key_str = str(key)

    buffer.append(["unknown", key_str, "up", time.time()])

    if key == keyboard.Key.esc:
        print("[INFO] ESC pressed. Stopping and predicting user...")
        # Process everything when ESC is pressed
        process_all(buffer)
        return False

# --- Start ---
url = "https://monkeytype.com/"
webbrowser.open(url)

print("[INFO] Opening website...")
time.sleep(3)
print("[INFO] Real-time keystroke verification started... Press ESC to stop and get final prediction.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
