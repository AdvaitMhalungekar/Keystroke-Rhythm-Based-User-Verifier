from pynput import keyboard
import pandas as pd
import time
import joblib
from features_flattener import flatten_features
import webbrowser

model = joblib.load("models/xgb_keystroke_model.pkl")
le = joblib.load("models/label_encoder.pkl")
top_digraphs = joblib.load("models/top_digraphs.pkl")
feature_names = joblib.load("models/feature_names.pkl")

fieldnames = ["user_id", "key", "event_type", "timestamp"]
buffer = []

def process_all(buffer):
    """Convert full key stream into feature vector & predict user."""
    
    df = pd.DataFrame(buffer, columns=fieldnames)

    feats = flatten_features(df, label=None, top_digraphs=top_digraphs)

    df_feat = pd.DataFrame([feats])

    df_feat = df_feat.reindex(columns=feature_names, fill_value=0)

    X = df_feat.astype(float).to_numpy()

    pred = model.predict(X)
    user = le.inverse_transform(pred)[0]

    print(f"[FINAL PREDICTION] User: {user}")
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
        print("\n[INFO] ESC pressed → Running prediction...")
        process_all(buffer)
        return False

url = "https://monkeytype.com/"
webbrowser.open(url)

print("[INFO] Opening MonkeyType for typing...")
time.sleep(3)
print("[INFO] Keystroke Verification Active. Type normally.")
print("[INFO] Press ESC to finish & identify user.\n")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
