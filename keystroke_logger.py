from pynput import keyboard
import csv
import time
import os
import webbrowser

USER_ID = "advait"

filename = f"{USER_ID}_keystrokes.csv"
fieldnames = ["user_id", "key", "event_type", "timestamp"]

if not os.path.exists(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def on_press(key):
    try:
        key_str = key.char
    except AttributeError:
        key_str = str(key)

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "user_id": USER_ID,
            "key": key_str,
            "event_type": "down",
            "timestamp": time.time()
        })

def on_release(key):
    try:
        key_str = key.char
    except AttributeError:
        key_str = str(key)

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "user_id": USER_ID,
            "key": key_str,
            "event_type": "up",
            "timestamp": time.time()
        })

    if key == keyboard.Key.esc:
        print("[INFO] ESC pressed. Stopping logger.")
        return False  


url = "https://monkeytype.com/"


webbrowser.open(url)

print("[INFO] Opening website...")
time.sleep(3)

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print(f"[INFO] Logging started for {USER_ID}... Press ESC to stop.")
    listener.join()
