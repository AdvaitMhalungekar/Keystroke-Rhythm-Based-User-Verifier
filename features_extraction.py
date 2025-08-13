import pandas as pd

def extract_features(file):
    df = pd.read_csv(file)
    df = df.sort_values('timestamp')

    hold_times = {}
    dd_times = []
    last_down_time = None
    last_key = None

    down_times = {}

    for index, row in df.iterrows():
        key = row['key']
        event = row['event_type']
        time = row['timestamp']

        if event == 'down':
            down_times[key] = time

            if last_down_time:
                dd = time - last_down_time
                dd_times.append((last_key, key, dd))

            last_down_time = time
            last_key = key

        elif event == 'up' and key in down_times:
            hold_time = time - down_times[key]
            hold_times[key] = hold_times.get(key, []) + [hold_time]

    return {
        "avg_hold_time": {k: sum(v)/len(v) for k, v in hold_times.items()},
        "dd_times": dd_times
    }

# Usage
features = extract_features("advait_keystrokes.csv")
print(features)
