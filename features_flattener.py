import pandas as pd
from features_extraction import extract_features
from collections import Counter

def clean_feature_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[\[\]<]", "", regex=True)   # remove forbidden chars
        .str.replace("Key.", "Key_", regex=False) # replace dots in Key.*
        .str.replace(" ", "_", regex=False)       # replace spaces
    )
    return df

def split_sessions(file, session_size=100):
    """Split raw keystroke log into smaller session CSVs."""
    df = pd.read_csv(file, on_bad_lines="skip")
    sessions = []
    for i in range(0, len(df), session_size):
        chunk = df.iloc[i:i+session_size]
        if len(chunk) > 10:  # discard tiny sessions
            sessions.append(chunk)
    return sessions

def get_top_digraphs(files, N=20):
    """Get most common digraphs across all sessions of all files."""
    counter = Counter()
    for file in files:
        sessions = split_sessions(file)
        for sess in sessions:
            feats = extract_features(sess)
            counter.update([(a,b) for a,b,_ in feats["dd_times"]])
    return [dig for dig,_ in counter.most_common(N)]

def extract_features_from_df(df):
    """Wrapper to allow extract_features to work on dataframe chunks."""
    return extract_features(df)

def flatten_features(df, label, top_digraphs=None):
    features = extract_features_from_df(df)
    
    # hold times
    flat = {f"hold_{k}": v for k,v in features["avg_hold_time"].items()}
    
    # digraph aggregates
    dd_durations = [dd for _,_,dd in features["dd_times"]]
    if dd_durations:
        flat["dd_mean"] = sum(dd_durations)/len(dd_durations)
        flat["dd_std"] = pd.Series(dd_durations).std()
        flat["dd_min"] = min(dd_durations)
        flat["dd_max"] = max(dd_durations)
    else:
        flat.update({"dd_mean":0,"dd_std":0,"dd_min":0,"dd_max":0})
    
    # top-N digraphs
    if top_digraphs:
        digraph_map = {}
        for a,b,dd in features["dd_times"]:
            digraph_map.setdefault((a,b), []).append(dd)
        for dig in top_digraphs:
            vals = digraph_map.get(dig, [])
            flat[f"dd_{dig[0]}_{dig[1]}"] = sum(vals)/len(vals) if vals else 0
    
    flat["label"] = label
    return flat
