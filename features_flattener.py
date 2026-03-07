import pandas as pd
from features_extraction import extract_features, remove_outliers_zscore
from collections import Counter

import math
from collections import Counter

def compute_entropy(values, bins=10):
    """
    Compute Shannon entropy of typing timings.
    """
    
    if len(values) == 0:
        return 0

    # discretize values into bins
    min_v = min(values)
    max_v = max(values)

    if min_v == max_v:
        return 0

    step = (max_v - min_v) / bins
    buckets = []

    for v in values:
        idx = int((v - min_v) / step)
        if idx == bins:
            idx -= 1
        buckets.append(idx)

    counts = Counter(buckets)
    total = len(values)

    entropy = 0

    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)

    return entropy

def minmax_scale_dict(features):
    """
    Apply Min-Max scaling to numeric feature dictionary.
    """
    
    values = list(features.values())
    
    if len(values) == 0:
        return features

    mn = min(values)
    mx = max(values)

    if mx == mn:
        return {k:0 for k in features}

    scaled = {}

    for k,v in features.items():
        scaled[k] = (v - mn) / (mx - mn)

    return scaled

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
    
    # Remove abnormal typing delays
    dd_durations = remove_outliers_zscore(dd_durations)
    
    
    if dd_durations:
        flat["dd_entropy"] = compute_entropy(dd_durations)
        flat["dd_mean"] = sum(dd_durations)/len(dd_durations)
        flat["dd_std"] = pd.Series(dd_durations).std()
        flat["dd_min"] = min(dd_durations)
        flat["dd_max"] = max(dd_durations)
    else:
        flat.update({
            "dd_mean":0,
            "dd_std":0,
            "dd_min":0,
            "dd_max":0,
            "dd_entropy":0
        })
    
    # top-N digraphs
    if top_digraphs:
        digraph_map = {}
        for a,b,dd in features["dd_times"]:
            digraph_map.setdefault((a,b), []).append(dd)
        for dig in top_digraphs:
            vals = digraph_map.get(dig, [])
            flat[f"dd_{dig[0]}_{dig[1]}"] = sum(vals)/len(vals) if vals else 0
    
    label_value = label
    flat.pop("label", None)

    flat = minmax_scale_dict(flat)

    flat["label"] = label_value
    
    return flat
