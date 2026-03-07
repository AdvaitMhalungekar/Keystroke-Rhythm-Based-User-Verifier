# Keystroke Rhythm Based User Verifier

## Overview

This project implements a **Keystroke Dynamics-based User Verification System** using machine learning. It captures users' typing patterns, extracts timing features, and uses a trained model to identify or verify users based on their unique keystroke rhythms.

## Features

- **Web-based Typing Test**: Users type a random paragraph in a browser, and their keystroke events are logged in real time.
- **Feature Extraction**: Extracts hold times, digraph (key-pair) timings, and statistical features from raw keystroke data.
- **Machine Learning Model**: Uses XGBoost for multi-class classification to identify users.
- **Real-time Prediction**: Predicts the user identity as soon as typing is finished.
- **Data Collection Utility**: Scripted logger for collecting keystroke data from multiple users.
- **Custom Paragraph Generation**: Uses Gemini API to generate random typing paragraphs.

## Technical Architecture

### 1. Data Collection

- **Script**: `data_creator/keystroke_logger.py`
- **How it works**: Uses `pynput` to record key press/release events, storing user, key, event type, and timestamp in CSV files (one per user).
- **Typing Prompt**: Opens [MonkeyType](https://monkeytype.com/) for natural typing.

### 2. Feature Engineering

- **Scripts**: `features_extraction.py`, `features_flattener.py`
- **Features**:
  - **Hold Time**: Duration each key is held down.
  - **Digraph Timing**: Time between consecutive key presses (down-down).
  - **Statistical Aggregates**: Mean, std, min, max of digraphs, top-N digraph timings.
- **Sessionization**: Splits raw logs into sessions for robust training.

### 3. Model Training

- **Notebook**: `main.ipynb`
- **Process**:
  - Loads all user CSVs, splits into sessions, extracts features.
  - Trains an XGBoost classifier (`xgb_keystroke_model.pkl`) on the processed data.
  - Saves label encoder, top digraphs, and feature names for inference.

### 4. Web Application

- **Script**: `app.py`
- **Framework**: Flask
- **Frontend**: `templates/index.html` (custom JS for real-time logging)
- **Endpoints**:
  - `/` : Renders typing test with random paragraph.
  - `/log` : Receives keystroke events (AJAX POST).
  - `/predict` : Runs feature extraction and model prediction, returns user and confidence.
- **Model Inference**: Loads trained model and encoders, predicts user from current session.

### 5. Real-time CLI Prediction

- **Script**: `realtime_prediction.py`
- **How it works**: Listens to keyboard events, processes buffer on ESC, prints predicted user.

### 6. Paragraph Generation

- **Script**: `data_creator/para_generator.py`
- **API**: Google Gemini (Generative AI) for random, natural-looking practice paragraphs.

## Data & Models

- **Raw Data**: `data/*.csv` (per-user keystroke logs)
- **Processed Data**: `data/processed_data.csv` (feature matrix for ML)
- **Models**: `models/` (XGBoost model, label encoder, digraphs, feature names)

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies (Flask, pandas, scikit-learn, xgboost, pynput, google-generativeai, etc.)
- For paragraph generation: Set `GEMINI_API_KEY_1` in `.env` file

## How to Run

### 1. Data Collection

- Edit `USER_ID` in `data_creator/keystroke_logger.py`
- Run the script: `python data_creator/keystroke_logger.py`
- Type in the browser window, press ESC to finish

### 2. Model Training

- Run all cells in `main.ipynb` to process data and train the model

### 3. Web App (User Verification)

- Start Flask app: `python app.py`
- Open browser at `http://localhost:5000`
- Type the displayed paragraph, press ESC to get prediction

### 4. Real-time CLI Prediction

- Run: `python realtime_prediction.py`
- Type in the MonkeyType window, press ESC to see predicted user

## File Structure

```
├── app.py                  # Flask web server
├── features_extraction.py  # Feature extraction logic
├── features_flattener.py   # Feature flattening & sessionization
├── main.ipynb              # Model training notebook
├── realtime_prediction.py  # CLI-based real-time prediction
├── requirements.txt        # Python dependencies
├── data/                   # Raw and processed keystroke data
├── data_creator/           # Data collection & paragraph generation scripts
├── models/                 # Trained model and encoders
├── templates/index.html    # Web UI
```

## Technical Details

- **Keystroke Buffering**: All keystroke events are buffered and processed as a session.
- **Feature Consistency**: Uses saved feature names and digraphs for consistent inference.
- **Thresholding**: Model can reject uncertain predictions (configurable threshold).
- **Extensible**: Add more users by collecting new data and retraining.

## Credits

- Developed by: [Your Team/Names]
- Typing prompt powered by Google Gemini API

## License

[Specify your license here]
