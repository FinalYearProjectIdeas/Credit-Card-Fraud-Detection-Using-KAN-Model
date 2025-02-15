import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from waitress import serve

app = Flask(__name__)

# Define paths
DATA_PATH = r"C:\Users\XPS\OneDrive\Desktop\flask\synthetic_creditcard.csv"
MODEL_PATH = r"C:\Users\XPS\OneDrive\Desktop\flask\data\kan_model.h5"
SCALER_PATH = r"C:\Users\XPS\OneDrive\Desktop\flask\data\scaler.pkl"

# Load model and scaler
try:
    kan_model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit(1)  # Exit if loading fails

# Load dataset
df = pd.read_csv(DATA_PATH)
if 'Class' not in df.columns:
    raise KeyError("Dataset missing 'Class' column.")

X = df.drop(columns=['Class'])  # Drop target column

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('features')  # Expecting "features" key in request body
        if data is None:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        data = np.array(data).reshape(1, -1)  # Reshape for model input
        data_scaled = scaler.transform(data)  # Scale input data
        prediction = kan_model.predict(data_scaled)  # Get prediction
        
        return jsonify({'fraud_probability': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
