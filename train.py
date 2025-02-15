import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load scaler and model
DATA_DIR = r"C:\Users\XPS\OneDrive\Desktop\flask\data"
scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
kan_model = load_model(os.path.join(DATA_DIR, 'kan_model.h5'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('features')
        if data is None:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        data = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data)
        prediction = kan_model.predict(data_scaled)
        
        return jsonify({'fraud_probability': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
