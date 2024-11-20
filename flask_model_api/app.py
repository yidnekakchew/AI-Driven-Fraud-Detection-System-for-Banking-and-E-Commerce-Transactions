from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import struct
import socket
import logging
from flask.logging import default_handler

app = Flask(__name__)

# Load models
with open('models/random_forest_fraud_model.pkl', 'rb') as f:
    fraud_model = pickle.load(f)

# Set up logging
logger = logging.getLogger('fraud_app_logger')
logger.setLevel(logging.INFO)

# Define file handler and set formatter
file_handler = logging.FileHandler('fraud_app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to the logger
logger.addHandler(file_handler)

# Optionally, remove the default Flask logger handler
app.logger.removeHandler(default_handler)
app.logger.addHandler(file_handler)

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return None

@app.before_request
def log_request_info():
    """Log each incoming request."""
    logger.info(f'Incoming request: {request.method} {request.path} - Params: {request.form}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    try:
        # Extract features from the form data
        data = {
            'ip_address': ip_to_int(request.form['ip_address']),
            'purchase_value': float(request.form['purchase_value']),
            'transaction_frequency': float(request.form['transaction_frequency']),
            'time_diff': float(request.form['time_diff']),
            'hour_of_day': float(request.form['hour_of_day']),
            'day_of_week': float(request.form['day_of_week']),
            'country_encoded': float(request.form['country_encoded']),
            'browser_FireFox': int(request.form['browser_FireFox']),
            'browser_IE': int(request.form['browser_IE']),
            'browser_Opera': int(request.form['browser_Opera']),
            'browser_Safari': int(request.form['browser_Safari']),
            'sex_M': int(request.form['sex_M']),
            'source_Direct': int(request.form['source_Direct']),
            'source_SEO': int(request.form['source_SEO']),
            'age': float(request.form['age']),
            'purchase_time': float(request.form['purchase_time']),
            'signup_time': float(request.form['signup_time']),
        }

        # Convert to DataFrame
        fraud_features = pd.DataFrame([data])

        # Ensure the DataFrame has the same order of columns as the training set
        fraud_features = fraud_features[['signup_time', 'purchase_time', 'purchase_value', 
                                          'age', 'ip_address', 'hour_of_day', 
                                          'transaction_frequency', 'time_diff', 
                                          'day_of_week', 'country_encoded', 
                                          'browser_FireFox', 'browser_IE', 
                                          'browser_Opera', 'browser_Safari', 
                                          'sex_M', 'source_Direct', 'source_SEO']]

        # Make prediction
        fraud_prediction = fraud_model.predict(fraud_features)

        # Map prediction to user-friendly label
        fraud_prediction_label = 'Fraudulent' if fraud_prediction == 1 else 'Legitimate'

        # Log prediction result
        logger.info(f'Prediction: {fraud_prediction_label} for IP: {request.form["ip_address"]}')

        return render_template('index.html', fraud_prediction=fraud_prediction_label)
    except Exception as e:
        # Log the error
        logger.error(f'Error processing request: {str(e)}')
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
