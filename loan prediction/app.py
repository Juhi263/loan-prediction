from flask import Flask, request, jsonify, render_template
import numpy as np
import json
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

app = Flask(__name__)

# Load the trained model
model = load_model('loan_approval_model.keras')

# Load the scaler configuration
with open('scaler.json', 'r') as f:
    scaler_data = json.load(f)
scaler = StandardScaler()
scaler.mean_ = np.array(scaler_data['mean'])
scaler.scale_ = np.array(scaler_data['scale'])

# Load the label encoders
label_encoders = {}
for feature in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    encoder_file = f'{feature}_encoder.json'
    if os.path.exists(encoder_file):
        with open(encoder_file, 'r') as f:
            encoder_data = json.load(f)
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoder_data['classes'])
            label_encoders[feature] = encoder
    else:
        raise FileNotFoundError(f"Label encoder file not found: {encoder_file}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Read JSON data from request
        print(data)  # Print data to debug

        # Convert categorical variables to numeric using label encoders
        features = [
            label_encoders['Gender'].transform([data['Gender']])[0],
            label_encoders['Married'].transform([data['Married']])[0],
            label_encoders['Dependents'].transform([data['Dependents']])[0],
            label_encoders['Education'].transform([data['Education']])[0],
            label_encoders['Self_Employed'].transform([data['Self_Employed']])[0],
            float(data['ApplicantIncome']),
            float(data['CoapplicantIncome']),
            float(data['LoanAmount']),
            float(data['Loan_Amount_Term']),
            float(data['Credit_History']),
            label_encoders['Property_Area'].transform([data['Property_Area']])[0]
        ]

        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict with model
        prediction_prob = model.predict(features_scaled)
        print(f"Prediction Probability: {prediction_prob}")
        
        # Make binary prediction
        prediction = (prediction_prob > 0.5).astype(int)[0][0]
        result = 'Approved' if prediction == 1 else 'Rejected'
        
        return jsonify({'result': result})
    except KeyError as e:
        app.logger.error(f"Missing key in data: {e}")
        return jsonify({'error': 'Missing data in request'}), 400
    except ValueError as e:
        app.logger.error(f"Value error: {e}")
        return jsonify({'error': 'Invalid value in data'}), 400
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
