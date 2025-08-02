from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None

def load_uci_heart_disease_data():
    """
    Load UCI Heart Disease Dataset (Cleveland)
    Source: https://archive.ics.uci.edu/ml/datasets/heart+disease
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        'age', 'gender', 'chest_pain', 'blood_pressure', 'cholesterol',
        'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina',
        'oldpeak', 'slope', 'vessels', 'thalassemia', 'target'
    ]
    df = pd.read_csv(url, names=columns, na_values='?')
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target: 0 = no disease, 1 = disease (originally 0-4)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    print(f"Loaded UCI dataset with {len(df)} samples")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df

def train_model():
    """
    Train the heart disease prediction model with hyperparameter tuning
    """
    global model, scaler
    
    print("Loading UCI heart disease dataset...")
    df = load_uci_heart_disease_data()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define hyperparameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    model = grid_search.best_estimator_
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save the model and scaler
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")
    
    return model, scaler

def load_model():
    """
    Load the trained model and scaler
    """
    global model, scaler
    
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model and scaler loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        model, scaler = train_model()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_heart_disease(input_data):
    """
    Predict heart disease based on input features
    """
    global model, scaler
    
    if model is None or scaler is None:
        raise ValueError("Model not loaded")
    
    # Create feature array
    features = np.array([[
        input_data['age'],
        input_data['gender'],
        input_data['chestPain'],
        input_data['bloodPressure'],
        input_data['cholesterol'],
        input_data['fastingBloodSugar'],
        input_data['restingECG'],
        input_data['maxHeartRate'],
        input_data['exerciseAngina'],
        input_data['oldpeak'],
        input_data['slope'],
        input_data['vessels'],
        input_data['thalassemia']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    # Calculate confidence (probability of predicted class)
    confidence = prediction_proba[prediction] * 100
    
    # Calculate risk score (probability of disease * 100)
    risk_score = prediction_proba[1] * 100
    
    # Generate recommendations
    recommendations = generate_recommendations(input_data, prediction, risk_score)
    
    return {
        'prediction': int(prediction),
        'confidence': round(confidence, 1),
        'risk_score': round(risk_score, 1),
        'recommendations': recommendations
    }

def generate_recommendations(input_data, prediction, risk_score):
    """
    Generate personalized health recommendations
    """
    recommendations = []
    
    if prediction == 1:  # High risk
        recommendations.append("⚠️ IMMEDIATE ACTION REQUIRED:")
        recommendations.append("• Consult a cardiologist as soon as possible")
        recommendations.append("• Consider getting an ECG and stress test")
        recommendations.append("• Monitor blood pressure daily")
        
        if input_data['chestPain'] >= 2:
            recommendations.append("• Avoid strenuous activities until medical clearance")
        
        if input_data['cholesterol'] > 240:
            recommendations.append("• Follow a low-cholesterol diet immediately")
        
        if input_data['bloodPressure'] > 140:
            recommendations.append("• Reduce sodium intake and consider medication")
            
    else:  # Low risk
        recommendations.append("✅ GREAT NEWS: Low risk detected!")
        recommendations.append("• Maintain your current healthy lifestyle")
        recommendations.append("• Continue regular exercise routine")
        recommendations.append("• Schedule annual health check-ups")
    
    # General recommendations based on risk factors
    if input_data['age'] > 50:
        recommendations.append("• Consider more frequent cardiac screenings due to age")
    
    if input_data['gender'] == 1:  # Male
        recommendations.append("• Men have higher cardiac risk - stay vigilant")
    
    if input_data['cholesterol'] > 200:
        recommendations.append("• Include more fiber and omega-3 in your diet")
    
    if input_data['exerciseAngina'] == 1:
        recommendations.append("• Discuss exercise-induced chest pain with your doctor")
    
    if input_data['fastingBloodSugar'] == 1:
        recommendations.append("• Monitor and manage blood sugar levels")
    
    # Lifestyle recommendations
    recommendations.extend([
        "",
        "📋 GENERAL HEALTH TIPS:",
        "• Exercise 150 minutes per week (moderate intensity)",
        "• Eat a Mediterranean-style diet",
        "• Maintain healthy weight (BMI 18.5-24.9)",
        "• Don't smoke and limit alcohol",
        "• Manage stress through meditation or yoga",
        "• Get 7-9 hours of quality sleep nightly"
    ])
    
    return "\n".join(recommendations)

# API Routes
@app.route('/', methods=['GET'])
def home():
    """
    Health check endpoint
    """
    return jsonify({
        'message': 'Heart Disease Prediction API',
        'status': 'running',
        'model_loaded': model is not None and scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'age', 'gender', 'chestPain', 'bloodPressure', 'cholesterol',
            'fastingBloodSugar', 'restingECG', 'maxHeartRate', 'exerciseAngina',
            'oldpeak', 'slope', 'vessels', 'thalassemia'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Validate input ranges (example ranges based on UCI dataset)
        errors = []
        if not (20 <= data['age'] <= 80):
            errors.append("Age must be between 20 and 80")
        if data['gender'] not in [0, 1]:
            errors.append("Gender must be 0 (female) or 1 (male)")
        if data['chestPain'] not in [1, 2, 3, 4]:
            errors.append("Chest pain type must be 1-4")
        if not (90 <= data['bloodPressure'] <= 200):
            errors.append("Blood pressure must be between 90 and 200")
        if not (120 <= data['cholesterol'] <= 400):
            errors.append("Cholesterol must be between 120 and 400")
        if data['fastingBloodSugar'] not in [0, 1]:
            errors.append("Fasting blood sugar must be 0 or 1")
        if data['restingECG'] not in [0, 1, 2]:
            errors.append("Resting ECG must be 0-2")
        if not (80 <= data['maxHeartRate'] <= 220):
            errors.append("Max heart rate must be between 80 and 220")
        if data['exerciseAngina'] not in [0, 1]:
            errors.append("Exercise angina must be 0 or 1")
        if not (0 <= data['oldpeak'] <= 6.2):
            errors.append("Oldpeak must be between 0 and 6.2")
        if data['slope'] not in [1, 2, 3]:
            errors.append("Slope must be 1-3")
        if data['vessels'] not in [0, 1, 2, 3]:
            errors.append("Vessels must be 0-3")
        if data['thalassemia'] not in [3, 6, 7]:
            errors.append("Thalassemia must be 3, 6, or 7")
        
        if errors:
            return jsonify({'error': 'Validation errors', 'details': errors}), 400
        
        # Make prediction
        result = predict_heart_disease(data)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'risk_score': result['risk_score'],
            'recommendations': result['recommendations'],
            'message': 'High Risk - Consult a doctor immediately' if result['prediction'] == 1 
                      else 'Low Risk - Keep up the healthy lifestyle!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get model information
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'features': [
            'age', 'gender', 'chest_pain', 'blood_pressure', 'cholesterol',
            'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina',
            'oldpeak', 'slope', 'vessels', 'thalassemia'
        ],
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'trained': True
    })

if __name__ == '__main__':
    # Initialize model
    print("Starting Heart Disease Prediction API...")
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)