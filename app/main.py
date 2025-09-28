from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.model_loaded = False
        
    def load_data_and_train_model(self):
        """Load data and train the model if not already trained"""
        try:
            # Load the dataset
            data_path = os.path.join('..', 'data', 'Telco-Customer-Churn.csv')
            df = pd.read_csv(data_path)
            
            # Data preprocessing (same as in notebook)
            df_processed = df.copy()
            
            # Convert TotalCharges to numeric
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
            
            # Remove customerID
            if 'customerID' in df_processed.columns:
                df_processed = df_processed.drop('customerID', axis=1)
            
            # Encode categorical variables
            categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            
            for column in categorical_columns:
                if column != 'Churn':
                    le = LabelEncoder()
                    df_processed[column] = le.fit_transform(df_processed[column])
                    self.label_encoders[column] = le
            
            # Encode target variable
            target_le = LabelEncoder()
            df_processed['Churn'] = target_le.fit_transform(df_processed['Churn'])
            
            # Split features and target
            X = df_processed.drop('Churn', axis=1)
            y = df_processed['Churn']
            
            self.feature_names = X.columns.tolist()
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Gradient Boosting model (our best performer)
            self.model = GradientBoostingClassifier(random_state=42)
            self.model.fit(X_scaled, y)
            
            self.model_loaded = True
            print("✅ Model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error loading/training model: {str(e)}")
            self.model_loaded = False
    
    def predict_churn(self, customer_data):
        """Predict churn probability for a customer"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_data])
            
            # Apply same preprocessing
            for column, encoder in self.label_encoders.items():
                if column in df.columns:
                    df[column] = encoder.transform(df[column])
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            return {
                "prediction": "High Risk" if prediction == 1 else "Low Risk",
                "probability": float(probability[1]),  # Probability of churning
                "confidence": float(max(probability))
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize predictor
predictor = ChurnPredictor()

@app.route('/')
def home():
    """Home page with project overview"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Prediction interface page"""
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    """Model performance dashboard"""
    # Model performance metrics (from our notebook results)
    model_results = {
        'Gradient Boosting': {'accuracy': 80.55, 'rank': 1},
        'Random Forest': {'accuracy': 78.28, 'rank': 2},
        'Logistic Regression': {'accuracy': 77.79, 'rank': 3},
        'Neural Network': {'accuracy': 74.66, 'rank': 4}
    }
    return render_template('dashboard.html', model_results=model_results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for churn prediction"""
    try:
        # Get JSON data from request
        data = request.json
        
        # Initialize model if not loaded
        if not predictor.model_loaded:
            predictor.load_data_and_train_model()
        
        # Make prediction
        result = predictor.predict_churn(data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model_info')
def model_info():
    """API endpoint for model information"""
    if not predictor.model_loaded:
        predictor.load_data_and_train_model()
    
    return jsonify({
        "model_type": "Gradient Boosting Classifier",
        "accuracy": "80.55%",
        "features_count": len(predictor.feature_names) if predictor.feature_names else 0,
        "status": "Ready" if predictor.model_loaded else "Loading"
    })

if __name__ == '__main__':
    print("🚀 Starting Customer Churn Prediction Web App...")
    print("📊 Loading model and data...")
    predictor.load_data_and_train_model()
    print("🌐 Starting Flask server...")
    
    # Get port from environment variable for Railway deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
