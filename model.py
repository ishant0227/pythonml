import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sqlalchemy import create_engine
import joblib
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load MySQL engine
engine = create_engine("mysql+mysqlconnector://root:root@heartdb.crq4w4a2ajsu.ap-south-1.rds.amazonaws.com/heartdb")

# Load data
df = pd.read_sql("SELECT * FROM clustered_lifestyle_data", engine)

# Columns to exclude from model features
columns_to_exclude = ['risk_label', 'diet_preference', 'email', 'id', 'age', 'height', 'weight', 'name', 'profession', 'cluster']

# Initialize scaler
scaler = StandardScaler()

# Encode categorical features
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns = [col for col in categorical_columns if col not in columns_to_exclude]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target label if necessary
if df['risk_label'].dtype == 'object':
    le_risk = LabelEncoder()
    df['risk_label'] = le_risk.fit_transform(df['risk_label'])
    label_encoders['risk_label'] = le_risk

# Prepare feature matrix and target vector
X = df.drop(columns=columns_to_exclude, errors='ignore')
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

y = pd.to_numeric(df['risk_label'], errors='coerce').fillna(0).astype(np.int64)

# Scale features
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load or train Random Forest model
rf_model_path = 'rf_model.pkl'
if os.path.exists(rf_model_path):
    logging.info("Loading Random Forest model from disk...")
    rf_model = joblib.load(rf_model_path)
else:
    logging.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, rf_model_path)

# Load or train ANN model
ann_model_path = 'ann_model.h5'
if os.path.exists(ann_model_path):
    logging.info("Loading ANN model from disk...")
    ann_model = load_model(ann_model_path)
else:
    logging.info("Training ANN model...")
    ann_model = Sequential([
        Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann_model.fit(X_train, y_train, epochs=15, batch_size=4, verbose=1)
    ann_model.save(ann_model_path)

# Evaluate model accuracies
rf_preds = rf_model.predict(X_test)

# Convert predictions if needed
if isinstance(rf_preds[0], str) and 'risk_label' in label_encoders:
    rf_preds = label_encoders['risk_label'].transform(rf_preds)

rf_accuracy = accuracy_score(y_test, rf_preds)
_, ann_accuracy = ann_model.evaluate(X_test, y_test, verbose=0)

logging.info(f"Random Forest Accuracy: {rf_accuracy}")
logging.info(f"ANN Accuracy: {ann_accuracy}")

logging.info(f"Training data columns: {X.columns.tolist()}")


required_fields =  list(X.columns)

# Health check
@app.route("/")
def health():
    return "Heart Attack Risk API is running!", 200

# Prediction endpoint
# Define global variable at the top
prediction_result = None

@app.route("/predict", methods=["POST"])
def predict():
    global prediction_result  # Allow modifying the global variable

    try:
        data = request.get_json()
        logging.info(f"Incoming JSON keys: {list(data.keys())}")
        logging.info(f"Expected fields: {sorted(required_fields)}")
        logging.info(f"Received fields: {sorted(data.keys())}")

        # Validate required fields
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing field(s): {', '.join(missing)}"}), 400

        input_data = [data[feature] for feature in required_fields]

        # Encode categorical values
        for idx, feature in enumerate(required_fields):
            if feature in categorical_columns:
                le = label_encoders.get(feature)
                if le:
                    input_data[idx] = le.transform([input_data[idx]])[0]
                else:
                    logging.error(f"Label encoder not found for {feature}")
                    return jsonify({"error": f"Label encoder not found for {feature}"}), 400

        # Scale input
        try:
            scaled = scaler.transform([input_data])
        except Exception as e:
            logging.error(f"Error during scaling: {str(e)}")
            return jsonify({"error": "Error during scaling input data."}), 400

        # Make predictions
        try:
            rf_pred = rf_model.predict_proba(scaled)[0][1]
            ann_pred = ann_model.predict(np.array(scaled))[0][0]
            final_score = round(((rf_pred + ann_pred) / 2) * 100, 2)
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": "Error during prediction."}), 500

        # Store result globally
        prediction_result = {
            "riskScore": float(final_score),
            "randomForestScore": float(round(rf_pred * 100, 2)),
            "annScore": float(round(ann_pred * 100, 2))
        }

        logging.info(f"Prediction result: {prediction_result}")
        return jsonify(prediction_result)

    except KeyError as e:
        logging.error(f"Missing required data: {e}")
        return jsonify({"error": f"Missing required data: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500

@app.route("/get_risk_score", methods=["GET"])
def get_risk_score():
    if prediction_result:
        return jsonify(prediction_result)
    else:
        return jsonify({"message": "No prediction has been made yet."}), 404



# Run Flask server
if __name__ == "__main__":
    app.run(port=5000, debug=True)
