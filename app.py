from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["SECRET_KEY"] = "health_detector_secret"


class HealthDataValidator:
    """Validates user health input data."""

    @staticmethod
    def validate_age(age):
        try:
            age = float(age)
            if 1 <= age <= 150:
                return True, age
            return False, "Age must be between 1 and 150"
        except ValueError:
            return False, "Age must be a valid number"

    @staticmethod
    def validate_gender(gender):
        gender = gender.lower().strip()
        if gender in ["male", "m", "female", "f"]:
            return True, "M" if gender in ["male", "m"] else "F"
        return False, "Gender must be 'Male' or 'Female'"

    @staticmethod
    def validate_bmi(bmi):
        try:
            bmi = float(bmi)
            if 10 <= bmi <= 100:
                return True, bmi
            return False, "BMI must be between 10 and 100"
        except ValueError:
            return False, "BMI must be a valid number"

    @staticmethod
    def validate_bp(systolic, diastolic):
        try:
            systolic = float(systolic)
            diastolic = float(diastolic)
            if 50 <= systolic <= 250 and 30 <= diastolic <= 150:
                return True, (systolic, diastolic)
            return False, "BP values out of valid range"
        except ValueError:
            return False, "BP values must be valid numbers"

    @staticmethod
    def validate_sugar_level(sugar):
        try:
            sugar = float(sugar)
            if 40 <= sugar <= 600:
                return True, sugar
            return False, "Blood sugar must be between 40 and 600 mg/dL"
        except ValueError:
            return False, "Blood sugar must be a valid number"

    @staticmethod
    def validate_symptoms(symptoms):
        if isinstance(symptoms, str) and len(symptoms.strip()) > 0:
            return True, symptoms
        return False, "Please enter at least one symptom"


class DataPreprocessor:
    """Handles data preprocessing and normalization."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.gender_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X):
        self.gender_encoder.fit(X["Gender"].unique())
        numeric_features = ["Age", "BMI", "BP_Systolic", "BP_Diastolic", "BloodSugar"]
        self.scaler.fit(X[numeric_features])
        self.is_fitted = True

    def transform(self, data):
        data["Gender"] = self.gender_encoder.transform([data["Gender"]])[0]
        numeric_features = ["Age", "BMI", "BP_Systolic", "BP_Diastolic", "BloodSugar"]
        numeric_data = np.array(
            [
                [
                    data["Age"],
                    data["BMI"],
                    data["BP_Systolic"],
                    data["BP_Diastolic"],
                    data["BloodSugar"],
                ]
            ]
        )

        normalized = self.scaler.transform(numeric_data)[0]

        return {
            "Age": normalized[0],
            "Gender": data["Gender"],
            "BMI": normalized[1],
            "BP_Systolic": normalized[2],
            "BP_Diastolic": normalized[3],
            "BloodSugar": normalized[4],
        }


class HealthRiskDetector:
    """AI/ML model for health risk detection."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        self.risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        self.risk_colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}

    def train(self):
        """Train the model with synthetic data."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            "Age": np.random.uniform(20, 80, n_samples),
            "Gender": np.random.choice(["M", "F"], n_samples),
            "BMI": np.random.uniform(15, 40, n_samples),
            "BP_Systolic": np.random.uniform(100, 180, n_samples),
            "BP_Diastolic": np.random.uniform(60, 110, n_samples),
            "BloodSugar": np.random.uniform(70, 250, n_samples),
        }

        risks = []
        for i in range(n_samples):
            risk_score = 0

            if data["Age"][i] > 60:
                risk_score += 1
            if data["BMI"][i] > 30:
                risk_score += 1
            if data["BP_Systolic"][i] > 140 or data["BP_Diastolic"][i] > 90:
                risk_score += 1
            if data["BloodSugar"][i] > 180:
                risk_score += 1

            if risk_score >= 3:
                risks.append(2)
            elif risk_score >= 1:
                risks.append(1)
            else:
                risks.append(0)

        df = pd.DataFrame(data)
        self.preprocessor.fit(df)

        X = self._prepare_features(df)
        y = np.array(risks)

        self.model.fit(X, y)
        self.is_trained = True

    def _prepare_features(self, data):
        """Prepare features for model."""
        features = data.copy()
        features["Gender"] = self.preprocessor.gender_encoder.transform(
            features["Gender"]
        )

        numeric_features = ["Age", "BMI", "BP_Systolic", "BP_Diastolic", "BloodSugar"]
        features[numeric_features] = self.preprocessor.scaler.transform(
            features[numeric_features]
        )

        return features[
            ["Age", "Gender", "BMI", "BP_Systolic", "BP_Diastolic", "BloodSugar"]
        ].values

    def predict(self, health_data):
        """Make prediction on health data."""
        if not self.is_trained:
            self.train()

        df = pd.DataFrame([health_data])
        processed = self._prepare_features(df)

        prediction = self.model.predict(processed)[0]
        probabilities = self.model.predict_proba(processed)[0]

        risk_label = self.risk_levels[prediction]
        confidence = max(probabilities) * 100

        return risk_label, confidence, probabilities


class HealthAdviceEngine:
    """Provides health advice and recommendations."""

    @staticmethod
    def get_health_advice(health_data, risk_level):
        """Generate health advice based on metrics."""
        advice = []

        if health_data["Age"] > 60:
            advice.append(
                "You're in the senior age group. Regular health checkups are essential."
            )

        bmi = health_data["BMI"]
        if bmi < 18.5:
            advice.append(
                "Your BMI indicates underweight. Consider consulting a nutritionist."
            )
        elif 18.5 <= bmi < 25:
            advice.append(
                "Your BMI is in the healthy range. Keep up the good lifestyle!"
            )
        elif 25 <= bmi < 30:
            advice.append(
                "Your BMI is overweight. Increase physical activity and monitor diet."
            )
        else:
            advice.append(
                "Your BMI indicates obesity. Consult a healthcare provider for a weight management plan."
            )

        systolic = health_data["BP_Systolic"]
        if systolic < 120:
            advice.append("Your blood pressure is normal. Continue healthy habits.")
        elif systolic < 140:
            advice.append(
                "Your blood pressure is elevated. Reduce salt intake and increase exercise."
            )
        else:
            advice.append(
                "Your blood pressure is high. Consult your doctor immediately."
            )

        sugar = health_data["BloodSugar"]
        if sugar < 100:
            advice.append("Your fasting blood sugar is normal.")
        elif sugar < 126:
            advice.append(
                "Your blood sugar is slightly elevated. Monitor and reduce sugar intake."
            )
        else:
            advice.append("Your blood sugar is high. Consult an endocrinologist.")

        return advice

    @staticmethod
    def get_recommendations(risk_level):
        """Get recommendations based on risk level."""
        recommendations = {
            "Low Risk": [
                "Continue your current healthy lifestyle",
                "Exercise regularly (150 min/week)",
                "Maintain a balanced diet",
                "Get annual health checkups",
            ],
            "Medium Risk": [
                "Schedule a doctor visit to discuss preventive measures",
                "Increase physical activity to 200 min/week",
                "Reduce processed food and sugar intake",
                "Monitor your health metrics regularly",
                "Manage stress through meditation or yoga",
            ],
            "High Risk": [
                "URGENT: Visit your doctor immediately",
                "Get a comprehensive health screening",
                "Follow prescribed medications strictly",
                "Implement a personalized fitness plan",
                "Avoid smoking and excessive alcohol",
                "Consider consulting multiple specialists",
            ],
        }

        return recommendations.get(risk_level, [])


# Initialize detector globally
detector = HealthRiskDetector()
detector.train()


@app.route("/")
def index():
    """Render the main form page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint for health prediction."""
    try:
        data = request.json

        # Validate inputs
        validator = HealthDataValidator()

        valid, age = validator.validate_age(data.get("age"))
        if not valid:
            return jsonify({"error": age}), 400

        valid, gender = validator.validate_gender(data.get("gender"))
        if not valid:
            return jsonify({"error": gender}), 400

        valid, bmi = validator.validate_bmi(data.get("bmi"))
        if not valid:
            return jsonify({"error": bmi}), 400

        valid, (bp_sys, bp_dia) = validator.validate_bp(
            data.get("bp_systolic"), data.get("bp_diastolic")
        )
        if not valid:
            return jsonify({"error": bp_sys}), 400

        valid, sugar = validator.validate_sugar_level(data.get("blood_sugar"))
        if not valid:
            return jsonify({"error": sugar}), 400

        valid, symptoms = validator.validate_symptoms(data.get("symptoms"))
        if not valid:
            return jsonify({"error": symptoms}), 400

        # Prepare health data
        health_data = {
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "BP_Systolic": bp_sys,
            "BP_Diastolic": bp_dia,
            "BloodSugar": sugar,
            "Symptoms": symptoms,
        }

        # Make prediction
        risk_level, confidence, probabilities = detector.predict(health_data)

        # Get advice
        advice_engine = HealthAdviceEngine()
        advice = advice_engine.get_health_advice(health_data, risk_level)
        recommendations = advice_engine.get_recommendations(risk_level)

        # Prepare response
        return jsonify(
            {
                "success": True,
                "health_data": health_data,
                "risk_level": risk_level,
                "confidence": round(confidence, 1),
                "risk_color": detector.risk_colors.get(
                    list(detector.risk_levels.values()).index(risk_level), "#gray"
                ),
                "advice": advice,
                "recommendations": recommendations,
                "probabilities": {
                    "low": round(probabilities[0] * 100, 1),
                    "medium": round(probabilities[1] * 100, 1),
                    "high": round(probabilities[2] * 100, 1),
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
