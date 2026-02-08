import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class HealthDataValidator:
    """Validates user health input data."""

    @staticmethod
    def validate_age(age):
        """Validate age input."""
        try:
            age = float(age)
            if 1 <= age <= 150:
                return True, age
            return False, "Age must be between 1 and 150"
        except ValueError:
            return False, "Age must be a valid number"

    @staticmethod
    def validate_gender(gender):
        """Validate gender input."""
        gender = gender.lower().strip()
        if gender in ["male", "m", "female", "f"]:
            return True, "M" if gender in ["male", "m"] else "F"
        return False, "Gender must be 'Male' or 'Female'"

    @staticmethod
    def validate_bmi(bmi):
        """Validate BMI input."""
        try:
            bmi = float(bmi)
            if 10 <= bmi <= 100:
                return True, bmi
            return False, "BMI must be between 10 and 100"
        except ValueError:
            return False, "BMI must be a valid number"

    @staticmethod
    def validate_bp(systolic, diastolic):
        """Validate blood pressure input."""
        try:
            systolic = float(systolic)
            diastolic = float(diastolic)
            if 50 <= systolic <= 250 and 30 <= diastolic <= 150:
                return True, (systolic, diastolic)
            return (
                False,
                "BP values out of valid range. Systolic: 50-250, Diastolic: 30-150",
            )
        except ValueError:
            return False, "BP values must be valid numbers"

    @staticmethod
    def validate_sugar_level(sugar):
        """Validate blood sugar level input."""
        try:
            sugar = float(sugar)
            if 40 <= sugar <= 600:
                return True, sugar
            return False, "Blood sugar must be between 40 and 600 mg/dL"
        except ValueError:
            return False, "Blood sugar must be a valid number"

    @staticmethod
    def validate_symptoms(symptoms):
        """Validate symptoms input."""
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
        """Fit preprocessor on training data."""
        self.gender_encoder.fit(X["Gender"].unique())
        numeric_features = ["Age", "BMI", "BP_Systolic", "BP_Diastolic", "BloodSugar"]
        self.scaler.fit(X[numeric_features])
        self.is_fitted = True

    def transform(self, data):
        """Transform input data."""
        # Encode gender
        data["Gender"] = self.gender_encoder.transform([data["Gender"]])[0]

        # Normalize numeric features
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

    def train(self):
        """Train the model with synthetic data."""
        # Create synthetic training data
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

        # Create risk labels based on health metrics
        risks = []
        for i in range(n_samples):
            risk_score = 0

            # Age factor
            if data["Age"][i] > 60:
                risk_score += 1

            # BMI factor
            if data["BMI"][i] > 30:
                risk_score += 1

            # BP factor
            if data["BP_Systolic"][i] > 140 or data["BP_Diastolic"][i] > 90:
                risk_score += 1

            # Blood sugar factor
            if data["BloodSugar"][i] > 180:
                risk_score += 1

            if risk_score >= 3:
                risks.append(2)  # High Risk
            elif risk_score >= 1:
                risks.append(1)  # Medium Risk
            else:
                risks.append(0)  # Low Risk

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
        """
        Make prediction on health data.

        Args:
            health_data: Dict with keys - Age, Gender, BMI, BP_Systolic, BP_Diastolic, BloodSugar

        Returns:
            Tuple of (risk_level, probability)
        """
        if not self.is_trained:
            self.train()

        # Create DataFrame for preprocessing
        df = pd.DataFrame([health_data])

        # Preprocess
        processed = self._prepare_features(df)

        # Predict
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

        # Age-related advice
        if health_data["Age"] > 60:
            advice.append(
                "• You're in the senior age group. Regular health checkups are essential."
            )

        # BMI-related advice
        bmi = health_data["BMI"]
        if bmi < 18.5:
            advice.append(
                "• Your BMI indicates underweight. Consider consulting a nutritionist."
            )
        elif 18.5 <= bmi < 25:
            advice.append(
                "• Your BMI is in the healthy range. Keep up the good lifestyle!"
            )
        elif 25 <= bmi < 30:
            advice.append(
                "• Your BMI is overweight. Increase physical activity and monitor diet."
            )
        else:
            advice.append(
                "• Your BMI indicates obesity. Consult a healthcare provider for a weight management plan."
            )

        # Blood Pressure advice
        systolic = health_data["BP_Systolic"]
        if systolic < 120:
            advice.append("• Your blood pressure is normal. Continue healthy habits.")
        elif systolic < 140:
            advice.append(
                "• Your blood pressure is elevated. Reduce salt intake and increase exercise."
            )
        else:
            advice.append(
                "• Your blood pressure is high. Consult your doctor immediately."
            )

        # Blood Sugar advice
        sugar = health_data["BloodSugar"]
        if sugar < 100:
            advice.append("• Your fasting blood sugar is normal.")
        elif sugar < 126:
            advice.append(
                "• Your blood sugar is slightly elevated. Monitor and reduce sugar intake."
            )
        else:
            advice.append("• Your blood sugar is high. Consult an endocrinologist.")

        return advice

    @staticmethod
    def get_recommendations(risk_level):
        """Get recommendations based on risk level."""
        recommendations = {
            "Low Risk": [
                "✓ Continue your current healthy lifestyle",
                "✓ Exercise regularly (150 min/week)",
                "✓ Maintain a balanced diet",
                "✓ Get annual health checkups",
            ],
            "Medium Risk": [
                "⚠ Schedule a doctor visit to discuss preventive measures",
                "⚠ Increase physical activity to 200 min/week",
                "⚠ Reduce processed food and sugar intake",
                "⚠ Monitor your health metrics regularly",
                "⚠ Manage stress through meditation or yoga",
            ],
            "High Risk": [
                "🔴 URGENT: Visit your doctor immediately",
                "🔴 Get a comprehensive health screening",
                "🔴 Follow prescribed medications strictly",
                "🔴 Implement a personalized fitness plan",
                "🔴 Avoid smoking and excessive alcohol",
                "🔴 Consider consulting multiple specialists",
            ],
        }

        return recommendations.get(risk_level, [])


def get_user_input():
    """Get and validate user health data from input."""
    print("\n" + "=" * 60)
    print("       AI HEALTH DISEASE DETECTOR")
    print("=" * 60)
    print("\nPlease enter your health details:")
    print("-" * 60)

    validator = HealthDataValidator()

    # Age
    while True:
        try:
            age = input("Age: ").strip()
            valid, result = validator.validate_age(age)
            if valid:
                age = result
                break
            print(f"❌ {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

    # Gender
    while True:
        try:
            gender = input("Gender (Male/Female): ").strip()
            valid, result = validator.validate_gender(gender)
            if valid:
                gender = result
                break
            print(f"❌ {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

    # BMI
    while True:
        try:
            bmi = input("BMI (Body Mass Index): ").strip()
            valid, result = validator.validate_bmi(bmi)
            if valid:
                bmi = result
                break
            print(f"❌ {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

    # Blood Pressure
    while True:
        try:
            bp_input = input(
                "Blood Pressure (Systolic/Diastolic, e.g., 120/80): "
            ).strip()
            parts = bp_input.split("/")
            if len(parts) != 2:
                print("❌ Please enter BP in format: Systolic/Diastolic")
                continue
            valid, result = validator.validate_bp(parts[0], parts[1])
            if valid:
                bp_systolic, bp_diastolic = result
                break
            print(f"❌ {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

    # Blood Sugar
    while True:
        try:
            sugar = input("Blood Sugar Level (mg/dL): ").strip()
            valid, result = validator.validate_sugar_level(sugar)
            if valid:
                sugar = result
                break
            print(f"❌ {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

    # Symptoms
    while True:
        try:
            symptoms = input(
                "Symptoms (comma-separated, e.g., fatigue, headache): "
            ).strip()
            valid, result = validator.validate_symptoms(symptoms)
            if valid:
                symptoms = result
                break
            print(f"❌ {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

    return {
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "BP_Systolic": bp_systolic,
        "BP_Diastolic": bp_diastolic,
        "BloodSugar": sugar,
        "Symptoms": symptoms,
    }


def display_results(health_data, risk_level, confidence, advice, recommendations):
    """Display prediction results and recommendations."""
    print("\n" + "=" * 60)
    print("       HEALTH ASSESSMENT RESULTS")
    print("=" * 60)

    # Risk Level
    if risk_level == "Low Risk":
        risk_symbol = "✓"
        color = "GREEN"
    elif risk_level == "Medium Risk":
        risk_symbol = "⚠"
        color = "YELLOW"
    else:
        risk_symbol = "🔴"
        color = "RED"

    print(f"\n{risk_symbol} RISK LEVEL: {risk_level}")
    print(f"   Confidence: {confidence:.1f}%")

    # Health Summary
    print("\n" + "-" * 60)
    print("Health Metrics Summary:")
    print("-" * 60)
    print(f"  Age:           {health_data['Age']:.0f} years")
    print(f"  Gender:        {health_data['Gender']}")
    print(f"  BMI:           {health_data['BMI']:.1f}")
    print(
        f"  Blood Pressure: {health_data['BP_Systolic']:.0f}/{health_data['BP_Diastolic']:.0f} mmHg"
    )
    print(f"  Blood Sugar:   {health_data['BloodSugar']:.0f} mg/dL")
    print(f"  Symptoms:      {health_data['Symptoms']}")

    # Health Advice
    print("\n" + "-" * 60)
    print("Health Advice:")
    print("-" * 60)
    for advice_item in advice:
        print(advice_item)

    # Recommendations
    print("\n" + "-" * 60)
    print("Recommendations:")
    print("-" * 60)
    for rec in recommendations:
        print(rec)

    print("\n" + "=" * 60)
    print("⚕️  DISCLAIMER: This is an AI-based assessment tool.")
    print("Please consult with a healthcare professional for")
    print("accurate diagnosis and treatment.")
    print("=" * 60 + "\n")


def main():
    """Main application flow."""
    try:
        # Step 1: User Opens App
        print("\n🏥 Welcome to AI Health Disease Detector")
        print("Powered by Machine Learning")

        # Step 2: User Enters Health Details
        health_data = get_user_input()

        # Step 3: Data Validation (Done in get_user_input)
        print("\n✓ Data validation successful")

        # Step 4: Initialize Detector and Preprocess
        print("⏳ Processing data...")
        detector = HealthRiskDetector()
        detector.train()

        # Step 5: AI/ML Model Prediction
        print("⏳ Running AI model analysis...")
        risk_level, confidence, probabilities = detector.predict(health_data)

        # Step 6: Get Health Advice
        advice_engine = HealthAdviceEngine()
        advice = advice_engine.get_health_advice(health_data, risk_level)
        recommendations = advice_engine.get_recommendations(risk_level)

        # Step 7: Display Results
        display_results(health_data, risk_level, confidence, advice, recommendations)

    except KeyboardInterrupt:
        print("\n\nApplication closed by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again with valid inputs.")


if __name__ == "__main__":
    main()
