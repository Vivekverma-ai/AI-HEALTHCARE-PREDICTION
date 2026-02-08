#!/usr/bin/env python3
"""Test script for Health Disease Detector"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from practice import HealthRiskDetector, HealthAdviceEngine, display_results


def test_detector():
    """Test the health detector with sample data."""

    print("\n🧪 Testing Health Disease Detector...")
    print("=" * 60)

    # Sample test cases
    test_cases = [
        {
            "name": "Low Risk Case",
            "data": {
                "Age": 30,
                "Gender": "M",
                "BMI": 22.5,
                "BP_Systolic": 120,
                "BP_Diastolic": 80,
                "BloodSugar": 95,
                "Symptoms": "None",
            },
        },
        {
            "name": "Medium Risk Case",
            "data": {
                "Age": 55,
                "Gender": "F",
                "BMI": 28.5,
                "BP_Systolic": 135,
                "BP_Diastolic": 85,
                "BloodSugar": 120,
                "Symptoms": "Fatigue, occasional headache",
            },
        },
        {
            "name": "High Risk Case",
            "data": {
                "Age": 70,
                "Gender": "M",
                "BMI": 35.0,
                "BP_Systolic": 160,
                "BP_Diastolic": 95,
                "BloodSugar": 200,
                "Symptoms": "Chest pain, shortness of breath, fatigue",
            },
        },
    ]

    # Initialize detector
    detector = HealthRiskDetector()
    print("\n⏳ Training model with synthetic data...")
    detector.train()
    print("✓ Model trained successfully\n")

    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")

        health_data = test_case["data"]

        # Predict
        print("⏳ Making prediction...")
        risk_level, confidence, probabilities = detector.predict(health_data)

        # Get advice
        advice_engine = HealthAdviceEngine()
        advice = advice_engine.get_health_advice(health_data, risk_level)
        recommendations = advice_engine.get_recommendations(risk_level)

        # Display results
        display_results(health_data, risk_level, confidence, advice, recommendations)

    print("\n✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_detector()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
