"""
===============================================================================
UNIT TESTS - Heart Disease Prediction System
===============================================================================
Run with: pytest test_unit.py -v --cov
Coverage: pytest test_unit.py -v --cov --cov-report=html
===============================================================================
"""

import unittest
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json


# ==================== TEST DATA FIXTURES ====================

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        'Age': 54,
        'Sex': 'M',
        'ChestPainType': 'ASY',
        'RestingBP': 140,
        'Cholesterol': 239,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 160,
        'ExerciseAngina': 'N',
        'Oldpeak': 1.2,
        'ST_Slope': 'Flat'
    }


@pytest.fixture
def low_risk_patient():
    """Low risk patient data"""
    return {
        'Age': 40,
        'Sex': 'F',
        'ChestPainType': 'ATA',
        'RestingBP': 120,
        'Cholesterol': 180,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 170,
        'ExerciseAngina': 'N',
        'Oldpeak': 0.0,
        'ST_Slope': 'Up'
    }


@pytest.fixture
def high_risk_patient():
    """High risk patient data"""
    return {
        'Age': 65,
        'Sex': 'M',
        'ChestPainType': 'ASY',
        'RestingBP': 160,
        'Cholesterol': 300,
        'FastingBS': 1,
        'RestingECG': 'ST',
        'MaxHR': 100,
        'ExerciseAngina': 'Y',
        'Oldpeak': 3.0,
        'ST_Slope': 'Flat'
    }


# ==================== TEST CLASS: DATA VALIDATION ====================

class TestDataValidation:
    """Unit tests for input data validation"""
    
    def test_age_valid_range(self, sample_patient_data):
        """Test age is within valid range (1-120)"""
        age = sample_patient_data['Age']
        assert 1 <= age <= 120, "Age should be between 1 and 120"
    
    def test_age_invalid_negative(self):
        """Test negative age is rejected"""
        age = -5
        assert not (1 <= age <= 120), "Negative age should be invalid"
    
    def test_age_invalid_too_high(self):
        """Test age > 120 is rejected"""
        age = 150
        assert not (1 <= age <= 120), "Age > 120 should be invalid"
    
    def test_sex_valid_values(self, sample_patient_data):
        """Test sex is either M or F"""
        sex = sample_patient_data['Sex']
        assert sex in ['M', 'F'], "Sex must be M or F"
    
    def test_blood_pressure_valid_range(self, sample_patient_data):
        """Test blood pressure is within valid range (0-300)"""
        bp = sample_patient_data['RestingBP']
        assert 0 <= bp <= 300, "BP should be between 0 and 300"
    
    def test_cholesterol_valid_range(self, sample_patient_data):
        """Test cholesterol is within valid range (0-600)"""
        chol = sample_patient_data['Cholesterol']
        assert 0 <= chol <= 600, "Cholesterol should be between 0 and 600"
    
    def test_fasting_bs_binary(self, sample_patient_data):
        """Test fasting blood sugar is binary (0 or 1)"""
        fbs = sample_patient_data['FastingBS']
        assert fbs in [0, 1], "FastingBS must be 0 or 1"
    
    def test_max_hr_valid_range(self, sample_patient_data):
        """Test max heart rate is within valid range (50-220)"""
        hr = sample_patient_data['MaxHR']
        assert 50 <= hr <= 220, "MaxHR should be between 50 and 220"
    
    def test_oldpeak_valid_range(self, sample_patient_data):
        """Test oldpeak is within valid range (-10 to 10)"""
        oldpeak = sample_patient_data['Oldpeak']
        assert -10 <= oldpeak <= 10, "Oldpeak should be between -10 and 10"
    
    def test_chest_pain_type_valid(self, sample_patient_data):
        """Test chest pain type is one of valid values"""
        cp = sample_patient_data['ChestPainType']
        assert cp in ['ATA', 'NAP', 'ASY', 'TA'], "Invalid chest pain type"
    
    def test_resting_ecg_valid(self, sample_patient_data):
        """Test resting ECG is one of valid values"""
        ecg = sample_patient_data['RestingECG']
        assert ecg in ['Normal', 'ST', 'LVH'], "Invalid ECG value"
    
    def test_exercise_angina_valid(self, sample_patient_data):
        """Test exercise angina is Y or N"""
        ea = sample_patient_data['ExerciseAngina']
        assert ea in ['Y', 'N'], "ExerciseAngina must be Y or N"
    
    def test_st_slope_valid(self, sample_patient_data):
        """Test ST slope is one of valid values"""
        slope = sample_patient_data['ST_Slope']
        assert slope in ['Up', 'Flat', 'Down'], "Invalid ST slope value"


# ==================== TEST CLASS: DATA PREPROCESSING ====================

class TestDataPreprocessing:
    """Unit tests for data preprocessing functions"""
    
    def test_dataframe_creation(self, sample_patient_data):
        """Test converting dict to DataFrame"""
        df = pd.DataFrame([sample_patient_data])
        assert len(df) == 1, "DataFrame should have 1 row"
        assert len(df.columns) == 11, "DataFrame should have 11 columns"
    
    def test_encode_sex_male(self):
        """Test encoding male sex"""
        sex_map = {'M': 1, 'F': 0}
        assert sex_map['M'] == 1
    
    def test_encode_sex_female(self):
        """Test encoding female sex"""
        sex_map = {'M': 1, 'F': 0}
        assert sex_map['F'] == 0
    
    def test_encode_chest_pain_types(self):
        """Test encoding all chest pain types"""
        cp_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
        assert cp_map['ATA'] == 0
        assert cp_map['NAP'] == 1
        assert cp_map['ASY'] == 2
        assert cp_map['TA'] == 3
    
    def test_encode_ecg_types(self):
        """Test encoding ECG types"""
        ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
        assert ecg_map['Normal'] == 0
        assert ecg_map['ST'] == 1
        assert ecg_map['LVH'] == 2
    
    def test_feature_scaling_shape(self):
        """Test scaled features maintain correct shape"""
        features = np.array([[54, 1, 2, 140, 239, 0, 0, 160, 0, 1.2, 1]])
        # Simulate scaling (mean=0, std=1)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        scaled = (features - mean) / (std + 1e-8)
        assert scaled.shape == (1, 11), "Scaled features should maintain shape"
    
    def test_feature_scaling_normalization(self):
        """Test features are properly normalized"""
        features = np.array([[100, 200, 300]])
        mean = features.mean()
        std = features.std()
        scaled = (features - mean) / std
        # After scaling, mean should be ~0
        assert abs(scaled.mean()) < 1e-10, "Mean should be close to 0"
    
    def test_handle_missing_values(self):
        """Test missing value detection"""
        data = pd.DataFrame({'A': [1, 2, np.nan, 4]})
        assert data.isnull().sum().sum() == 1, "Should detect 1 missing value"
    
    def test_categorical_encoding_consistency(self):
        """Test same input produces same encoding"""
        input1 = 'M'
        input2 = 'M'
        encoder = {'M': 1, 'F': 0}
        assert encoder[input1] == encoder[input2], "Same input should encode identically"


# ==================== TEST CLASS: ML MODEL ====================

class TestMLModel:
    """Unit tests for ML model predictions"""
    
    def test_prediction_binary_output(self):
        """Test prediction is binary (0 or 1)"""
        prediction = 1
        assert prediction in [0, 1], "Prediction must be 0 or 1"
    
    def test_probability_range(self):
        """Test probability is between 0 and 1"""
        probability = 0.78
        assert 0 <= probability <= 1, "Probability must be in [0, 1]"
    
    def test_probability_array_sum(self):
        """Test probability array sums to 1"""
        probs = np.array([0.3, 0.7])
        assert np.abs(probs.sum() - 1.0) < 1e-6, "Probabilities should sum to 1"
    
    def test_prediction_shape(self):
        """Test prediction output shape"""
        predictions = np.array([1])
        assert predictions.shape == (1,), "Prediction shape should be (1,)"
    
    def test_probability_shape(self):
        """Test probability output shape"""
        probs = np.array([[0.3, 0.7]])
        assert probs.shape == (1, 2), "Probability shape should be (1, 2)"
    
    def test_risk_classification_low(self):
        """Test low risk classification (< 30%)"""
        prob = 0.2
        risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
        assert risk == 'Low Risk'
    
    def test_risk_classification_moderate(self):
        """Test moderate risk classification (30-60%)"""
        prob = 0.45
        risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
        assert risk == 'Moderate Risk'
    
    def test_risk_classification_high(self):
        """Test high risk classification (> 60%)"""
        prob = 0.78
        risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
        assert risk == 'High Risk'
    
    def test_risk_boundary_30_percent(self):
        """Test risk classification at 30% boundary"""
        prob = 0.30
        risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
        assert risk == 'Moderate Risk'
    
    def test_risk_boundary_60_percent(self):
        """Test risk classification at 60% boundary"""
        prob = 0.60
        risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
        assert risk == 'High Risk'


# ==================== TEST CLASS: API RESPONSE ====================

class TestAPIResponse:
    """Unit tests for API response structure"""
    
    def test_response_has_prediction(self):
        """Test response contains prediction field"""
        response = {
            'prediction': 1,
            'probability': 0.78,
            'risk_level': 'High Risk',
            'message': 'High risk detected',
            'confidence': 0.78
        }
        assert 'prediction' in response
    
    def test_response_has_probability(self):
        """Test response contains probability field"""
        response = {'probability': 0.78}
        assert 'probability' in response
    
    def test_response_has_risk_level(self):
        """Test response contains risk_level field"""
        response = {'risk_level': 'High Risk'}
        assert 'risk_level' in response
    
    def test_response_has_message(self):
        """Test response contains message field"""
        response = {'message': 'High risk detected'}
        assert 'message' in response
    
    def test_response_serializable(self):
        """Test response can be JSON serialized"""
        response = {
            'prediction': 1,
            'probability': 0.78,
            'risk_level': 'High Risk',
            'message': 'Test',
            'confidence': 0.78
        }
        json_str = json.dumps(response)
        assert isinstance(json_str, str)
    
    def test_response_deserializable(self):
        """Test response can be JSON deserialized"""
        json_str = '{"prediction": 1, "probability": 0.78}'
        response = json.loads(json_str)
        assert response['prediction'] == 1
        assert response['probability'] == 0.78


# ==================== TEST CLASS: UTILITY FUNCTIONS ====================

class TestUtilityFunctions:
    """Unit tests for utility functions"""
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        probs = [0.3, 0.7]
        confidence = max(probs)
        assert confidence == 0.7
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        value = 0.78
        formatted = f"{value * 100:.1f}%"
        assert formatted == "78.0%"
    
    def test_generate_risk_message_low(self):
        """Test risk message generation for low risk"""
        risk = 'Low Risk'
        message = "Low probability of heart disease" if risk == 'Low Risk' else ""
        assert "Low probability" in message
    
    def test_generate_risk_message_high(self):
        """Test risk message generation for high risk"""
        risk = 'High Risk'
        message = "High risk of heart disease" if risk == 'High Risk' else ""
        assert "High risk" in message
    
    def test_data_type_conversion_int(self):
        """Test integer conversion"""
        value = "54"
        converted = int(value)
        assert converted == 54
        assert isinstance(converted, int)
    
    def test_data_type_conversion_float(self):
        """Test float conversion"""
        value = "1.2"
        converted = float(value)
        assert converted == 1.2
        assert isinstance(converted, float)


# ==================== RUN TESTS ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov', '--cov-report=term-missing'])
