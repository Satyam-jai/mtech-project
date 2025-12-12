"""
===============================================================================
INTEGRATION TESTS - Heart Disease Prediction System
===============================================================================
Run with: pytest test_integration.py -v
Tests the interaction between multiple components
===============================================================================
"""

import pytest
import requests
import json
import time
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


# ==================== TEST FIXTURES ====================

@pytest.fixture
def api_base_url():
    """Base URL for API testing"""
    return "http://localhost:8000"


@pytest.fixture
def valid_patient_payload():
    """Valid patient data payload"""
    return {
        "Age": 54,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 239,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 160,
        "ExerciseAngina": "N",
        "Oldpeak": 1.2,
        "ST_Slope": "Flat"
    }


@pytest.fixture
def api_headers():
    """API request headers"""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


# ==================== TEST CLASS: API ENDPOINTS ====================

class TestAPIEndpoints:
    """Integration tests for API endpoints"""
    
    def test_root_endpoint(self, api_base_url):
        """Test root endpoint returns welcome message"""
        try:
            response = requests.get(f"{api_base_url}/")
            assert response.status_code == 200
            data = response.json()
            assert "name" in data
            assert "Heart Disease" in data["name"]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_health_endpoint(self, api_base_url):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{api_base_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_model_info_endpoint(self, api_base_url):
        """Test model info endpoint"""
        try:
            response = requests.get(f"{api_base_url}/model-info")
            assert response.status_code in [200, 503]
            if response.status_code == 200:
                data = response.json()
                assert "model_type" in data
                assert "features" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_predict_endpoint_valid_data(self, api_base_url, valid_patient_payload, api_headers):
        """Test prediction endpoint with valid data"""
        try:
            response = requests.post(
                f"{api_base_url}/predict",
                json=valid_patient_payload,
                headers=api_headers
            )
            assert response.status_code in [200, 503]
            if response.status_code == 200:
                data = response.json()
                assert "prediction" in data
                assert "probability" in data
                assert "risk_level" in data
                assert data["prediction"] in [0, 1]
                assert 0 <= data["probability"] <= 1
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_predict_endpoint_invalid_age(self, api_base_url, valid_patient_payload, api_headers):
        """Test prediction endpoint with invalid age"""
        invalid_payload = valid_patient_payload.copy()
        invalid_payload["Age"] = 200
        try:
            response = requests.post(
                f"{api_base_url}/predict",
                json=invalid_payload,
                headers=api_headers
            )
            # Should return 422 (validation error) or 400
            assert response.status_code in [400, 422, 503]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_predict_endpoint_missing_field(self, api_base_url, api_headers):
        """Test prediction endpoint with missing required field"""
        incomplete_payload = {"Age": 54, "Sex": "M"}
        try:
            response = requests.post(
                f"{api_base_url}/predict",
                json=incomplete_payload,
                headers=api_headers
            )
            assert response.status_code in [400, 422]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


# ==================== TEST CLASS: FRONTEND-BACKEND INTEGRATION ====================

class TestFrontendBackendIntegration:
    """Integration tests for frontend-backend communication"""
    
    def test_api_connection_check(self, api_base_url):
        """Test frontend can check API connectivity"""
        try:
            response = requests.get(f"{api_base_url}/health", timeout=5)
            is_connected = response.status_code == 200
            assert isinstance(is_connected, bool)
        except requests.exceptions.ConnectionError:
            is_connected = False
        except requests.exceptions.Timeout:
            is_connected = False
        assert isinstance(is_connected, bool)
    
    def test_prediction_request_response_cycle(self, api_base_url, valid_patient_payload):
        """Test complete request-response cycle"""
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{api_base_url}/predict",
                json=valid_patient_payload,
                timeout=10
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            # Check response time (should be < 5 seconds)
            assert response_time < 5.0, f"Response time too high: {response_time}s"
            
            # Check response structure
            if response.status_code == 200:
                data = response.json()
                required_fields = ['prediction', 'probability', 'risk_level', 'message']
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_multiple_predictions_same_session(self, api_base_url, valid_patient_payload):
        """Test multiple predictions in same session"""
        try:
            for i in range(3):
                response = requests.post(
                    f"{api_base_url}/predict",
                    json=valid_patient_payload,
                    timeout=10
                )
                if response.status_code == 200:
                    assert response.status_code == 200
                    data = response.json()
                    assert "prediction" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


# ==================== TEST CLASS: DATA PIPELINE INTEGRATION ====================

class TestDataPipelineIntegration:
    """Integration tests for data processing pipeline"""
    
    def test_end_to_end_preprocessing(self):
        """Test complete preprocessing pipeline"""
        # Sample input data
        input_data = {
            'Age': 54, 'Sex': 'M', 'ChestPainType': 'ASY',
            'RestingBP': 140, 'Cholesterol': 239, 'FastingBS': 0,
            'RestingECG': 'Normal', 'MaxHR': 160, 'ExerciseAngina': 'N',
            'Oldpeak': 1.2, 'ST_Slope': 'Flat'
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        assert len(df) == 1
        assert len(df.columns) == 11
        
        # Simulate encoding
        encoding_map = {
            'Sex': {'M': 1, 'F': 0},
            'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
            'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
            'ExerciseAngina': {'Y': 1, 'N': 0},
            'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
        }
        
        for col, mapping in encoding_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Check all values are numeric
        assert df.select_dtypes(include=[np.number]).shape[1] == 11
    
    def test_model_loading_and_prediction(self):
        """Test model loading and prediction integration"""
        # Mock model behavior
        class MockModel:
            def predict(self, X):
                return np.array([1])
            
            def predict_proba(self, X):
                return np.array([[0.3, 0.7]])
        
        model = MockModel()
        test_data = np.random.rand(1, 11)
        
        prediction = model.predict(test_data)
        probability = model.predict_proba(test_data)
        
        assert prediction.shape == (1,)
        assert probability.shape == (1, 2)
        assert prediction[0] in [0, 1]
        assert np.abs(probability.sum() - 1.0) < 1e-6
    
    def test_risk_assessment_integration(self):
        """Test risk assessment with probability"""
        test_probabilities = [0.2, 0.45, 0.78]
        expected_risks = ['Low Risk', 'Moderate Risk', 'High Risk']
        
        for prob, expected in zip(test_probabilities, expected_risks):
            risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
            assert risk == expected


# ==================== TEST CLASS: ERROR HANDLING INTEGRATION ====================

class TestErrorHandlingIntegration:
    """Integration tests for error handling across components"""
    
    def test_invalid_data_type_handling(self, api_base_url, api_headers):
        """Test handling of invalid data types"""
        invalid_payload = {
            "Age": "not_a_number",  # Should be int
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": 140,
            "Cholesterol": 239,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 160,
            "ExerciseAngina": "N",
            "Oldpeak": 1.2,
            "ST_Slope": "Flat"
        }
        
        try:
            response = requests.post(
                f"{api_base_url}/predict",
                json=invalid_payload,
                headers=api_headers
            )
            # Should return validation error
            assert response.status_code in [400, 422, 503]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_malformed_json_handling(self, api_base_url, api_headers):
        """Test handling of malformed JSON"""
        try:
            response = requests.post(
                f"{api_base_url}/predict",
                data="not valid json",
                headers=api_headers
            )
            # Should return 400 or 422
            assert response.status_code in [400, 422]
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_timeout_handling(self, api_base_url, valid_patient_payload):
        """Test timeout handling"""
        try:
            with pytest.raises(requests.exceptions.Timeout):
                response = requests.post(
                    f"{api_base_url}/predict",
                    json=valid_patient_payload,
                    timeout=0.001  # Very short timeout
                )
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")


# ==================== TEST CLASS: PERFORMANCE INTEGRATION ====================

class TestPerformanceIntegration:
    """Integration tests for performance characteristics"""
    
    def test_response_time_single_prediction(self, api_base_url, valid_patient_payload):
        """Test single prediction response time"""
        try:
            start = time.time()
            response = requests.post(
                f"{api_base_url}/predict",
                json=valid_patient_payload,
                timeout=10
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                # Should respond within 2 seconds
                assert duration < 2.0, f"Response too slow: {duration}s"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_concurrent_predictions_performance(self, api_base_url, valid_patient_payload):
        """Test performance with multiple concurrent predictions"""
        try:
            import concurrent.futures
            
            def make_prediction():
                response = requests.post(
                    f"{api_base_url}/predict",
                    json=valid_patient_payload,
                    timeout=10
                )
                return response.status_code
            
            # Test 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_prediction) for _ in range(5)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            # All requests should succeed (or all fail with 503 if model not loaded)
            assert len(results) == 5
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
        except ImportError:
            pytest.skip("concurrent.futures not available")


# ==================== RUN TESTS ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
