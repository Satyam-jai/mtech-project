import pytest
import requests
import time
import json


class TestEndToEndWorkflow:
    """System tests for complete end-to-end workflows"""
    
    def test_complete_prediction_workflow_low_risk(self):
        """Test complete workflow for low-risk patient"""
        # Step 1: Prepare patient data
        patient_data = {
            "Age": 40, "Sex": "F", "ChestPainType": "ATA",
            "RestingBP": 120, "Cholesterol": 180, "FastingBS": 0,
            "RestingECG": "Normal", "MaxHR": 170, "ExerciseAngina": "N",
            "Oldpeak": 0.0, "ST_Slope": "Up"
        }
        
        # Step 2: Send to API
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=patient_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Step 3: Verify response structure
                assert "prediction" in result
                assert "probability" in result
                assert "risk_level" in result
                assert "message" in result
                
                # Step 4: Verify prediction logic
                assert result["prediction"] in [0, 1]
                assert 0 <= result["probability"] <= 1
                
                # Step 5: Verify risk classification
                prob = result["probability"]
                expected_risk = 'Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.6 else 'High Risk'
                assert result["risk_level"] == expected_risk
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
    
    def test_complete_prediction_workflow_high_risk(self):
        """Test complete workflow for high-risk patient"""
        patient_data = {
            "Age": 65, "Sex": "M", "ChestPainType": "ASY",
            "RestingBP": 160, "Cholesterol": 300, "FastingBS": 1,
            "RestingECG": "ST", "MaxHR": 100, "ExerciseAngina": "Y",
            "Oldpeak": 3.0, "ST_Slope": "Flat"
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=patient_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                assert "prediction" in result
                assert "probability" in result
                # High risk patients typically have higher probability
                assert result["probability"] > 0.3
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
    
    def test_system_availability_check(self):
        """Test system availability and health"""
        try:
            # Check backend health
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            assert "status" in health_data
            assert "model_loaded" in health_data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
    
    def test_error_recovery_invalid_input(self):
        """Test system recovers from invalid input"""
        invalid_data = {
            "Age": -10,  # Invalid age
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
                "http://localhost:8000/predict",
                json=invalid_data,
                timeout=10
            )
            
            # Should return error
            assert response.status_code in [400, 422]
            
            # System should still be responsive after error
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            assert health_response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
    
    def test_stress_sequential_predictions(self):
        """Test system handles sequential predictions"""
        patient_data = {
            "Age": 54, "Sex": "M", "ChestPainType": "ASY",
            "RestingBP": 140, "Cholesterol": 239, "FastingBS": 0,
            "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
            "Oldpeak": 1.2, "ST_Slope": "Flat"
        }
        
        try:
            success_count = 0
            for i in range(10):
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=patient_data,
                    timeout=10
                )
                if response.status_code == 200:
                    success_count += 1
            
            # At least 80% should succeed
            assert success_count >= 8, f"Only {success_count}/10 requests succeeded"
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
