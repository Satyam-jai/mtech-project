"""
Heart Disease Prediction - FastAPI Backend
Run with: uvicorn backend:app --reload
API will be available at: http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Machine Learning API for heart disease risk prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, scaler, and encoders
model = None
scaler = None
label_encoders = None

# Load model and preprocessing objects
@app.on_event("startup")
async def load_model():
    """Load the trained model, scaler, and encoders on startup"""
    global model, scaler, label_encoders
    
    print("\n" + "="*60)
    print("LOADING MODEL AND PREPROCESSING OBJECTS")
    print("="*60)
    
    try:
        # Check if files exist
        required_files = ['heart_disease_model.pkl', 'scaler.pkl', 'label_encoders.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"\n❌ Missing files: {', '.join(missing_files)}")
            print("Please run 'python train_model.py' first to create these files.")
            return
        
        # Load model
        model = joblib.load('heart_disease_model.pkl')
        print("✅ Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        print("✅ Scaler loaded successfully")
        
        # Load label encoders
        label_encoders = joblib.load('label_encoders.pkl')
        print("✅ Label encoders loaded successfully")
        
        print("\n🚀 Backend is ready to accept requests!")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Please run 'python train_model.py' first.")


# Pydantic models for request/response
class PatientData(BaseModel):
    """Patient data model for prediction request"""
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Sex: str = Field(..., description="M (Male) or F (Female)")
    ChestPainType: str = Field(..., description="ATA, NAP, ASY, or TA")
    RestingBP: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=0, le=600, description="Cholesterol level (mg/dl)")
    FastingBS: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=Yes, 0=No)")
    RestingECG: str = Field(..., description="Normal, ST, or LVH")
    MaxHR: int = Field(..., ge=50, le=220, description="Maximum heart rate achieved")
    ExerciseAngina: str = Field(..., description="Y (Yes) or N (No)")
    Oldpeak: float = Field(..., ge=-10, le=10, description="ST depression induced by exercise")
    ST_Slope: str = Field(..., description="Up, Flat, or Down")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int
    probability: float
    risk_level: str
    message: str
    confidence: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    message: str


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and scaler is not None and label_encoders is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        message="All systems operational" if model_loaded else "Model not loaded. Run train_model.py first."
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Make a heart disease prediction based on patient data
    
    Returns:
    - prediction: 0 (No Disease) or 1 (Heart Disease)
    - probability: Probability of heart disease (0-1)
    - risk_level: Low Risk, Moderate Risk, or High Risk
    - message: Detailed message about the prediction
    - confidence: Confidence level of the prediction
    """
    
    # Check if model is loaded
    if model is None or scaler is None or label_encoders is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure train_model.py has been run and model files exist."
        )
    
    try:
        # Convert patient data to DataFrame
        data = pd.DataFrame([patient.dict()])
        
        # Validate and encode categorical variables
        for col, le in label_encoders.items():
            if col in data.columns:
                value = data[col].iloc[0]
                if value not in le.classes_:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value '{value}' for {col}. Allowed values: {list(le.classes_)}"
                    )
                data[col] = le.transform(data[col])
        
        # Scale features
        data_scaled = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(data_scaled)[0]
        probabilities = model.predict_proba(data_scaled)[0]
        
        # Get disease probability
        disease_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        
        # Determine risk level and message
        if disease_probability < 0.3:
            risk_level = "Low Risk"
            message = "Based on the provided data, the patient shows a LOW risk of heart disease. Continue maintaining a healthy lifestyle with regular exercise, balanced diet, and routine check-ups."
        elif disease_probability < 0.6:
            risk_level = "Moderate Risk"
            message = "Based on the provided data, the patient shows MODERATE risk of heart disease. It's recommended to consult with a healthcare professional for preventive measures and lifestyle modifications."
        else:
            risk_level = "High Risk"
            message = "Based on the provided data, the patient shows HIGH risk of heart disease. Please consult with a healthcare professional IMMEDIATELY for a comprehensive cardiovascular evaluation and treatment plan."
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=disease_probability,
            risk_level=risk_level,
            message=message,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the trained model"""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "Random Forest Classifier",
        "algorithm": "Random Forest",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features": [
            "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
            "Oldpeak", "ST_Slope"
        ],
        "feature_count": 11,
        "output_classes": ["No Disease (0)", "Heart Disease (1)"],
        "preprocessing": {
            "scaling": "StandardScaler",
            "encoding": "LabelEncoder for categorical features"
        }
    }


@app.get("/feature-info", tags=["Model"])
async def feature_info():
    """Get detailed information about input features"""
    
    return {
        "features": {
            "Age": {
                "type": "Numeric",
                "range": "1-120 years",
                "description": "Patient's age in years"
            },
            "Sex": {
                "type": "Categorical",
                "values": ["M", "F"],
                "description": "Patient's biological sex"
            },
            "ChestPainType": {
                "type": "Categorical",
                "values": ["TA", "ATA", "NAP", "ASY"],
                "description": "Type of chest pain experienced",
                "details": {
                    "TA": "Typical Angina",
                    "ATA": "Atypical Angina",
                    "NAP": "Non-Anginal Pain",
                    "ASY": "Asymptomatic"
                }
            },
            "RestingBP": {
                "type": "Numeric",
                "range": "0-300 mm Hg",
                "description": "Resting blood pressure"
            },
            "Cholesterol": {
                "type": "Numeric",
                "range": "0-600 mg/dl",
                "description": "Serum cholesterol level"
            },
            "FastingBS": {
                "type": "Binary",
                "values": [0, 1],
                "description": "Fasting blood sugar > 120 mg/dl (1=Yes, 0=No)"
            },
            "RestingECG": {
                "type": "Categorical",
                "values": ["Normal", "ST", "LVH"],
                "description": "Resting electrocardiogram results",
                "details": {
                    "Normal": "Normal ECG",
                    "ST": "ST-T wave abnormality",
                    "LVH": "Left ventricular hypertrophy"
                }
            },
            "MaxHR": {
                "type": "Numeric",
                "range": "50-220 bpm",
                "description": "Maximum heart rate achieved"
            },
            "ExerciseAngina": {
                "type": "Binary",
                "values": ["Y", "N"],
                "description": "Exercise-induced angina (chest pain)"
            },
            "Oldpeak": {
                "type": "Numeric",
                "range": "-10 to 10",
                "description": "ST depression induced by exercise relative to rest"
            },
            "ST_Slope": {
                "type": "Categorical",
                "values": ["Up", "Flat", "Down"],
                "description": "Slope of the peak exercise ST segment"
            }
        }
    }


# Run with: uvicorn backend:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
