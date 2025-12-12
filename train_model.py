"""
Heart Disease Prediction - Model Training Script
Run this first to train and save the model
Command: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Train the model and save it along with encoders and scaler"""
    
    print("="*60)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load the dataset
    try:
        df = pd.read_csv('heart.csv')
        print("\n✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"\n📈 Target distribution:")
        print(df['HeartDisease'].value_counts())
        print(f"\n🔍 Features: {list(df.columns)}")
    except FileNotFoundError:
        print("\n❌ Error: heart.csv not found!")
        print("Please make sure heart.csv is in the same directory.")
        return
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("\n⚠️  Missing values detected. Handling...")
        df = df.fillna(df.median(numeric_only=True))
    
    # Prepare features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    print("\n🔄 Encoding categorical variables...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   {col}: {list(le.classes_)}")
    
    # Split the data
    print("\n✂️  Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Scale features
    print("\n⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\n🌲 Training Random Forest Classifier...")
    print("   This may take a minute...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print("   ✅ Training completed!")
    
    # Evaluate on training data
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test data
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE")
    print("="*60)
    print(f"\n🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"🎯 Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['No Disease', 'Heart Disease']))
    
    print("\n🔢 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"   True Negatives:  {cm[0][0]}")
    print(f"   False Positives: {cm[0][1]}")
    print(f"   False Negatives: {cm[1][0]}")
    print(f"   True Positives:  {cm[1][1]}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model, scaler, and encoders
    print("\n💾 Saving model files...")
    joblib.dump(model, 'heart_disease_model.pkl')
    print("   ✅ heart_disease_model.pkl saved")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("   ✅ scaler.pkl saved")
    
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("   ✅ label_encoders.pkl saved")
    
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start backend:  uvicorn backend:app --reload")
    print("2. Start frontend: streamlit run frontend.py")
    print("="*60 + "\n")
    
    return model, scaler, label_encoders, test_accuracy

if __name__ == "__main__":
    train_and_save_model()
