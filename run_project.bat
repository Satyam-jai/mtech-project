@echo off
echo =============================
echo Starting Heart Disease Project
echo =============================

REM Start FastAPI backend
start cmd /k "py -3.11 -m uvicorn backend:app --reload"

REM Wait a few seconds for backend to start
timeout /t 5 >nul

REM Start Streamlit frontend
start cmd /k "py -3.11 -m streamlit run frontend.py"
