@echo off
REM ==============================================================================
REM FraudShield AI — Automated Windows Startup Script
REM Intelligent Financial Fraud Detection Using Machine Learning and Explainable AI
REM ==============================================================================

echo ==============================================================================
echo                      FraudShield AI Startup Manager
echo ==============================================================================
echo.

REM 1. Verify Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)

echo [1/4] Checking Python environment...
python -c "import flask, sklearn, xgboost, shap; print('Dependencies verified.')" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Dependencies not fully installed. Installing requirements...
    pip install -r requirements.txt
) else (
    echo [OK] Core dependencies verified.
)
echo.

echo [2/4] Running automated test suite...
python -m pytest tests/ --quiet
if errorlevel 1 (
    echo [WARN] Some tests failed. Proceeding with caution...
) else (
    echo [OK] All automated tests passed successfully.
)
echo.

echo [3/4] Checking trained models and artifacts...
if not exist "models\xgboost.joblib" (
    echo [INFO] Trained models not found. Running training pipeline in demo mode...
    python train_pipeline.py --demo
) else (
    echo [OK] Model artifacts found.
)
echo.

echo [4/4] Starting FraudShield AI Web Dashboard...
echo.
echo ==============================================================================
echo FraudShield AI Dashboard will be available at:
echo    http://localhost:5000
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo ==============================================================================
echo.

REM Automatically open default browser after launching
start http://localhost:5000

REM Launch Flask application server
python backend/app.py

pause
