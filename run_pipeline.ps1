# GPCR Biased Signaling & Binding Predictor Pipeline Orchestrator
# For Windows PowerShell

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "🧬 GPCR Biased Signaling & Binding Predictor 🧬" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Check Python installation
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in system PATH." -ForegroundColor Red
    Exit
}

# 2. Virtual Environment Setup
$VENV_DIR = "venv"
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "--> Creating Python Virtual Environment (venv)..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
}

Write-Host "--> Activating Virtual Environment..." -ForegroundColor Yellow
& ".\$VENV_DIR\Scripts\Activate.ps1"

# 3. Installing dependencies
Write-Host "--> Installing project requirements from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# 4. Running automated tests
Write-Host "`n--> Phase 3: Executing Automated Test Suite..." -ForegroundColor Yellow
pytest tests/test_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some unit tests failed. Please review the errors." -ForegroundColor Red
} else {
    Write-Host "Success: All unit tests passed cleanly!" -ForegroundColor Green
}

# 5. Training the model
Write-Host "`n--> Phase 4: Training Multi-Task Model on Curated GPCR Dataset..." -ForegroundColor Yellow
python src/train.py

# 6. Running virtual screen targeting 5-HT2A
Write-Host "`n--> Phase 5: Executing Virtual Screen Targeting Human 5-HT2A..." -ForegroundColor Yellow
python src/screen.py

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   Pipeline Finished Successfully! " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "To boot the interactive visualization dashboard, run:" -ForegroundColor Yellow
Write-Host "   streamlit run src/app.py" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Green
