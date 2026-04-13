# ==========================================
# RESONANCE - WINDOWS MASTER EXECUTION SCRIPT
# ==========================================

$ErrorActionPreference = "Continue"

# File and Directory Definitions
$VENV_DIR = "resonance_env"
$REQ_FILE = "requirements.txt"
$DATA_FILE = "resonance_massive_mimo_data.h5"
$WEIGHTS_DIR = "weights"
$LOGS_DIR = "logs"
$VIS_DIR = "visualizations"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "      RESONANCE: 6G MASSIVE MIMO PIPELINE       " -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# ==========================================
# STEP 1: ENVIRONMENT SETUP & ACTIVATION
# ==========================================
Write-Host "[1/5] Checking Virtual Environment..." -ForegroundColor Yellow

if (Test-Path $VENV_DIR) {
    Write-Host "[OK] Virtual environment '$VENV_DIR' found." -ForegroundColor Green
} else {
    Write-Host "[WARN] Virtual environment not found. Creating '$VENV_DIR'..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Failed to create virtual environment. Halting execution." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created successfully." -ForegroundColor Green
}

Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Cyan
# Dot-sourcing is required here so the environment variables apply to this current session
. ".\$VENV_DIR\Scripts\Activate.ps1"

if (Test-Path $REQ_FILE) {
    Write-Host "[INFO] Syncing dependencies from requirements.txt..." -ForegroundColor Cyan
    # Using 'python -m pip' ensures it explicitly uses the pip inside the activated venv
    python -m pip install --upgrade pip --quiet
    python -m pip install -r $REQ_FILE --quiet
} else {
    Write-Host "[WARN] requirements.txt not found. Skipping dependency installation." -ForegroundColor Yellow
}

# ==========================================
# STEP 2: DATA GENERATION CHECK
# ==========================================
Write-Host "`n[2/5] Checking Data Pipeline..." -ForegroundColor Yellow

if (Test-Path $DATA_FILE) {
    Write-Host "[OK] Dataset found ($DATA_FILE). Skipping DeepMIMO generation." -ForegroundColor Green
} else {
    Write-Host "[WARN] Dataset not found. Initiating Data Generation..." -ForegroundColor Yellow
    python data_gen.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Data generation failed. Halting execution." -ForegroundColor Red
        exit 1
    }
}

# ==========================================
# STEP 3: REFRESH TRAINING CACHE
# ==========================================
Write-Host "`n[3/5] Refreshing Training Cache..." -ForegroundColor Yellow

# Recreate directories if they don't exist
New-Item -ItemType Directory -Force -Path $WEIGHTS_DIR, $LOGS_DIR, $VIS_DIR | Out-Null

# Clear out old weights, tensorboard logs, and old heatmaps
Write-Host "[INFO] Wiping old weights..."
Remove-Item -Path "$WEIGHTS_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[INFO] Wiping old TensorBoard logs..."
Remove-Item -Path "$LOGS_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[INFO] Wiping old visual outputs..."
Remove-Item -Path "$VIS_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[OK] Cache cleared. Ready for fresh training run." -ForegroundColor Green

# ==========================================
# STEP 4: EXECUTE TRAINING
# ==========================================
Write-Host "`n[4/5] Initiating Neural Network Training..." -ForegroundColor Yellow
python train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Training encountered a fatal error. Halting execution." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Training sequence completed." -ForegroundColor Green

# ==========================================
# STEP 5: EVALUATION & TELECOM METRICS
# ==========================================
Write-Host "`n[5/5] Generating Performance Reports..." -ForegroundColor Yellow

Write-Host "[INFO] Running Visual Evaluation (Generating Heatmaps)..."
python eval.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Evaluation script failed." -ForegroundColor Red
}

Write-Host "`n[INFO] Executing Telecom Metrics Translation..."
python metrics.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Metrics script failed." -ForegroundColor Red
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  RESONANCE PIPELINE EXECUTED SUCCESSFULLY      " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan