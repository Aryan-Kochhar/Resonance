# ==========================================
# RESONANCE - WINDOWS MASTER EXECUTION SCRIPT
# ==========================================

$ErrorActionPreference = "Stop"

# File and Directory Definitions
$DATA_FILE = "resonance_massive_mimo_data.h5"
$WEIGHTS_DIR = "weights"
$LOGS_DIR = "logs"
$VIS_DIR = "visualizations"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "      RESONANCE: 6G MASSIVE MIMO PIPELINE       " -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# ==========================================
# STEP 1: DATA GENERATION CHECK
# ==========================================
Write-Host "[1/4] Checking Data Pipeline..." -ForegroundColor Yellow

if (Test-Path $DATA_FILE) {
    Write-Host " Dataset found ($DATA_FILE). Skipping DeepMIMO generation." -ForegroundColor Green
} else {
    Write-Host " Dataset not found. Initiating Data Generation..." -ForegroundColor Yellow
    python data_gen.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host " Data generation failed. Halting execution." -ForegroundColor Red
        exit 1
    }
}

# ==========================================
# STEP 2: REFRESH TRAINING CACHE
# ==========================================
Write-Host "`n[2/4] Refreshing Training Cache..." -ForegroundColor Yellow

# Recreate directories if they don't exist
New-Item -ItemType Directory -Force -Path $WEIGHTS_DIR, $LOGS_DIR, $VIS_DIR | Out-Null

# Clear out old weights, tensorboard logs, and old heatmaps
Write-Host " Wiping old weights..."
Remove-Item -Path "$WEIGHTS_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host " Wiping old TensorBoard logs..."
Remove-Item -Path "$LOGS_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host " Wiping old visual outputs..."
Remove-Item -Path "$VIS_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host " Cache cleared. Ready for fresh training run." -ForegroundColor Green

# ==========================================
# STEP 3: EXECUTE TRAINING
# ==========================================
Write-Host "`n[3/4] Initiating Neural Network Training..." -ForegroundColor Yellow
python train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host " Training encountered a fatal error. Halting execution." -ForegroundColor Red
    exit 1
}
Write-Host " Training sequence completed." -ForegroundColor Green

# ==========================================
# STEP 4: EVALUATION & TELECOM METRICS
# ==========================================
Write-Host "`n[4/4] Generating Performance Reports..." -ForegroundColor Yellow

Write-Host " Running Visual Evaluation (Generating Heatmaps)..."
python eval.py
if ($LASTEXITCODE -ne 0) {
    Write-Host " Evaluation script failed." -ForegroundColor Red
}

Write-Host "`n Executing Telecom Metrics Translation..."
python metrics.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Metrics script failed." -ForegroundColor Red
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  RESONANCE PIPELINE EXECUTED SUCCESSFULLY      " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan