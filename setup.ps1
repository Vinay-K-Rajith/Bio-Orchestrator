# BioOrchestrator setup (Windows PowerShell).
#
# Runs the full build from a clean checkout: dependencies, model training,
# ChEMBL downloads, pool cache, and verification. Safe to re-run -- each step
# is skipped if its artifact already exists, so a failed ChEMBL download can be
# retried without retraining anything.
#
# Usage:   .\setup.ps1            normal run
#          .\setup.ps1 -Force     rebuild everything from scratch

param([switch]$Force)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($n, $msg) {
    Write-Host ""
    Write-Host "=== [$n] $msg ===" -ForegroundColor Cyan
}

function Need($path) {
    return $Force -or -not (Test-Path $path)
}

Step 1 "Python dependencies"
python -m pip install -q -r backend/requirements.txt
if (-not $?) { throw "pip install failed" }
Write-Host "  installed" -ForegroundColor Green

Step 2 "Training dataset"
if (Need "backend/data/egfr_chembl_labeled.csv") {
    if (Test-Path "egfr_chembl_labeled_dataset.csv") {
        Copy-Item "egfr_chembl_labeled_dataset.csv" "backend/data/egfr_chembl_labeled.csv"
        Write-Host "  staged from repo root" -ForegroundColor Green
    } else {
        throw "egfr_chembl_labeled_dataset.csv not found in $root"
    }
} else { Write-Host "  already present" -ForegroundColor DarkGray }

Step 3 "Affinity surrogate (~70s)"
if (Need "backend/models/egfr_xgboost.json") {
    python backend/scripts/train_surrogate.py
    if (-not $?) { throw "surrogate training failed" }
} else { Write-Host "  already trained" -ForegroundColor DarkGray }

Step 4 "ChEMBL downloads (hERG + kinase pool)"
# ChEMBL has intermittent 500s; retry rather than failing the whole setup.
if (Need "backend/data/herg_chembl.csv") {
    for ($i = 1; $i -le 5; $i++) {
        python backend/scripts/fetch_chembl.py herg
        if ($?) { break }
        Write-Host "  ChEMBL unavailable, retry $i/5 in 60s" -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
} else { Write-Host "  hERG data already present" -ForegroundColor DarkGray }

if (Need "backend/data/candidate_pool.csv") {
    for ($i = 1; $i -le 5; $i++) {
        python backend/scripts/fetch_chembl.py kinase_pool
        if ($?) { break }
        Write-Host "  ChEMBL unavailable, retry $i/5 in 60s" -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
} else { Write-Host "  candidate pool already present" -ForegroundColor DarkGray }

Step 5 "hERG classifier (~60s)"
if (Need "backend/models/herg_xgboost.json") {
    if (Test-Path "backend/data/herg_chembl.csv") {
        python backend/scripts/train_herg.py
    } else {
        Write-Host "  skipped -- no hERG data (pipeline runs without it)" -ForegroundColor Yellow
    }
} else { Write-Host "  already trained" -ForegroundColor DarkGray }

Step 6 "Pool feature cache (~90s)"
if (Need "backend/models/pool_cache/pool_features.npy") {
    python backend/scripts/build_pool_cache.py
    if (-not $?) { throw "pool cache build failed" }
} else { Write-Host "  already built" -ForegroundColor DarkGray }

Step 7 "Frontend dependencies"
if (Need "frontend/node_modules") {
    Push-Location frontend
    npm install --silent
    Pop-Location
} else { Write-Host "  already installed" -ForegroundColor DarkGray }

Step 8 "Verification"
python backend/scripts/verify.py
if (-not $?) { throw "verification failed" }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Backend:  cd backend; python -m uvicorn app.server:app --port 8000"
Write-Host "  Frontend: cd frontend; npm run dev"
Write-Host "  Then open http://localhost:5173"
