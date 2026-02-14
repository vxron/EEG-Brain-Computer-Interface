# C:\Users\fsdma\capstone\capstone\test_training_bug.ps1

$RepoRoot = "C:\Users\fsdma\capstone\capstone"
$BuildDir = "$RepoRoot\build_bisect"   # separate dir so we don't pollute your normal build

# ── 1. Clean + Rebuild ───────────────────────────────────────────────────────
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

Push-Location $RepoRoot
cmake -B $BuildDir -DCMAKE_BUILD_TYPE=Debug -DUSE_FAKE_ACQ=ON
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMAKE CONFIGURE FAILED - skipping"
    Pop-Location
    exit 125
}

cmake --build $BuildDir --target CapstoneProject --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Host "BUILD FAILED - skipping this commit"
    Pop-Location
    exit 125
}
Pop-Location

# ── 2. Run training script directly ──────────────────────────────────────────
$DataDir  = "C:\Users\fsdma\capstone\capstone\data\victoria\2026-02-12_11-03-01"
$ModelDir = "C:\Users\fsdma\capstone\capstone\models\victoria\2026-02-12_11-03-01"
$Script   = "$RepoRoot\CapstoneProject\model train\python\train_ssvep.py"

python $Script --data $DataDir --model $ModelDir --arch "CNN" --calibsetting "most_recent_only" --tunehparams "OFF" --zscorenormalization "ON"

# ── 3. Parse train_result.json ────────────────────────────────────────────────
$ResultPath = "$ModelDir\train_result.json"

if (-not (Test-Path $ResultPath)) {
    Write-Host "train_result.json missing"
    exit 1
}

$json = Get-Content $ResultPath | ConvertFrom-Json

foreach ($issue in $json.issues) {
    if ($issue.stage -eq "LOAD" -and $issue.message -like "*mismatch*") {
        Write-Host "BUG FOUND: window length mismatch (T=$($issue.details.T) target=$($issue.details.target_T))"
        exit 1
    }
}

foreach ($issue in $json.issues) {
    if ($issue.stage -eq "PAIR_SEARCH") {
        Write-Host "BUG FOUND: PAIR_SEARCH failure"
        exit 1
    }
}

if ($json.train_ok -eq $true) {
    Write-Host "Training OK - this commit is GOOD"
    exit 0
}

Write-Host "Training failed with unknown reason"
exit 1