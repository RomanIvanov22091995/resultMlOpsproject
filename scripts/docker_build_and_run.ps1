# Сборка Docker-образа PD-модели, запуск контейнера и проверка API (Windows PowerShell).
# Использование: .\scripts\docker_build_and_run.ps1 [-Port 8000]
param([int]$Port = 8000)

# Docker пишет прогресс сборки в stderr, поэтому ошибки определяем
# по коду возврата ($LASTEXITCODE), а не по $ErrorActionPreference.
$ErrorActionPreference = "Continue"
$Image = "credit-scoring-api:latest"
$Container = "credit-scoring-api"
Set-Location (Join-Path $PSScriptRoot "..")

$Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

function Assert-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! $Step failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path "models\credit_default_model.pkl")) {
    Write-Host "!!! Model not found. Run 'dvc repro' (or 'dvc pull') first." -ForegroundColor Red
    exit 1
}

Write-Host ">>> Building image $Image"
docker build -t $Image .
Assert-Success "docker build"

Write-Host ">>> (Re)starting container $Container on port $Port"
$existing = docker ps -aq --filter "name=^${Container}$"
if ($existing) { docker rm -f $Container | Out-Null }
docker run -d --name $Container -p "${Port}:8000" $Image
Assert-Success "docker run"

Write-Host ">>> Smoke test"
& $Python scripts/smoke_test_api.py --url "http://localhost:$Port" --wait 90
Assert-Success "smoke test"

Write-Host ">>> API is running: http://localhost:$Port/docs" -ForegroundColor Green
Write-Host ">>> Stop it with: docker rm -f $Container"
