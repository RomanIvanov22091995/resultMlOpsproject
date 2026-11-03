# Сборка Docker-образа PD-модели, запуск контейнера и проверка API (Windows PowerShell).
# Использование: .\scripts\docker_build_and_run.ps1 [-Port 8000]
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$Image = "credit-scoring-api:latest"
$Container = "credit-scoring-api"
Set-Location (Join-Path $PSScriptRoot "..")

$Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

if (-not (Test-Path "models\credit_default_model.pkl")) {
    Write-Error "Model not found. Run 'dvc repro' (or 'dvc pull') first."
}

Write-Host ">>> Building image $Image"
docker build -t $Image .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host ">>> (Re)starting container $Container on port $Port"
$existing = docker ps -aq --filter "name=^${Container}$"
if ($existing) { docker rm -f $Container | Out-Null }
docker run -d --name $Container -p "${Port}:8000" $Image
if ($LASTEXITCODE -ne 0) { throw "docker run failed" }

Write-Host ">>> Smoke test"
& $Python scripts/smoke_test_api.py --url "http://localhost:$Port" --wait 90
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

Write-Host ">>> API is running: http://localhost:$Port/docs"
Write-Host ">>> Stop it with: docker rm -f $Container"
