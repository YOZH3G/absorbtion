$ErrorActionPreference = "Stop"

python -m PyInstaller --noconfirm --clean dynamics_control_lab.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller завершил сборку с кодом $LASTEXITCODE."
}

$taskOutput = Join-Path (Get-Location) "build\release-check.json"
$taskProcess = Start-Process -FilePath (Resolve-Path "dist\DynamicsControlLab.exe").Path -ArgumentList @("--smoke-test", ('"' + $taskOutput + '"')) -WindowStyle Hidden -PassThru
if (-not $taskProcess.WaitForExit(60000)) {
    Stop-Process -Id $taskProcess.Id
    throw "Проверка приложения превысила 60 секунд."
}
$taskProcess.Refresh()
if ($taskProcess.ExitCode -ne 0) {
    Get-Content -LiteralPath $taskOutput -ErrorAction SilentlyContinue
    throw "Проверка приложения завершилась с кодом $($taskProcess.ExitCode)."
}
$taskReport = Get-Content -LiteralPath $taskOutput -Raw | ConvertFrom-Json
if ($taskReport.status -ne "passed" -or -not $taskReport.frozen) {
    throw "Упакованное приложение не прошло проверку."
}

$performanceOutput = Join-Path (Get-Location) "build\performance-check.json"
$performanceProcess = Start-Process -FilePath (Resolve-Path "dist\DynamicsControlLab.exe").Path -ArgumentList @("--performance-check", ('"' + $performanceOutput + '"')) -WindowStyle Hidden -PassThru
if (-not $performanceProcess.WaitForExit(90000)) {
    Stop-Process -Id $performanceProcess.Id
    throw "Измерение производительности превысило 90 секунд."
}
$performanceProcess.Refresh()
$performanceReport = Get-Content -LiteralPath $performanceOutput -Raw | ConvertFrom-Json
if ($performanceProcess.ExitCode -ne 0 -or $performanceReport.status -ne "passed" -or -not $performanceReport.frozen) {
    throw "Измерение производительности EXE завершилось с ошибкой."
}

Write-Host "Готово: dist\DynamicsControlLab.exe"
