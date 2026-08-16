$ErrorActionPreference = "Stop"

python -m PyInstaller --noconfirm --clean absorption_trainer.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller завершил сборку с кодом $LASTEXITCODE."
}

Write-Host "Готово: dist\AbsorptionTrainer.exe"
