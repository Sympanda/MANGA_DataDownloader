# Create a clean MaNGA conda env (CUDA PyTorch + astro/ML via conda).
#
# Usage (from repo root):
#   .\scripts\setup_manga_env.ps1
#   .\scripts\setup_manga_env.ps1 -EnvName manga -Recreate

param(
    [string]$EnvName = "manga",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if ($Recreate) {
    conda env remove -n $EnvName -y 2>$null
}

$exists = conda env list | Select-String "^\s*$EnvName\s"
if ($exists) {
    Write-Host "Env '$EnvName' already exists."
    Write-Host "Updating from environment.yml..."
    conda env update -n $EnvName -f environment.yml --prune
} else {
    Write-Host "Creating conda env '$EnvName' from environment.yml..."
    conda env create -f environment.yml -n $EnvName
}

Write-Host ""
Write-Host "Verify CUDA:"
conda run -n $EnvName python -c "import torch; print('torch', torch.__version__); print('cuda built', torch.backends.cuda.is_built()); print('cuda available', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

Write-Host ""
Write-Host "Done. Activate with:  conda activate $EnvName"
