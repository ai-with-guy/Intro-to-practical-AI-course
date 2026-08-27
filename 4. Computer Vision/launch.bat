@echo off
echo Launching your local Jupyter environment via uv...

set "TORCH_EXTRA=cuda"
for /f %%i in ('powershell -NoProfile -Command "$gpus = Get-CimInstance Win32_VideoController; if (($gpus.AdapterCompatibility -match 'Intel') -and -not ($gpus.AdapterCompatibility -match 'NVIDIA')) { 'xpu' }"') do set "TORCH_EXTRA=xpu"

if "%TORCH_EXTRA%"=="xpu" echo Intel-only GPU system detected; using the PyTorch XPU build.
uv run --exact --extra %TORCH_EXTRA% jupyter lab "computer_vision.ipynb"
