@echo off
setlocal
cd /d "%~dp0"
rem Make the shared attention viewer available to Jupyter and its kernels.
for %%I in ("%~dp0..\5. Transformer") do set "VIEWER_PATH=%%~fI"
if defined PYTHONPATH (
    set "PYTHONPATH=%VIEWER_PATH%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%VIEWER_PATH%"
)
echo Launching your local Jupyter environment via uv...

if not defined TORCH_EXTRA (
    set "TORCH_EXTRA=cuda"
    for /f %%i in ('powershell -NoProfile -Command "$gpus = Get-CimInstance Win32_VideoController; if (($gpus.AdapterCompatibility -match 'Intel') -and -not ($gpus.AdapterCompatibility -match 'NVIDIA')) { 'xpu' }"') do set "TORCH_EXTRA=xpu"
)

if "%TORCH_EXTRA%"=="xpu" echo Using the PyTorch XPU build.
uv run --exact --extra %TORCH_EXTRA% jupyter lab "attention.ipynb"
