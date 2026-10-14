@echo off
echo Launching your local Jupyter environment via uv...
@echo off
for /r %%f in (*.ipynb) do (
    echo %%f | findstr /i /c:".ipynb_checkpoints" >nul || uv run jupyter nbconvert --clear-output --inplace "%%f"
)
uv run jupyter lab
 