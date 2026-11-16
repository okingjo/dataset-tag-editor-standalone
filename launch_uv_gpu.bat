@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem === GPU launcher (skips uv auto-sync) =====================================
rem Use this AFTER manually installing GPU packages into .venv, e.g. CUDA
rem PyTorch for the timm/BLIP interrogators:
rem   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
rem Plain launch_uv.bat re-syncs from pyproject.toml and would restore the
rem CPU-only torch pin. ONNX taggers (WD/PixAI/Z3D) already default to
rem DirectML on Windows via pyproject.toml - no manual step needed for them.
set "UV_CACHE_DIR=%~dp0runtime\uv_cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0runtime\python"
set "HF_HOME=%~dp0runtime\hf_cache"
set "GRADIO_TEMP_DIR=%~dp0runtime\gradio_tmp"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "UV_NO_SYNC=1"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv was not found on PATH.
    echo         Install it once from https://docs.astral.sh/uv/  e.g. in PowerShell:
    echo         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

uv run scripts\launch.py %*
pause
endlocal
