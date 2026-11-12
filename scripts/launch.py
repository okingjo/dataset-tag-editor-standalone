import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Keep downloaded models and temporary files inside the repository so the app
# runs fully self-contained (no writes to the user's home directory / C: drive).
# setdefault respects any value the user has already configured externally.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_DIR = _REPO_ROOT / "runtime"
os.environ.setdefault("HF_HOME", str(_RUNTIME_DIR / "hf_cache"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(_RUNTIME_DIR / "gradio_tmp"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cmd_args, logger


# ================================================================
# brought from AUTOMATIC1111/stable-diffusion-webui and modified



def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and minor >= 9):
        logger.error(
            f"""
INCOMPATIBLE PYTHON VERSION

This program is aimed to work on Python >=3.9 (developed with 3.10.11), but you have {major}.{minor}.{micro}.
"""
        )


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run(command):
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ,
    )


# ================================================================


def prepare_environment():
    # Dependencies are managed by uv (pyproject.toml / uv.lock). Nothing is
    # pip-installed at runtime: if torch is missing, the environment was not
    # created by uv, so fail with guidance instead of installing packages.
    if not is_installed("torch") or not is_installed("torchvision"):
        logger.error(
            "PyTorch is not installed. Start the app with launch_uv.bat / webui_uv.sh "
            "(or run `uv sync` first) so the pinned dependencies are installed into .venv."
        )
        raise SystemExit(1)
    check_python_version()

    import devices

    logger.write(f"PyTorch device: {devices.device}")


if __name__ == "__main__":
    prepare_environment()

    import interface

    interface.main()
