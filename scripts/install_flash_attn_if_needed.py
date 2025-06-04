# Standard
import importlib.util
import subprocess
import sys


def flash_attn_is_installed() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


if flash_attn_is_installed():
    print("flash-attn already installed, skipping.")
else:
    print("flash-attn not found in environment, installing...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-cuda.txt",
            "-c",
            "constraints-dev.txt",
            "--no-build-isolation",
        ]
    )
