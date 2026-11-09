#!/usr/bin/env python3
"""Cross-platform setup for the Haltere + synthetic-camera workspace.

Examples:
  python scripts/setup.py --hardware cpu
  python scripts/setup.py --hardware cuda
  python scripts/setup.py --hardware rocm

The script intentionally uses Python's standard venv module instead of relying
on PowerShell, Bash, uv, or a platform-specific package manager.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".venv"
GRAPHICS_LOCK = ROOT / "integrations" / "graphics_engine.json"


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def environment_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def install_torch(python: Path, hardware: str, index_override: str | None) -> None:
    if hardware == "auto":
        hardware = "cpu"
        print("Hardware auto-detection defaults to CPU. Use --hardware cuda or --hardware rocm explicitly.")
    if hardware == "rocm" and os.name == "nt":
        raise SystemExit("The official PyTorch ROCm wheels are for Linux; use --hardware cpu on Windows.")

    indexes = {
        "cpu": "https://download.pytorch.org/whl/cpu",
        "cuda": "https://download.pytorch.org/whl/cu128",
        "rocm": "https://download.pytorch.org/whl/rocm6.3",
    }
    index_url = index_override or indexes[hardware]
    run([str(python), "-m", "pip", "install", "torch", "--index-url", index_url])


def prepare_graphics() -> None:
    if not GRAPHICS_LOCK.exists():
        print("Graphics lock file not found; skipping GraphicsEngine checkout.")
        return
    lock = json.loads(GRAPHICS_LOCK.read_text(encoding="utf-8"))
    target = ROOT / lock["directory"]
    if (target / ".git").exists():
        run(["git", "fetch", "origin", lock["branch"]], cwd=target)
    elif target.exists() and any(target.iterdir()):
        raise SystemExit(f"Graphics target is not empty and is not a Git checkout: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--branch", lock["branch"], lock["repository"], str(target)])
    run(["git", "checkout", "--detach", lock["ref"]], cwd=target)
    print(f"GraphicsEngine checked out at {lock['ref']} in {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up Haltere and its synthetic-camera renderer.")
    parser.add_argument("--hardware", choices=["auto", "cpu", "cuda", "rocm"], default="auto",
                        help="PyTorch backend: cpu, NVIDIA CUDA, Linux AMD ROCm, or safe CPU default")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to create the virtual environment")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help="virtual-environment directory")
    parser.add_argument("--torch-index-url", default=None,
                        help="override the PyTorch wheel index for a local/custom build")
    parser.add_argument("--skip-graphics", action="store_true", help="do not download GraphicsEngine")
    args = parser.parse_args()

    env_dir = Path(args.env).expanduser().resolve()
    python = environment_python(env_dir)
    if not python.exists():
        print(f"Creating virtual environment at {env_dir}")
        venv.EnvBuilder(with_pip=True, clear=False).create(env_dir)

    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    install_torch(python, args.hardware, args.torch_index_url)
    run([str(python), "-m", "pip", "install", "-e", ".[dev,vision]"])

    if not args.skip_graphics:
        prepare_graphics()

    run([str(python), "-c", "import torch; print('PyTorch', torch.__version__, '| CUDA/ROCm available:', torch.cuda.is_available())"])
    print()
    print("Setup complete.")
    if os.name == "nt":
        print(f"Activate with: {env_dir / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"Activate with: source {env_dir / 'bin' / 'activate'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
