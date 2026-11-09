#!/usr/bin/env python3
"""Clone and set up the complete Haltere workspace from one command.

Example:
  python scripts/bootstrap.py \
    --repo https://github.com/YOUR_ACCOUNT/haltere.git \
    --hardware cpu
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone and set up the Haltere shared workspace.")
    parser.add_argument("--repo", required=True, help="URL of your Haltere fork")
    parser.add_argument("--branch", default="feature/shareable-workspace",
                        help="Haltere branch to clone")
    parser.add_argument("--directory", default="haltere-workspace",
                        help="directory to create for the workspace")
    parser.add_argument("--hardware", choices=["auto", "cpu", "cuda", "rocm"], default="auto",
                        help="PyTorch backend to install")
    parser.add_argument("--skip-graphics", action="store_true",
                        help="set up Haltere without downloading GraphicsEngine")
    args = parser.parse_args()

    target = Path(args.directory).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Target already exists and is not empty: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--branch", args.branch, args.repo, str(target)])

    command = [sys.executable, "scripts/setup.py", "--hardware", args.hardware]
    if args.skip_graphics:
        command.append("--skip-graphics")
    run(command, cwd=target)

    print()
    print(f"Workspace ready at: {target}")
    if os.name == "nt":
        print(f"Activate with: {target / '.venv' / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"Activate with: source {target / '.venv' / 'bin' / 'activate'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
