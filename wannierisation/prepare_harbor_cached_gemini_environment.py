#!/usr/bin/env python3
"""Prepare Harbor Wannier tasks to avoid per-trial apt/npm setup."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
COMBINED_IMAGE = "wannier-qe-gemini-base:0.46.0"
GEMINI_IMAGE = "harbor-gemini-agent-base:0.46.0"
QE_IMAGE = "wannier-qe-local:latest"


EXPECTED_DOCKERFILE = """FROM wannier-qe-gemini-base:0.46.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY material /app/material
COPY .geminiignore /app/.geminiignore
COPY safe_cat /usr/local/bin/cat
COPY safe_ls /usr/local/bin/ls
COPY safe_grep /usr/local/bin/grep
COPY safe_sed /usr/local/bin/sed
COPY safe_awk /usr/local/bin/awk
COPY safe_head /usr/local/bin/head
COPY safe_tail /usr/local/bin/tail
COPY safe_rg /usr/local/bin/rg
RUN chmod +x /usr/local/bin/cat /usr/local/bin/ls /usr/local/bin/grep /usr/local/bin/sed /usr/local/bin/awk /usr/local/bin/head /usr/local/bin/tail /usr/local/bin/rg
COPY README.md /app/README.md
"""


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(command, check=check, text=True)


def image_exists(name: str) -> bool:
    return run(["docker", "image", "inspect", name], check=False).returncode == 0


def require_image(name: str, message: str) -> None:
    if not image_exists(name):
        raise SystemExit(message)


def patch_dockerfiles(dataset: Path) -> int:
    dockerfiles = sorted(dataset.glob("*/environment/Dockerfile"))
    if not dockerfiles:
        raise SystemExit(f"No task Dockerfiles found under {dataset}")
    changed = 0
    for dockerfile in dockerfiles:
        old = dockerfile.read_text(encoding="utf-8")
        if old != EXPECTED_DOCKERFILE:
            dockerfile.write_text(EXPECTED_DOCKERFILE, encoding="utf-8")
            changed += 1
    return changed


def verify_image() -> None:
    run(
        [
            "docker",
            "run",
            "--rm",
            COMBINED_IMAGE,
            "sh",
            "-lc",
            "gemini --version && which pw.x && which pw2wannier90.x && which wannier90.x && which rg && which git",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    require_image(
        QE_IMAGE,
        f"Missing {QE_IMAGE}. Build or restore the local QE/Wannier base image first.",
    )
    require_image(
        GEMINI_IMAGE,
        f"Missing {GEMINI_IMAGE}. Build it with: docker build -f docker/gemini-agent-base.Dockerfile -t {GEMINI_IMAGE} .",
    )

    if not args.skip_build:
        run(
            [
                "docker",
                "build",
                "-f",
                "docker/wannier-qe-gemini-base.Dockerfile",
                "-t",
                COMBINED_IMAGE,
                ".",
            ]
        )

    verify_image()
    changed = patch_dockerfiles(args.dataset)
    print(f"Patched {changed} Dockerfile(s); {len(list(args.dataset.glob('*/environment/Dockerfile')))} total checked.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
