#!/usr/bin/env python3
"""Sync the Harbor grading note and JSON contracts across instructions.

Dry-run by default. Use --apply to modify files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SYNC_HEADING = "## Harbor grading note"


def extract_synced_tail(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    start = text.find(SYNC_HEADING)
    if start == -1:
        raise SystemExit(f"missing {SYNC_HEADING!r} in {source}")
    return text[start:].strip() + "\n"


def instruction_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("*/instruction.md"))


def sync_tail(path: Path, synced_tail: str) -> bool:
    text = path.read_text(encoding="utf-8")
    start = text.find(SYNC_HEADING)
    if start == -1:
        suffix = "\n" if text.endswith("\n") else "\n\n"
        new_text = text + suffix + synced_tail
    else:
        new_text = text[:start].rstrip() + "\n\n" + synced_tail

    if new_text == text:
        return False

    path.write_text(new_text, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the W instruction.md Harbor grading note and Required JSON "
            "contracts block to wannier_200 material instructions."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="harbor_datasets/wannier_200",
        type=Path,
        help="Dataset directory containing per-material instruction.md files.",
    )
    parser.add_argument(
        "--source",
        default=Path("harbor_datasets/wannier_200/W/instruction.md"),
        type=Path,
        help="Instruction file to copy the synced tail from.",
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        help="Limit to one material name. Can be passed more than once.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this, only prints what would change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synced_tail = extract_synced_tail(args.source)
    wanted = set(args.material)
    paths = instruction_files(args.dataset_dir)
    if wanted:
        paths = [path for path in paths if path.parent.name in wanted]

    changed: list[Path] = []
    unchanged: list[Path] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        start = text.find(SYNC_HEADING)
        current_tail = text[start:].strip() + "\n" if start != -1 else ""
        if current_tail == synced_tail:
            unchanged.append(path)
        else:
            changed.append(path)

    action = "updated" if args.apply else "would update"
    for path in changed:
        if args.apply:
            sync_tail(path, synced_tail)
        print(f"{action}: {path}")

    print(
        f"summary: {len(unchanged)} already synced, "
        f"{len(changed)} {'updated' if args.apply else 'would change'}"
    )


if __name__ == "__main__":
    main()
