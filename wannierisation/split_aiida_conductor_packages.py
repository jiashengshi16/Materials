#!/usr/bin/env python3
"""Split exact AiiDA conductor packages into agent and reference folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_DIR = ROOT / "staging" / "aiida_conductor_exact"
DEFAULT_AGENT_DIR = ROOT / "input_packages" / "materials_for_agent"
DEFAULT_REFERENCE_DIR = ROOT / "input_packages" / "materials_reference_for_evaluation"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def clean_or_create(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def relative_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def update_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = read_json(path) if path.exists() else []
    by_material = {row["material"]: row for row in existing}
    for row in rows:
        by_material[row["material"]] = row
    write_json(path, [by_material[key] for key in sorted(by_material)])


def split_one(staging_dir: Path, agent_dir: Path, reference_dir: Path, material: str, overwrite: bool) -> dict[str, Any]:
    source = staging_dir / material
    if not source.exists():
        raise FileNotFoundError(source)
    source_metadata = read_json(source / "metadata.json")
    manifest = read_json(source / "wannier" / "manifest.json")
    reference_params = source_metadata["reference_params"]

    agent_material = agent_dir / material
    reference_material = reference_dir / material
    clean_or_create(agent_material, overwrite)
    clean_or_create(reference_material, overwrite)

    for rel in ("structure", "scf", "nscf"):
        copytree(source / rel, agent_material / rel)

    agent_metadata = {
        "material_id": material,
        "formula": material,
        "species": source_metadata["species"],
        "num_atoms": source_metadata["num_atoms"],
        "num_species": source_metadata["num_species"],
        "structure_files": source_metadata["structure_files"],
        "dft": {
            "num_bands": reference_params["num_bands"],
            "scf": {
                "input": "scf/input/scf.in",
                "output": "scf/output/aiida.out",
                "xml": "scf/output/data-file.xml",
            },
            "nscf": {
                "input": "nscf/input/nscf.in",
                "output": "nscf/output/aiida.out",
                "xml": "nscf/output/data-file.xml",
            },
        },
    }
    write_json(agent_material / "metadata.json", agent_metadata)

    copytree(source / "dft", reference_material / "dft")
    copytree(source / "wannier", reference_material / "wannier")

    reference_metadata = {
        "material": material,
        "purpose": "Evaluation-only reference artifacts withheld from materials_for_agent.",
        "agent_folder": display_path(agent_material),
        "reference_folder": display_path(reference_material),
        "exact_reference_params": reference_params,
        "source": source_metadata["source"],
        "archive_formula": source_metadata["archive_formula"],
        "wannier_manifest": manifest,
        "all_files": relative_files(reference_material),
    }
    write_json(reference_material / "reference_metadata.json", reference_metadata)

    return {
        "material": material,
        "status": "ok_aiida_conductor_agent_input",
        "num_bands": reference_params["num_bands"],
        "num_wann": reference_params["num_wann"],
        "archive_formula": source_metadata["archive_formula"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--materials", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = [
        split_one(args.staging_dir, args.agent_dir, args.reference_dir, material, args.overwrite)
        for material in args.materials
    ]
    update_summary(args.agent_dir / "summary.json", rows)
    print(f"Split {len(rows)} packages")
    for row in rows:
        print(
            f"{row['material']:8s} archive_formula={row['archive_formula']:8s} "
            f"num_wann={row['num_wann']:3d} num_bands={row['num_bands']:3d}"
        )


if __name__ == "__main__":
    main()
