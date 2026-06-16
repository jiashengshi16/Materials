#!/usr/bin/env python3
"""Create a local Harbor dataset for the 200 Materials Cloud Wannier tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_DIR = ROOT / "input_packages" / "materials_200_for_agent"
DEFAULT_REFERENCE_DIR = ROOT / "input_packages" / "materials_200_reference_for_evaluation"
DEFAULT_PROMPT_DIR = ROOT / "codex_experiments" / "conductors_with_num_wann"
DEFAULT_OUTPUT_DIR = ROOT / "harbor_datasets" / "wannier_200"
DEFAULT_FERMI = ROOT / "200materials" / "fermi_energies.json"


DOCKERFILE = """FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \\
    bash \\
    build-essential \\
    ca-certificates \\
    coreutils \\
    curl \\
    findutils \\
    gawk \\
    git \\
    grep \\
    python3 \\
    python3-numpy \\
    python3-pip \\
    python3-venv \\
    quantum-espresso \\
    sed \\
    wannier90 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY material /app/material
COPY README.md /app/README.md
"""


TEST_SH = """#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
python3 /tests/grade.py
"""


GRADE_PY = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np


APP = Path("/app")
REF = Path("/tests/reference")
LOGS = Path("/logs/verifier")


def write_reward(data: dict[str, Any]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data.setdefault("reward", 0.0)
    numeric_rewards = {
        key: value
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    numeric_rewards.setdefault("reward", 0.0)
    (LOGS / "reward.json").write_text(json.dumps(numeric_rewards, indent=2) + "\n", encoding="utf-8")
    (LOGS / "diagnostics.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_manifest() -> tuple[Path | None, dict[str, Any]]:
    report_path = APP / "report.json"
    if report_path.exists():
        try:
            report = read_json(report_path)
            manifest_path = report.get("run_manifest_path")
            if manifest_path:
                path = Path(manifest_path)
                if not path.is_absolute():
                    path = APP / path
                if path.exists():
                    return path, read_json(path)
        except Exception:
            pass

    manifests = sorted((APP / "artifacts").glob("attempt_*/run_manifest.json"))
    for path in reversed(manifests):
        try:
            return path, read_json(path)
        except Exception:
            continue
    return None, {}


def read_hr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as handle:
        next(handle)
        num_wann = int(next(handle).strip())
        nrpts = int(next(handle).strip())
        degeneracies: list[int] = []
        while len(degeneracies) < nrpts:
            degeneracies.extend(int(value) for value in next(handle).split())
        r_vectors = np.zeros((nrpts, 3), dtype=int)
        hoppings = np.zeros((nrpts, num_wann, num_wann), dtype=np.complex128)
        for ir in range(nrpts):
            for _ in range(num_wann * num_wann):
                parts = next(handle).split()
                rx, ry, rz = (int(parts[0]), int(parts[1]), int(parts[2]))
                m = int(parts[3]) - 1
                n = int(parts[4]) - 1
                real = float(parts[5])
                imag = float(parts[6])
                r_vectors[ir] = (rx, ry, rz)
                hoppings[ir, m, n] = real + 1j * imag
    return r_vectors, hoppings / np.asarray(degeneracies, dtype=float)[:, None, None]


def interpolate_bands(kpoints: np.ndarray, r_vectors: np.ndarray, hoppings: np.ndarray) -> np.ndarray:
    phases = np.exp(2j * np.pi * (kpoints @ r_vectors.T))
    h_of_k = np.einsum("kr,rmn->kmn", phases, hoppings, optimize=True)
    h_of_k = 0.5 * (h_of_k + np.conjugate(np.swapaxes(h_of_k, 1, 2)))
    return np.linalg.eigvalsh(h_of_k)


def target_range(manifest: dict[str, Any], ref_meta: dict[str, Any]) -> tuple[int, int]:
    start = manifest.get("target_dft_band_start")
    end = manifest.get("target_dft_band_end")
    if start is not None and end is not None:
        return int(start), int(end)
    dft = manifest.get("dft_reference", {})
    start = dft.get("target_dft_band_start")
    end = dft.get("target_dft_band_end")
    if start is not None and end is not None:
        return int(start), int(end)
    params = ref_meta["exact_reference_params"]
    return 1, int(params["num_wann"])


def locate_hr(manifest_path: Path, manifest: dict[str, Any]) -> Path | None:
    attempt_dir = manifest_path.parent
    seed = manifest.get("seedname") or manifest.get("material_id")
    files = manifest.get("files", {})
    candidates: list[Path] = []
    if isinstance(files, dict) and files.get("hr"):
        candidates.append(attempt_dir / str(files["hr"]))
    if seed:
        candidates.append(attempt_dir / f"{seed}_hr.dat")
    candidates.extend(sorted(attempt_dir.glob("*_hr.dat")))
    candidates.extend(sorted((APP / "artifacts").glob("attempt_*/*_hr.dat")))
    for path in candidates:
        if path.exists():
            return path
    return None


def manifest_schema_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "material_id",
        "seedname",
        "attempt",
        "status",
        "executed_successfully",
        "target_dft_band_start",
        "target_dft_band_end",
        "num_wann",
        "num_bands",
    ]
    for key in required:
        if key not in manifest:
            errors.append(f"missing {key}")
    status = str(manifest.get("status", "")).lower()
    if status and status not in {"success", "partial", "failed"}:
        errors.append("status must be success, partial, or failed")
    if "executed_successfully" in manifest and not isinstance(manifest["executed_successfully"], bool):
        errors.append("executed_successfully must be boolean")
    return errors


def score_from_rmse(rmse: float, completed: bool) -> float:
    if not completed or not math.isfinite(rmse):
        return 0.0
    return max(0.0, min(1.0, 1.0 - rmse))


def main() -> None:
    try:
        ref_meta = read_json(REF / "reference_metadata.json")
        grading_meta = read_json(REF / "grading_metadata.json")
        manifest_path, manifest = find_manifest()
        if manifest_path is None:
            write_reward({
                "reward": 0.0,
                "status": "failed",
                "error": "No artifacts/attempt_*/run_manifest.json found",
            })
            return

        hr_path = locate_hr(manifest_path, manifest)
        if hr_path is None:
            write_reward({
                "reward": 0.0,
                "status": "failed",
                "manifest_path": str(manifest_path),
                "error": "No *_hr.dat file found in final artifacts",
            })
            return

        schema_errors = manifest_schema_errors(manifest)
        start, end = target_range(manifest, ref_meta)
        fermi = float(grading_meta["fermi_energy_eV"])
        off_ref = REF / "dft" / "offmesh" / "reference"
        dft_bands = np.load(off_ref / "bands.npy")[:, start - 1 : end]
        kpoints = np.load(off_ref / "kpoints.npy")
        r_vectors, hoppings = read_hr(hr_path)
        wannier_bands = interpolate_bands(kpoints, r_vectors, hoppings)[:, : dft_bands.shape[1]]
        below_mask = dft_bands <= fermi
        if not np.any(below_mask):
            below_mask = np.ones_like(dft_bands, dtype=bool)
        delta = wannier_bands - dft_bands
        below_delta = delta[below_mask]
        abs_delta = np.abs(below_delta)
        rmse = float(np.sqrt(np.mean(below_delta**2)))
        mae = float(np.mean(abs_delta))
        max_abs = float(np.max(abs_delta))
        p95_abs = float(np.percentile(abs_delta, 95))
        completed = not schema_errors and str(manifest.get("status", "")).lower() == "success"
        reward = score_from_rmse(rmse, completed)
        write_reward({
            "reward": reward,
            "status": "success" if completed else "partial",
            "executed_successfully": 1.0 if completed else 0.0,
            "material": ref_meta["material"],
            "manifest_path": str(manifest_path),
            "manifest_schema_errors": schema_errors,
            "hr_path": str(hr_path),
            "target_dft_band_start": start,
            "target_dft_band_end": end,
            "num_target_bands": end - start + 1,
            "fermi_energy_eV": fermi,
            "num_offmesh_kpoints": int(dft_bands.shape[0]),
            "num_below_fermi_points": int(np.count_nonzero(below_mask)),
            "rmse_eV": rmse,
            "mae_eV": mae,
            "max_abs_eV": max_abs,
            "p95_abs_eV": p95_abs,
        })
    except Exception as exc:
        write_reward({
            "reward": 0.0,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


if __name__ == "__main__":
    main()
'''


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_or_create(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def write_text(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def task_toml(material: str) -> str:
    return f'''schema_version = "1.3"

[task]
name = "wannier_200/{material}"
description = "Materials Cloud with-conduction Wannierisation task for {material}"
keywords = ["materials-cloud", "wannierisation", "with-conduction"]

[agent]
timeout_sec = 7200
user = "root"

[verifier]
timeout_sec = 900
user = "root"

[environment]
network_mode = "public"
cpus = 8
memory_mb = 32768
storage_mb = 20480
'''


def task_readme(material: str) -> str:
    return f"""# {material} Wannierisation Harbor Task

The agent-visible material package is available at `/app/material`.
Write final workflow files under `/app/workflow`, report files under `/app`,
and final Wannier artifacts under `/app/artifacts/attempt_<N>/`.

The hidden verifier reads `/tests/reference` after the agent finishes.
"""


def instruction_text(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").rstrip() + """

## Harbor grading note

The hidden verifier compares your final Wannier Hamiltonian against withheld
DFT reference bands. The best/final attempt must include `<seed>_hr.dat` in
`artifacts/attempt_<N>/`; without that file the run receives reward 0 because
there is no Hamiltonian to evaluate.

## Required JSON contracts

The best/final attempt's `artifacts/attempt_<N>/run_manifest.json` must be
valid JSON and must include these top-level fields. If the workflow completed
and produced the final Wannier Hamiltonian, set `status` to `"success"` and
`executed_successfully` to `true`.

```json
{
  "material_id": "<material>",
  "seedname": "<seed>",
  "attempt": "attempt_1",
  "status": "success | partial | failed",
  "executed_successfully": true,
  "workflow_entrypoint": "workflow/run.sh",
  "target_dft_band_start": 1,
  "target_dft_band_end": null,
  "num_wann": null,
  "num_bands": null,
  "projections": [],
  "wannier_parameters": {
    "num_bands": null,
    "num_wann": null,
    "dis_win_min": null,
    "dis_win_max": null,
    "dis_froz_min": null,
    "dis_froz_max": null
  },
  "files": {
    "win": "<seed>.win",
    "wout": "<seed>.wout",
    "eig": "<seed>.eig",
    "hr": "<seed>_hr.dat",
    "chk": "<seed>.chk",
    "nnkp": "<seed>.nnkp"
  },
  "commands": [],
  "produced_files": [],
  "missing_files": [],
  "notes": []
}
```

The root `report.json` must also include at least:

```json
{
  "material_id": "<material>",
  "status": "success | partial | failed",
  "executed_successfully": true,
  "attempt_count": 1,
  "runtime_notes": [],
  "key_outputs": [],
  "key_workflow_files": [],
  "blockers": [],
  "run_manifest_path": "artifacts/attempt_1/run_manifest.json",
  "artifact_root": "artifacts/attempt_1",
  "target_dft_band_start": 1,
  "target_dft_band_end": null,
  "num_wann": null,
  "num_bands": null
}
```

Your final response must be a concise JSON object with the same top-level
status/path fields as `report.json`. Do not omit `status` or
`executed_successfully`.
"""


def grading_metadata(material: str, reference_meta: dict[str, Any], fermi_energies: dict[str, float]) -> dict[str, Any]:
    if material not in fermi_energies:
        raise KeyError(f"Missing Fermi energy for {material}")
    params = reference_meta["exact_reference_params"]
    return {
        "material": material,
        "fermi_energy_eV": float(fermi_energies[material]),
        "target_dft_band_start": 1,
        "target_dft_band_end": int(params["num_wann"]),
        "num_wann": int(params["num_wann"]),
        "num_bands": int(params["num_bands"]),
        "scoring": "Off-mesh DFT-vs-Wannier RMSE/MAE below Fermi level for DFT bands 1-num_wann.",
    }


def material_ids(agent_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.name for path in agent_dir.iterdir() if path.is_dir())


def create_task(
    material: str,
    output_dir: Path,
    agent_dir: Path,
    reference_dir: Path,
    prompt_dir: Path,
    fermi_energies: dict[str, float],
    overwrite: bool,
) -> None:
    task_dir = output_dir / material
    clean_or_create(task_dir, overwrite)

    agent_material = agent_dir / material
    reference_material = reference_dir / material
    prompt_path = prompt_dir / material / "prompt.md"
    reference_meta = read_json(reference_material / "reference_metadata.json")

    write_text(task_dir / "instruction.md", instruction_text(prompt_path))
    write_text(task_dir / "task.toml", task_toml(material))
    write_text(task_dir / "environment" / "Dockerfile", DOCKERFILE)
    write_text(task_dir / "environment" / "README.md", task_readme(material))
    copytree(agent_material, task_dir / "environment" / "material")

    write_text(task_dir / "tests" / "test.sh", TEST_SH, mode=0o755)
    write_text(task_dir / "tests" / "grade.py", GRADE_PY, mode=0o755)
    copytree(reference_material, task_dir / "tests" / "reference")
    write_text(
        task_dir / "tests" / "reference" / "grading_metadata.json",
        json.dumps(grading_metadata(material, reference_meta, fermi_energies), indent=2) + "\n",
    )


def write_dataset_readme(output_dir: Path, materials: list[str]) -> None:
    text = f"""# Wannier 200 Harbor Dataset

Generated local Harbor dataset for the 200 Materials Cloud `with_conduction`
Wannierisation tasks.

Task count: {len(materials)}

Run a smoke test first:

```bash
harbor run -p harbor_datasets/wannier_200/W -a gemini-cli -m <gemini-model>
```

Then run the full dataset:

```bash
harbor run -p harbor_datasets/wannier_200 -a gemini-cli -m <gemini-model> -n 8
```
"""
    write_text(output_dir / "README.md", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fermi-energies", type=Path, default=DEFAULT_FERMI)
    parser.add_argument("--materials", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fermi_energies = read_json(args.fermi_energies)
    materials = material_ids(args.agent_dir, args.materials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for material in materials:
        create_task(
            material,
            args.output_dir,
            args.agent_dir,
            args.reference_dir,
            args.prompt_dir,
            fermi_energies,
            args.overwrite,
        )
    write_dataset_readme(args.output_dir, materials)
    print(f"Wrote {len(materials)} Harbor tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
