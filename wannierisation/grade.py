#!/usr/bin/env python3
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
