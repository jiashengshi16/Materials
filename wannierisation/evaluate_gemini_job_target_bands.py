#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOTS = (
    ROOT / "input_packages/materials_200_reference_for_evaluation",
    ROOT / "input_packages/materials_reference_for_evaluation",
)
DEFAULT_AGENT_ROOTS = (
    ROOT / "input_packages/materials_200_for_agent",
    ROOT / "input_packages/materials_for_agent",
)
DEFAULT_OUT = ROOT / "evaluation/conductor_target_bands/gemini_job_results"


@dataclass(frozen=True)
class GeminiAttempt:
    trial_dir: Path
    attempt_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    material: str
    seed: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Gemini job DFT-vs-Wannier target-band overlays from jobs/* artifacts."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Gemini job root(s), trial dir(s), attempt dir(s), or run_manifest.json file(s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for PNG overlays, metrics JSON, and pointwise CSV files.",
    )
    parser.add_argument(
        "--reference-root",
        action="append",
        type=Path,
        default=[],
        help="Reference root containing MATERIAL/dft/offmesh/reference. Can be repeated.",
    )
    parser.add_argument(
        "--agent-root",
        action="append",
        type=Path,
        default=[],
        help="Agent input root containing MATERIAL/nscf/output/aiida.out. Can be repeated.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_manifest_paths(path: Path) -> list[Path]:
    if path.is_file() and path.name == "run_manifest.json":
        return [path]
    if path.is_dir() and path.name.startswith("attempt_") and (path / "run_manifest.json").exists():
        return [path / "run_manifest.json"]
    if path.is_dir() and (path / "artifacts").exists():
        found = sorted(path.glob("artifacts/attempt_*/run_manifest.json"))
        if found:
            return found
    if path.is_dir():
        return sorted(path.glob("*/artifacts/attempt_*/run_manifest.json"))
    raise FileNotFoundError(f"Could not find Gemini run_manifest.json under {path}")


def resolve_attempt(manifest_path: Path) -> GeminiAttempt:
    manifest = read_json(manifest_path)
    attempt_dir = manifest_path.parent
    trial_dir = attempt_dir.parents[1] if attempt_dir.parent.name == "artifacts" else attempt_dir.parent
    material = str(manifest.get("material_id") or manifest.get("material") or trial_dir.name.split("__", 1)[0])
    seed = str(manifest.get("seedname") or material)
    return GeminiAttempt(trial_dir, attempt_dir, manifest_path, manifest, material, seed)


def read_eig(path: Path) -> np.ndarray:
    rows: list[tuple[int, int, float]] = []
    max_band = max_kpoint = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        band_s, kpoint_s, energy_s = stripped.split()[:3]
        band = int(band_s)
        kpoint = int(kpoint_s)
        rows.append((band, kpoint, float(energy_s)))
        max_band = max(max_band, band)
        max_kpoint = max(max_kpoint, kpoint)
    eig = np.full((max_kpoint, max_band), np.nan, dtype=float)
    for band, kpoint, energy in rows:
        eig[kpoint - 1, band - 1] = energy
    if np.isnan(eig).any():
        raise ValueError(f"Missing eigenvalues in {path}")
    return eig


def read_win_kpoints(path: Path, expected: int | None = None) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = end = None
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped == "begin kpoints":
            start = idx + 1
        elif stripped == "end kpoints":
            end = idx
            break
    if start is None or end is None or end <= start:
        raise ValueError(f"Could not locate k-point block in {path}")
    kpoints = np.asarray(
        [[float(value) for value in line.split()[:3]] for line in lines[start:end] if line.strip()],
        dtype=float,
    )
    if expected is not None and kpoints.shape[0] != expected:
        raise ValueError(f"{path} has {kpoints.shape[0]} k-points, expected {expected}")
    return kpoints


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
                r_vectors[ir] = [int(parts[0]), int(parts[1]), int(parts[2])]
                m = int(parts[3]) - 1
                n = int(parts[4]) - 1
                hoppings[ir, m, n] = float(parts[5]) + 1j * float(parts[6])
    return r_vectors, hoppings / np.asarray(degeneracies, dtype=float)[:, None, None]


def read_wsvec(path: Path) -> dict[tuple[tuple[int, int, int], int, int], list[tuple[int, int, int]]]:
    if not path.exists():
        return {}
    wsvec: dict[tuple[tuple[int, int, int], int, int], list[tuple[int, int, int]]] = {}
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            r_vec = (int(parts[0]), int(parts[1]), int(parts[2]))
            m = int(parts[3]) - 1
            n = int(parts[4]) - 1
            count = int(next(handle).strip())
            wsvec[(r_vec, m, n)] = [tuple(int(v) for v in next(handle).split()[:3]) for _ in range(count)]
    return wsvec


def apply_wsvec(
    r_vectors: np.ndarray,
    hoppings: np.ndarray,
    wsvec: dict[tuple[tuple[int, int, int], int, int], list[tuple[int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    if not wsvec:
        return r_vectors, hoppings
    num_wann = hoppings.shape[1]
    terms: dict[tuple[int, int, int], np.ndarray] = {}
    for ir, r_array in enumerate(r_vectors):
        r_tuple = tuple(int(value) for value in r_array)
        for m in range(num_wann):
            for n in range(num_wann):
                amplitude = hoppings[ir, m, n]
                if amplitude == 0:
                    continue
                shifts = wsvec.get((r_tuple, m, n), [(0, 0, 0)])
                scaled = amplitude / len(shifts)
                for shift in shifts:
                    key = (r_tuple[0] + shift[0], r_tuple[1] + shift[1], r_tuple[2] + shift[2])
                    terms.setdefault(key, np.zeros((num_wann, num_wann), dtype=np.complex128))[m, n] += scaled
    new_r = np.asarray(list(terms), dtype=int)
    new_h = np.stack([terms[tuple(r)] for r in new_r])
    return new_r, new_h


def interpolate_bands(kpoints: np.ndarray, r_vectors: np.ndarray, hoppings: np.ndarray) -> np.ndarray:
    chunks = []
    for block in np.array_split(kpoints, max(1, len(kpoints) // 64 + 1)):
        phases = np.exp(2j * np.pi * (block @ r_vectors.T))
        h_of_k = np.einsum("kr,rmn->kmn", phases, hoppings, optimize=True)
        h_of_k = 0.5 * (h_of_k + np.conjugate(np.swapaxes(h_of_k, 1, 2)))
        chunks.append(np.linalg.eigvalsh(h_of_k))
    return np.vstack(chunks)


def parse_exclude_bands(value: Any) -> list[int]:
    if value in (None, "", False):
        return []
    if isinstance(value, list):
        return sorted({int(v) for v in value})
    bands: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if not token or token.lower() in {"none", "null"}:
            continue
        if "-" in token:
            start, end = (int(v.strip()) for v in token.split("-", 1))
            bands.update(range(start, end + 1))
        else:
            bands.add(int(token))
    return sorted(bands)


def target_range(manifest: dict[str, Any]) -> tuple[int, int]:
    start = manifest.get("target_dft_band_start")
    end = manifest.get("target_dft_band_end")
    if start is None or end is None:
        dft_ref = manifest.get("dft_reference", {})
        best = dft_ref.get("best_candidate_range", {})
        start = best.get("start") or dft_ref.get("target_dft_band_start")
        end = best.get("end") or dft_ref.get("target_dft_band_end")
    if start is None or end is None:
        raise ValueError("Could not determine target_dft_band_start/end from manifest")
    return int(start), int(end)


def find_number_key(value: Any, keys: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in keys and nested is not None:
                try:
                    return float(nested)
                except (TypeError, ValueError):
                    pass
        for nested in value.values():
            found = find_number_key(nested, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_number_key(nested, keys)
            if found is not None:
                return found
    return None


def parse_fermi_from_stdout(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    matches = re.findall(r"the Fermi energy is\s+([-+0-9.]+)\s+ev", text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def fermi_energy(attempt: GeminiAttempt, agent_roots: list[Path]) -> float:
    keys = {"fermi_energy_eV", "fermi_energy_ev", "fermi_energy", "fermi_ev"}
    found = find_number_key(attempt.manifest, keys)
    if found is not None:
        return found

    candidate_json = [
        attempt.trial_dir / "verifier/diagnostics.json",
        attempt.trial_dir / "verifier/reward.json",
        attempt.trial_dir / "result.json",
        attempt.trial_dir / "artifacts/report.json",
        attempt.trial_dir.parent / "result.json",
    ]
    for path in candidate_json:
        if path.exists():
            found = find_number_key(read_json(path), keys)
            if found is not None:
                return found

    for root in agent_roots:
        found = parse_fermi_from_stdout(root / attempt.material / "nscf/output/aiida.out")
        if found is not None:
            return found
    raise ValueError(f"Could not determine Fermi energy for {attempt.material}")


def reference_dir(material: str, reference_roots: list[Path]) -> Path:
    for root in reference_roots:
        candidate = root / material / "dft/offmesh/reference"
        if (candidate / "bands.npy").exists() and (candidate / "kpoints.npy").exists():
            return candidate
    roots = ", ".join(str(root) for root in reference_roots)
    raise FileNotFoundError(f"Could not find off-mesh reference data for {material} under {roots}")


def active_indices_for_target(manifest: dict[str, Any], eig_bands: int, start: int, end: int) -> list[int]:
    params = manifest.get("wannier_parameters", {})
    excluded = parse_exclude_bands(params.get("exclude_bands") or manifest.get("exclude_bands"))
    if not excluded:
        return list(range(start - 1, end))
    num_available = int(
        manifest.get("num_bands_available")
        or manifest.get("dft_reference", {}).get("num_bands_available")
        or eig_bands + len(excluded)
    )
    active_original = [band for band in range(1, num_available + 1) if band not in set(excluded)]
    original_to_active = {band: idx for idx, band in enumerate(active_original)}
    missing = [band for band in range(start, end + 1) if band not in original_to_active]
    if missing:
        raise ValueError(f"Target bands excluded or missing from active pool: {missing}")
    return [original_to_active[band] for band in range(start, end + 1)]


def metrics(dft: np.ndarray, wannier: np.ndarray, fermi: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    wannier_match = wannier[:, : dft.shape[1]]
    delta = wannier_match - dft
    below = dft <= fermi
    selected = delta[below]
    if selected.size == 0:
        raise ValueError("No below-Fermi DFT target points selected")
    abs_selected = np.abs(selected)
    occupied_delta = (wannier_match <= fermi).sum(axis=1) - below.sum(axis=1)
    return (
        {
            "num_kpoints": int(dft.shape[0]),
            "num_target_bands": int(dft.shape[1]),
            "num_below_fermi_points": int(selected.size),
            "below_fermi_fraction": float(selected.size / delta.size),
            "rmse_eV": float(np.sqrt(np.mean(selected**2))),
            "mae_eV": float(np.mean(abs_selected)),
            "max_abs_eV": float(np.max(abs_selected)),
            "p95_abs_eV": float(np.percentile(abs_selected, 95)),
            "mean_signed_eV": float(np.mean(selected)),
            "occupied_count_mismatch_kpoints": int(np.count_nonzero(occupied_delta)),
        },
        delta,
        below,
    )


def path_axis(kpoints: np.ndarray) -> np.ndarray:
    if len(kpoints) == 0:
        return np.asarray([], dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kpoints, axis=0), axis=1))])


def pyplot():
    cache_root = Path(tempfile.gettempdir()) / "wannierisation_benchmarking_cache"
    cache_root.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_overlay(
    path: Path,
    title: str,
    x_axis: np.ndarray,
    dft: np.ndarray,
    wannier: np.ndarray,
    fermi: float,
    stats: dict[str, Any],
    labels: list[str] | None = None,
    label_numbers: list[int] | None = None,
) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    dft_rel = dft - fermi
    wannier_rel = wannier[:, : dft.shape[1]] - fermi
    delta = wannier_rel - dft_rel
    below = dft_rel <= 0.0
    for band_idx in range(dft.shape[1]):
        axes[0].plot(
            x_axis,
            dft_rel[:, band_idx],
            color="0.18",
            linewidth=1.05,
            alpha=0.55,
            zorder=2,
            label="DFT target" if band_idx == 0 else None,
        )
        axes[0].plot(
            x_axis,
            wannier_rel[:, band_idx],
            color="#e66100",
            linewidth=1.0,
            alpha=1.0,
            linestyle=":",
            marker=None,
            zorder=5,
            label="Wannier" if band_idx == 0 else None,
        )
        selected = below[:, band_idx]
        axes[1].scatter(
            x_axis[selected],
            delta[selected, band_idx],
            s=8,
            alpha=0.5,
            linewidths=0,
        )
    axes[0].axhline(0.0, color="0.2", linewidth=0.9, alpha=0.9)
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    for axis in axes:
        axis.grid(True, alpha=0.22)
    if labels and label_numbers:
        ticks = []
        tick_labels = []
        for label, index in zip(labels, label_numbers):
            if 0 <= index < len(x_axis):
                ticks.append(x_axis[index])
                tick_labels.append(label.replace("GAMMA", r"$\Gamma$"))
        if ticks:
            axes[1].set_xticks(ticks)
            axes[1].set_xticklabels(tick_labels)
            for tick in ticks:
                for axis in axes:
                    axis.axvline(tick, color="0.7", linewidth=0.8, alpha=0.65)
    axes[0].set_ylabel("E - EF (eV)")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[1].set_ylabel("Wannier - DFT (eV)")
    axes[1].set_xlabel("k-path distance / k-point index")
    axes[1].set_title(
        "Below-EF target error | "
        f"RMSE={stats['rmse_eV']:.3g} eV, "
        f"MAE={stats['mae_eV']:.3g} eV, "
        f"Max={stats['max_abs_eV']:.3g} eV"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_pointwise_csv(
    path: Path,
    x_axis: np.ndarray,
    dft: np.ndarray,
    wannier: np.ndarray,
    below: np.ndarray,
    fermi: float,
    original_bands: list[int],
) -> None:
    lines = ["k_index,k_axis,original_dft_band,target_rank,dft_eV,wannier_eV,dft_minus_fermi_eV,wannier_minus_dft_eV,below_fermi"]
    wannier_match = wannier[:, : dft.shape[1]]
    delta = wannier_match - dft
    for k_idx in range(dft.shape[0]):
        for band_idx in range(dft.shape[1]):
            lines.append(
                f"{k_idx + 1},{x_axis[k_idx]:.12f},{original_bands[band_idx]},{band_idx + 1},"
                f"{dft[k_idx, band_idx]:.12f},{wannier_match[k_idx, band_idx]:.12f},"
                f"{dft[k_idx, band_idx] - fermi:.12f},{delta[k_idx, band_idx]:.12f},"
                f"{int(bool(below[k_idx, band_idx]))}"
            )
    path.write_text("\n".join(lines) + "\n")


def artifact_path(attempt: GeminiAttempt, manifest_key: str, fallback: str) -> Path:
    files = attempt.manifest.get("files", {})
    name = files.get(manifest_key) or fallback
    path = Path(name)
    return path if path.is_absolute() else attempt.attempt_dir / path


def evaluate_attempt(attempt: GeminiAttempt, output_root: Path, reference_roots: list[Path], agent_roots: list[Path]) -> dict[str, Any]:
    start, end = target_range(attempt.manifest)
    target_bands = list(range(start, end + 1))
    fermi = fermi_energy(attempt, agent_roots)

    eig_path = artifact_path(attempt, "eig", f"{attempt.seed}.eig")
    win_path = artifact_path(attempt, "win", f"{attempt.seed}.win")
    hr_path = artifact_path(attempt, "hr", f"{attempt.seed}_hr.dat")
    wsvec_path = attempt.attempt_dir / f"{attempt.seed}_wsvec.dat"

    dft_on_all = read_eig(eig_path)
    active_indices = active_indices_for_target(attempt.manifest, dft_on_all.shape[1], start, end)
    dft_on = dft_on_all[:, active_indices]
    k_on = read_win_kpoints(win_path, expected=dft_on_all.shape[0])
    r_vectors, hoppings = read_hr(hr_path)
    r_vectors, hoppings = apply_wsvec(r_vectors, hoppings, read_wsvec(wsvec_path))
    wannier_on = interpolate_bands(k_on, r_vectors, hoppings)
    on_stats, _, on_below = metrics(dft_on, wannier_on, fermi)

    ref_dir = reference_dir(attempt.material, reference_roots)
    dft_off_all = np.load(ref_dir / "bands.npy")
    k_off = np.load(ref_dir / "kpoints.npy")
    off_meta = read_json(ref_dir / "metadata.json") if (ref_dir / "metadata.json").exists() else {}
    dft_off = dft_off_all[:, start - 1 : end]
    wannier_off = interpolate_bands(k_off, r_vectors, hoppings)
    off_stats, _, off_below = metrics(dft_off, wannier_off, fermi)

    attempt_label = attempt.trial_dir.name
    out_dir = output_root / attempt.material / attempt_label
    on_dir = out_dir / "onmesh"
    off_dir = out_dir / "offmesh"
    on_dir.mkdir(parents=True, exist_ok=True)
    off_dir.mkdir(parents=True, exist_ok=True)

    on_axis = np.arange(1, len(k_on) + 1, dtype=float)
    off_axis = path_axis(k_off)
    plot_overlay(
        on_dir / f"{attempt.material}_gemini_onmesh_overlay.png",
        f"{attempt.material} Gemini on-mesh target bands {start}-{end}",
        on_axis,
        dft_on,
        wannier_on,
        fermi,
        on_stats,
    )
    plot_overlay(
        off_dir / f"{attempt.material}_gemini_offmesh_overlay.png",
        f"{attempt.material} Gemini off-mesh target bands {start}-{end}",
        off_axis,
        dft_off,
        wannier_off,
        fermi,
        off_stats,
        labels=off_meta.get("labels"),
        label_numbers=off_meta.get("label_numbers"),
    )
    write_pointwise_csv(on_dir / f"{attempt.material}_gemini_onmesh_pointwise.csv", on_axis, dft_on, wannier_on, on_below, fermi, target_bands)
    write_pointwise_csv(off_dir / f"{attempt.material}_gemini_offmesh_pointwise.csv", off_axis, dft_off, wannier_off, off_below, fermi, target_bands)

    summary = {
        "material": attempt.material,
        "model": "gemini",
        "trial_dir": str(attempt.trial_dir),
        "attempt_dir": str(attempt.attempt_dir),
        "manifest_path": str(attempt.manifest_path),
        "status": attempt.manifest.get("status"),
        "seedname": attempt.seed,
        "target_dft_band_start": start,
        "target_dft_band_end": end,
        "num_wann": attempt.manifest.get("num_wann"),
        "num_bands": attempt.manifest.get("num_bands"),
        "fermi_energy_eV": fermi,
        "reference_dir": str(ref_dir),
        "onmesh": on_stats,
        "offmesh": off_stats,
        "outputs": {
            "onmesh_overlay": str(on_dir / f"{attempt.material}_gemini_onmesh_overlay.png"),
            "offmesh_overlay": str(off_dir / f"{attempt.material}_gemini_offmesh_overlay.png"),
            "onmesh_pointwise": str(on_dir / f"{attempt.material}_gemini_onmesh_pointwise.csv"),
            "offmesh_pointwise": str(off_dir / f"{attempt.material}_gemini_offmesh_pointwise.csv"),
        },
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_combined_summary(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "combined_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Gemini Target-Band Evaluation",
        "",
        "| Material | Trial | Target Bands | On RMSE (eV) | Off RMSE (eV) | Off MAE (eV) | Off Max (eV) | Off Plot |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        trial = Path(item["trial_dir"]).name
        target = f"{item['target_dft_band_start']}-{item['target_dft_band_end']}"
        off_plot = Path(item["outputs"]["offmesh_overlay"]).name
        lines.append(
            f"| {item['material']} | {trial} | {target} | "
            f"{item['onmesh']['rmse_eV']:.6g} | {item['offmesh']['rmse_eV']:.6g} | "
            f"{item['offmesh']['mae_eV']:.6g} | {item['offmesh']['max_abs_eV']:.6g} | {off_plot} |"
        )
    lines.append("")
    lines.append("Each trial subdirectory contains on/off mesh overlays, pointwise CSV errors, and an evaluation_summary.json file.")
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    reference_roots = args.reference_root or list(DEFAULT_REFERENCE_ROOTS)
    agent_roots = args.agent_root or list(DEFAULT_AGENT_ROOTS)
    manifest_paths: list[Path] = []
    for path in args.paths:
        manifest_paths.extend(find_manifest_paths(path))
    attempts = [resolve_attempt(path) for path in dict.fromkeys(manifest_paths)]

    summaries = []
    for attempt in attempts:
        print(f"Evaluating {attempt.material}: {attempt.trial_dir}")
        summaries.append(evaluate_attempt(attempt, args.output_dir, reference_roots, agent_roots))
    write_combined_summary(args.output_dir, summaries)
    print(f"Wrote Gemini evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
