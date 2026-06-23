#!/usr/bin/env python3
"""Batch forensic analysis of Gemini Wannierisation failures.

The output separates three questions:

1. WHERE is the interpolation wrong (band, k-point, and energy region)?
2. HOW did Gemini configure the calculation differently from the reference?
3. Is there evidence for poor/non-converged localization, or did a localized
   solution still produce poor interpolation (suggesting subspace selection)?

The classifications are evidence buckets, not claims of causality.  They are
designed to select cohorts and concrete cases for controlled reruns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_gemini_error_scalars import (
    final_spreads,
    gemini_artifact,
    interpolate_bands,
    read_hr,
    trial_dir,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "jobs" / "num_wann_ordered_diagnostics_summary.json"
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
DEFAULT_OUTPUT = ROOT / "jobs" / "gemini_failure_modes"

ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z][\w]*)\s*=\s*(.*?)\s*$")
WF_RE = re.compile(
    r"WF centre and spread\s+\d+\s+\([^)]*\)\s+([-+0-9.Ee]+)", re.I
)
CONV_RE = re.compile(
    r"^\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+"
    r"([-+0-9.Ee]+).*?<--\s*CONV\s*$",
    re.M,
)
FINAL_OMEGA_I_RE = re.compile(r"Final Omega_I\s+([-+0-9.Ee]+)", re.I)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_scalar(value: str) -> Any:
    clean = value.strip().strip("'").strip('"')
    low = clean.lower()
    if low in {".true.", "true", "t"}:
        return True
    if low in {".false.", "false", "f"}:
        return False
    try:
        return float(clean) if any(c in clean for c in ".eEdD") else int(clean)
    except ValueError:
        return clean


def parse_win(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parameters: dict[str, Any] = {}
    in_block = False
    projection_lines: list[str] = []
    in_projections = False
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("begin "):
            in_block = True
            in_projections = low == "begin projections"
            continue
        if low.startswith("end "):
            in_block = False
            in_projections = False
            continue
        if in_projections:
            projection_lines.append(line)
        if in_block:
            continue
        match = ASSIGNMENT_RE.match(line)
        if match:
            parameters[match.group(1).lower()] = parse_scalar(match.group(2))

    if parameters.get("scdm_proj") is True:
        mode = "scdm"
    elif any(line.lower() == "random" for line in projection_lines):
        mode = "random"
    elif projection_lines:
        mode = "explicit"
    else:
        mode = "none/default"
    return {"parameters": parameters, "projection_mode": mode, "projections": projection_lines}


def parse_wout(path: Path, configured_num_iter: int | None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    final = text.rsplit("Final State", 1)[-1]
    wf_spreads = [float(value) for value in WF_RE.findall(final)]
    conv = CONV_RE.findall(text)
    last = conv[-1] if conv else None
    initial_spread = float(conv[0][3]) if conv else None
    final_iteration = int(last[0]) if last else None
    final_gradient = float(last[2]) if last else None
    final_total = float(last[3]) if last else None
    spread_parts = final_spreads(path)
    if spread_parts["total"] is not None:
        final_total = spread_parts["total"]
    omega_matches = FINAL_OMEGA_I_RE.findall(text)
    hit_cap = bool(
        configured_num_iter is not None
        and final_iteration is not None
        and final_iteration >= configured_num_iter
    )
    if "disentanglement convergence criteria satisfied" in lower:
        dis_status = "converged"
    elif "maximum number of disentanglement iterations reached" in lower:
        dis_status = "iteration_limit"
    elif "using band disentanglement" in lower:
        dis_status = "ran/status_unknown"
    else:
        dis_status = "not_used"
    return {
        "omega_I_A2": spread_parts["I"],
        "omega_D_A2": spread_parts["D"],
        "omega_OD_A2": spread_parts["OD"],
        "omega_total_A2": final_total,
        "final_disentanglement_omega_I_A2": float(omega_matches[-1]) if omega_matches else None,
        "wf_spread_max_A2": max(wf_spreads) if wf_spreads else None,
        "wf_spread_median_A2": float(np.median(wf_spreads)) if wf_spreads else None,
        "wf_spread_p90_A2": float(np.quantile(wf_spreads, 0.9)) if wf_spreads else None,
        "initial_spread_A2": initial_spread,
        "localization_final_iteration": final_iteration,
        "localization_final_gradient": final_gradient,
        "localization_hit_iteration_cap": hit_cap,
        "disentanglement_status": dis_status,
        "warning_count": sum("warning" in line.lower() for line in text.splitlines()),
    }


def ratio(a: Any, b: Any) -> float | None:
    return float(a) / float(b) if finite(a) and finite(b) and b != 0 else None


def rmse(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def prefixed(target: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    target.update({f"{prefix}_{key}": value for key, value in values.items()})


def analyse(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    material = str(record["material"])
    trial = trial_dir(record, args.jobs_root)
    gemini_win_path = gemini_artifact(trial, ".win")
    gemini_wout_path = gemini_artifact(trial, ".wout")
    hr_path = gemini_artifact(trial, "_hr.dat")
    reference = args.dataset / material / "tests" / "reference" / "wannier"
    reference_win_path = reference / "input" / "wannier90.win"
    reference_wout_path = reference / "output" / "wannier90" / "aiida.wout"
    offmesh = args.dataset / material / "tests" / "reference" / "dft" / "offmesh" / "reference"

    gw = parse_win(gemini_win_path)
    rw = parse_win(reference_win_path)
    gp, rp = gw["parameters"], rw["parameters"]
    go = parse_wout(gemini_wout_path, int(gp["num_iter"]) if finite(gp.get("num_iter")) else None)
    ro = parse_wout(reference_wout_path, int(rp["num_iter"]) if finite(rp.get("num_iter")) else None)

    start, end = int(record["target_dft_band_start"]), int(record["target_dft_band_end"])
    fermi = float(record["fermi_energy_eV"])
    dft = np.load(offmesh / "bands.npy")[:, start - 1 : end]
    kpoints = np.load(offmesh / "kpoints.npy")
    r_vectors, hoppings = read_hr(hr_path)
    wannier = interpolate_bands(kpoints, r_vectors, hoppings)[:, : dft.shape[1]]
    delta = wannier - dft
    occupied = dft <= fermi
    abs_occ = np.abs(delta[occupied])
    eligible = np.where(occupied, np.abs(delta), -np.inf)
    worst_k, worst_band = np.unravel_index(int(np.argmax(eligible)), eligible.shape)
    band_scores = np.array([rmse(delta[:, i][occupied[:, i]]) or -1 for i in range(delta.shape[1])])
    k_scores = np.array([rmse(delta[i][occupied[i]]) or -1 for i in range(delta.shape[0])])
    worst_rmse_band = int(np.argmax(band_scores))
    worst_rmse_k = int(np.argmax(k_scores))

    result: dict[str, Any] = {
        "material": material,
        "job_folder": record.get("job_folder"),
        "trial_folder": record.get("trial_folder"),
        "gemini_to_reference_rmse_ratio": record.get("gemini_to_reference_rmse_ratio"),
        "gemini_rmse_eV": record.get("rmse_eV"),
        "reference_rmse_eV": record.get("reference_offmesh_rmse_eV"),
        "num_wann": dft.shape[1],
        "num_offmesh_kpoints": dft.shape[0],
        "worst_abs_error_eV": float(eligible[worst_k, worst_band]),
        "worst_error_band_1based": start + int(worst_band),
        "worst_error_kpoint_0based": int(worst_k),
        "worst_error_kpoint_fractional": [float(x) for x in kpoints[worst_k]],
        "worst_band_by_rmse_1based": start + worst_rmse_band,
        "worst_band_rmse_eV": float(band_scores[worst_rmse_band]),
        "worst_kpoint_by_rmse_0based": worst_rmse_k,
        "worst_kpoint_rmse_eV": float(k_scores[worst_rmse_k]),
        "rmse_within_1eV_below_fermi_eV": rmse(delta[occupied & (dft >= fermi - 1)]),
        "rmse_1_to_5eV_below_fermi_eV": rmse(delta[(dft < fermi - 1) & (dft >= fermi - 5)]),
        "rmse_more_than_5eV_below_fermi_eV": rmse(delta[dft < fermi - 5]),
        "fraction_occupied_abs_error_gt_0.1eV": float(np.mean(abs_occ > 0.1)),
        "fraction_occupied_abs_error_gt_0.5eV": float(np.mean(abs_occ > 0.5)),
        "gemini_projection_mode": gw["projection_mode"],
        "reference_projection_mode": rw["projection_mode"],
        "gemini_win_path": str(gemini_win_path),
        "reference_win_path": str(reference_win_path),
        "gemini_wout_path": str(gemini_wout_path),
        "reference_wout_path": str(reference_wout_path),
    }
    prefixed(result, "gemini", go)
    prefixed(result, "reference", ro)
    for key in (
        "num_bands", "num_iter", "dis_num_iter", "dis_win_min", "dis_win_max",
        "dis_froz_min", "dis_froz_max", "conv_tol", "conv_window",
        "guiding_centres", "scdm_proj", "scdm_entanglement", "scdm_mu", "scdm_sigma",
    ):
        result[f"gemini_{key}"] = gp.get(key)
        result[f"reference_{key}"] = rp.get(key)

    n = dft.shape[1]
    result["gemini_omega_total_per_wf_A2"] = go["omega_total_A2"] / n if finite(go["omega_total_A2"]) else None
    result["reference_omega_total_per_wf_A2"] = ro["omega_total_A2"] / n if finite(ro["omega_total_A2"]) else None
    result["omega_total_per_wf_ratio"] = ratio(result["gemini_omega_total_per_wf_A2"], result["reference_omega_total_per_wf_A2"])
    result["omega_I_ratio"] = ratio(go["omega_I_A2"], ro["omega_I_A2"])
    result["max_wf_spread_ratio"] = ratio(go["wf_spread_max_A2"], ro["wf_spread_max_A2"])

    hit_cap = bool(go["localization_hit_iteration_cap"])
    poorer = bool(
        (finite(result["omega_total_per_wf_ratio"]) and result["omega_total_per_wf_ratio"] >= args.spread_ratio_threshold)
        or (finite(result["max_wf_spread_ratio"]) and result["max_wf_spread_ratio"] >= args.max_wf_ratio_threshold)
    )
    setup_diff = gw["projection_mode"] != rw["projection_mode"] or any(
        gp.get(k) != rp.get(k)
        for k in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max", "scdm_proj", "scdm_mu", "scdm_sigma")
    )
    if poorer:
        bucket = "localization_spreads_worse_than_reference"
    elif hit_cap and setup_diff:
        bucket = "hit_cap_but_spreads_comparable; inspect_subspace_setup"
    elif setup_diff:
        bucket = "localization_comparable; inspect_subspace_setup"
    elif hit_cap:
        bucket = "hit_cap_but_spreads_comparable"
    else:
        bucket = "no_clear_signal_from_wout_or_win"
    result["localization_evidence_poor"] = poorer
    result["subspace_setup_differs"] = setup_diff
    result["evidence_bucket"] = bucket
    return result


def csv_value(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return float(np.median(values)) if values else None


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}g}" if finite(value) else "n/a"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "0/0"


def write_report(rows: list[dict[str, Any]], failures: list[dict[str, str]], args: argparse.Namespace) -> None:
    bad = [r for r in rows if r["gemini_to_reference_rmse_ratio"] >= args.ratio_threshold]
    good = [r for r in rows if r["gemini_to_reference_rmse_ratio"] < args.ratio_threshold]
    buckets = Counter(r["evidence_bucket"] for r in bad)
    projection_pairs = Counter((r["gemini_projection_mode"], r["reference_projection_mode"]) for r in bad)
    bad_projections = Counter(r["gemini_projection_mode"] for r in bad)
    good_projections = Counter(r["gemini_projection_mode"] for r in good)
    cap_bad = sum(bool(r["gemini_localization_hit_iteration_cap"]) for r in bad)
    cap_good = sum(bool(r["gemini_localization_hit_iteration_cap"]) for r in good)
    poor_bad = sum(bool(r["localization_evidence_poor"]) for r in bad)
    poor_good = sum(bool(r["localization_evidence_poor"]) for r in good)

    lines = [
        "# Gemini Wannierisation failure-mode report", "",
        f"Analyzed {len(rows)} comparable successful runs; {len(bad)} have Gemini/reference RMSE >= {args.ratio_threshold:g}. "
        f"{len(failures)} records could not be analyzed.", "",
        "The buckets below are diagnostic evidence, not proof of causality. A definitive diagnosis requires a controlled rerun changing one setup choice at a time.", "",
        "## Main triage", "",
        "| Evidence bucket | Count |", "|---|---:|",
    ]
    lines += [f"| {name} | {count} |" for name, count in buckets.most_common()]
    lines += [
        "", "## Bad cohort versus <threshold cohort", "",
        "| Signal | Bad cohort | Other successful runs |", "|---|---:|---:|",
        f"| Gemini localization hit configured iteration cap | {pct(cap_bad, len(bad))} | {pct(cap_good, len(good))} |",
        f"| Gemini spread evidence worse than reference | {pct(poor_bad, len(bad))} | {pct(poor_good, len(good))} |",
        f"| Median total spread per WF ratio | {fmt(median(bad, 'omega_total_per_wf_ratio'))} | {fmt(median(good, 'omega_total_per_wf_ratio'))} |",
        f"| Median maximum-WF-spread ratio | {fmt(median(bad, 'max_wf_spread_ratio'))} | {fmt(median(good, 'max_wf_spread_ratio'))} |",
        f"| Median RMSE within 1 eV below Fermi (eV) | {fmt(median(bad, 'rmse_within_1eV_below_fermi_eV'))} | {fmt(median(good, 'rmse_within_1eV_below_fermi_eV'))} |",
        f"| Median RMSE 1–5 eV below Fermi (eV) | {fmt(median(bad, 'rmse_1_to_5eV_below_fermi_eV'))} | {fmt(median(good, 'rmse_1_to_5eV_below_fermi_eV'))} |",
        f"| Median RMSE >5 eV below Fermi (eV) | {fmt(median(bad, 'rmse_more_than_5eV_below_fermi_eV'))} | {fmt(median(good, 'rmse_more_than_5eV_below_fermi_eV'))} |",
        "", "## Projection/setup patterns among bad runs", "",
        "| Gemini projection | Reference projection | Count |", "|---|---|---:|",
    ]
    lines += [f"| {g} | {r} | {n} |" for (g, r), n in projection_pairs.most_common()]
    lines += [
        "", "Projection mode by itself is only useful if it is enriched relative to the comparison cohort:", "",
        "| Gemini projection | Bad cohort | Other successful runs | Fraction bad within mode |", "|---|---:|---:|---:|",
    ]
    for mode in sorted(set(bad_projections) | set(good_projections)):
        b, g = bad_projections[mode], good_projections[mode]
        lines.append(f"| {mode} | {b} | {g} | {100*b/(b+g):.1f}% |")
    lines += [
        "", "## Highest absolute-error cases", "",
        "| Material | RMSE ratio | Gemini RMSE (eV) | Bucket | Spread/WF ratio | Worst location |", "|---|---:|---:|---|---:|---|",
    ]
    for row in sorted(bad, key=lambda x: x["gemini_rmse_eV"], reverse=True)[:20]:
        k = row["worst_error_kpoint_fractional"]
        where = f"band {row['worst_error_band_1based']}, k={k}"
        lines.append(
            f"| {row['material']} | {row['gemini_to_reference_rmse_ratio']:.3g} | "
            f"{row['gemini_rmse_eV']:.4g} | {row['evidence_bucket']} | "
            f"{fmt(row['omega_total_per_wf_ratio'])} | {where} |"
        )
    lines += [
        "", "## How to use this report", "",
        "Start with `failure_modes.csv`: filter by `evidence_bucket`, then sort by absolute Gemini RMSE (not only the ratio, which can be huge when the reference error is tiny). "
        "The `worst_*` and energy-window columns say where interpolation fails. The paired `gemini_*`/`reference_*` columns say how the setup and localization differ. "
        "Use the file-path columns to inspect only a few representatives from each cohort.", "",
        "The strongest next experiment is a matched rerun: keep Gemini's calculation fixed and replace only its initialization/projection strategy with the reference strategy; then separately test the reference disentanglement settings. This distinguishes initialization/localization failure from wrong subspace selection.", "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--jobs-root", type=Path, default=ROOT / "jobs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratio-threshold", type=float, default=2.0)
    parser.add_argument("--spread-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--max-wf-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--only-bad", action="store_true", help="Skip the comparison cohort below the RMSE-ratio threshold")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    records = [
        r for r in data.get("results", [])
        if isinstance(r, dict) and r.get("successful") is True and finite(r.get("gemini_to_reference_rmse_ratio"))
        and (not args.only_bad or r["gemini_to_reference_rmse_ratio"] >= args.ratio_threshold)
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for i, record in enumerate(records, 1):
        try:
            rows.append(analyse(record, args))
        except Exception as exc:
            failures.append({"material": str(record.get("material")), "error": f"{type(exc).__name__}: {exc}"})
        if i % 10 == 0 or i == len(records):
            print(f"Analyzed {i}/{len(records)} records ({len(failures)} errors)")

    ordered = sorted(rows, key=lambda r: r["gemini_to_reference_rmse_ratio"], reverse=True)
    (args.output_dir / "failure_modes.json").write_text(
        json.dumps({"rows": ordered, "analysis_failures": failures}, indent=2, allow_nan=False), encoding="utf-8"
    )
    if ordered:
        fieldnames = list(ordered[0])
        with (args.output_dir / "failure_modes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: csv_value(v) for k, v in row.items()} for row in ordered)
    write_report(ordered, failures, args)
    print(f"Wrote report, CSV, and JSON under: {args.output_dir}")


if __name__ == "__main__":
    main()
