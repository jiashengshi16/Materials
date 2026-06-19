#!/usr/bin/env python3

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# ONLY INPUT YOU EDIT
# =========================

INPUT_JSON = "/Users/jshi/Documents/GitHub/WannierisationBenchmarking/jobs/num_wann_ordered_diagnostics_summary.json"

# =========================


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def get_first_number(d, keys):
    for key in keys:
        value = d.get(key)
        if is_number(value):
            return float(value)
    return None


def extract_per_run_rows(data):
    rows = []

    candidate_records = data.get("results", [])

    if not isinstance(candidate_records, list):
        candidate_records = []

    for d in candidate_records:
        if not isinstance(d, dict):
            continue

        if d.get("successful") is not True:
            continue

        reward = d.get("reward")
        if not is_number(reward) or reward == 0.0:
            continue

        num_wann = get_first_number(
            d,
            [
                "num_wann",
                "num_wann_from_folder",
                "num_wannier",
                "n_wann",
                "num_target_bands",
            ],
        )

        rmse = get_first_number(
            d,
            [
                "rmse_eV",
                "gemini_offmesh_rmse_eV",
                "offmesh_rmse_eV",
            ],
        )

        gemini_rmse = get_first_number(
            d,
            [
                "gemini_offmesh_rmse_eV",
                "rmse_eV",
                "offmesh_rmse_eV",
            ],
        )

        reference_rmse = get_first_number(
            d,
            [
                "reference_offmesh_rmse_eV",
                "reference_rmse_eV",
                "ref_offmesh_rmse_eV",
            ],
        )

        ratio = get_first_number(
            d,
            [
                "gemini_to_reference_rmse_ratio",
                "ratio",
            ],
        )

        if ratio is None and is_number(gemini_rmse) and is_number(reference_rmse) and reference_rmse > 0:
            ratio = gemini_rmse / reference_rmse

        if not is_number(num_wann):
            continue

        rows.append(
            {
                "material": d.get("material") or d.get("material_from_folder"),
                "num_wann": float(num_wann),
                "reward": float(reward),
                "rmse_eV": rmse,
                "gemini_offmesh_rmse_eV": gemini_rmse,
                "reference_offmesh_rmse_eV": reference_rmse,
                "ratio": ratio,
            }
        )

    return rows


def extract_summary_rows(data):
    rows = []

    for group in data.get("by_num_wann", []):
        if not isinstance(group, dict):
            continue

        counts = group.get("counts", {})
        reward = group.get("reward", {})
        rmse = group.get("rmse_eV", {})

        success_count = counts.get("success", 0)
        reward_mean = reward.get("mean")
        rmse_mean = rmse.get("mean")
        num_wann = group.get("num_wann")

        if not is_number(success_count) or success_count <= 0:
            continue
        if not is_number(reward_mean) or reward_mean == 0.0:
            continue
        if not is_number(num_wann):
            continue

        rows.append(
            {
                "material": None,
                "num_wann": float(num_wann),
                "reward": float(reward_mean),
                "rmse_eV": float(rmse_mean) if is_number(rmse_mean) else None,
                "gemini_offmesh_rmse_eV": float(rmse_mean) if is_number(rmse_mean) else None,
                "reference_offmesh_rmse_eV": None,
                "ratio": None,
            }
        )

    return rows


def valid_xy(rows, x_key, y_key):
    xs = []
    ys = []

    for r in rows:
        x = r.get(x_key)
        y = r.get(y_key)

        if is_number(x) and is_number(y) and x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    return xs, ys


def save_plot_1(rows, outdir):
    xs_rmse, ys_rmse = valid_xy(rows, "num_wann", "rmse_eV")
    xs_gemini, ys_gemini = valid_xy(rows, "num_wann", "gemini_offmesh_rmse_eV")

    print("Plot 1 point counts:")
    print(f"  rmse_eV points = {len(xs_rmse)}")
    print(f"  gemini_offmesh_rmse_eV points = {len(xs_gemini)}")

    plt.figure(figsize=(9, 5.5))

    plt.scatter(
        [x - 0.2 for x in xs_rmse],
        ys_rmse,
        s=55,
        marker="x",
        linewidths=1.8,
        color="tab:blue",
        alpha=0.9,
        label=f"rmse_eV (n={len(xs_rmse)})",
    )

    plt.scatter(
        [x + 0.2 for x in xs_gemini],
        ys_gemini,
        s=38,
        marker="o",
        color="tab:orange",
        alpha=0.7,
        label=f"gemini_offmesh_rmse_eV (n={len(xs_gemini)})",
    )

    plt.yscale("log")
    plt.xlabel("num_wann")
    plt.ylabel("RMSE error, eV")
    plt.title("RMSE vs number of Wannier functions")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = outdir / "01_num_wann_vs_rmse.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"Saved {path}")


def save_plot_2(rows, outdir):
    xs, ys = valid_xy(rows, "reference_offmesh_rmse_eV", "gemini_offmesh_rmse_eV")

    print(f"Plot 2 point count: {len(xs)}")

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(xs, ys, alpha=0.75, s=40, color="tab:green")

    lo = min(xs + ys)
    hi = max(xs + ys)

    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.5,
        color="black",
        alpha=0.7,
        label="Gemini = reference",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("reference_offmesh_rmse_eV")
    plt.ylabel("gemini_offmesh_rmse_eV")
    plt.title(f"Gemini RMSE vs reference RMSE (n={len(xs)})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = outdir / "02_reference_rmse_vs_gemini_rmse.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"Saved {path}")


def save_plot_3(rows, outdir):
    xs = []
    ys = []

    for r in rows:
        num_wann = r.get("num_wann")
        ratio = r.get("ratio")

        if is_number(num_wann) and is_number(ratio) and num_wann > 0 and ratio > 0:
            xs.append(num_wann)
            ys.append(ratio)

    print(f"Plot 3 point count: {len(xs)}")

    plt.figure(figsize=(9, 5.5))
    plt.scatter(xs, ys, alpha=0.75, s=40, color="tab:red")

    plt.axhline(1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.7, label="ratio = 1")
    plt.axhline(2.0, linestyle=":", linewidth=1.2, color="black", alpha=0.6, label="ratio = 2")
    plt.axhline(10.0, linestyle="-.", linewidth=1.2, color="black", alpha=0.5, label="ratio = 10")

    plt.yscale("log")
    plt.xlabel("num_wann")
    plt.ylabel("gemini_offmesh_rmse_eV / reference_offmesh_rmse_eV")
    plt.title(f"Gemini/reference RMSE ratio vs number of Wannier functions (n={len(xs)})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = outdir / "03_num_wann_vs_gemini_reference_ratio.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"Saved {path}")


def save_plot_4(rows, outdir):
    xs = []
    ys = []

    for r in rows:
        reference = r.get("reference_offmesh_rmse_eV")
        ratio = r.get("ratio")

        if is_number(reference) and is_number(ratio) and reference > 0 and ratio > 0:
            xs.append(reference)
            ys.append(ratio)

    print(f"Plot 4 point count: {len(xs)}")

    plt.figure(figsize=(6.8, 6.2))
    plt.scatter(xs, ys, alpha=0.75, s=40, color="tab:purple")

    plt.axhline(1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.7, label="ratio = 1")
    plt.axhline(2.0, linestyle=":", linewidth=1.2, color="black", alpha=0.6, label="ratio = 2")
    plt.axhline(10.0, linestyle="-.", linewidth=1.2, color="black", alpha=0.5, label="ratio = 10")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("reference_offmesh_rmse_eV")
    plt.ylabel("gemini_offmesh_rmse_eV / reference_offmesh_rmse_eV")
    plt.title(f"Gemini/reference ratio vs reference RMSE (n={len(xs)})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = outdir / "04_reference_rmse_vs_gemini_reference_ratio.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"Saved {path}")


def main():
    json_path = Path(INPUT_JSON).expanduser().resolve()
    outdir = json_path.parent

    with open(json_path, "r") as f:
        data = json.load(f)

    rows = extract_per_run_rows(data)

    if rows:
        print(f"Using per-run successful rows with reward != 0.0: {len(rows)}")
    else:
        print("No per-run rows found. Falling back to summary rows.")
        rows = extract_summary_rows(data)
        print(f"Using summary rows: {len(rows)}")

    if not rows:
        raise RuntimeError("No usable rows found.")

    print()
    print("Field availability among selected rows:")
    print(f"  rows = {len(rows)}")
    print(f"  rmse_eV available = {sum(is_number(r.get('rmse_eV')) for r in rows)}")
    print(f"  gemini_offmesh_rmse_eV available = {sum(is_number(r.get('gemini_offmesh_rmse_eV')) for r in rows)}")
    print(f"  reference_offmesh_rmse_eV available = {sum(is_number(r.get('reference_offmesh_rmse_eV')) for r in rows)}")
    print(f"  ratio available = {sum(is_number(r.get('ratio')) for r in rows)}")
    print()

    save_plot_1(rows, outdir)
    save_plot_2(rows, outdir)
    save_plot_3(rows, outdir)
    save_plot_4(rows, outdir)


if __name__ == "__main__":
    main()