#!/usr/bin/env python3
"""Compare Wannier90 .win projection and window choices across paired runs.

Run with this command

python scripts/compare_wannier_choices.py \
  jobs/case1_materials.json \
  --root . \
  --out jobs/win_choice_comparison.csv \
  --details jobs/win_choice_comparison.details.json

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

WINDOW_KEYS = ["dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"]
DIS_EXTRA_KEYS = ["dis_num_iter", "dis_mix_ratio"]
ORBITAL_DEGENERACY = {
    "s": 1,
    "p": 3,
    "d": 5,
    "f": 7,
    "sp": 4,
    "sp2": 3,
    "sp3": 4,
    "sp3d": 9,
    "sp3d2": 10,
}


def strip_comment(line: str) -> str:
    # Wannier90 uses ! for comments; many generated files also use #.
    for marker in ("!", "#"):
        if marker in line:
            line = line.split(marker, 1)[0]
    return line.strip()


def parse_scalar(value: str) -> Any:
    value = value.strip().strip(',')
    if not value:
        return None
    low = value.lower()
    if low in {"true", ".true.", "t"}:
        return True
    if low in {"false", ".false.", "f"}:
        return False
    try:
        if re.search(r"[.eEdD]", value):
            return float(value.replace("D", "E").replace("d", "e"))
        return int(value)
    except ValueError:
        return value


def canonical_projection(element: str, spec: str) -> str:
    element = element.strip()
    spec = re.sub(r"\s+", "", spec.strip().rstrip(','))
    spec = re.sub(r":=", "=", spec)
    spec = re.sub(r"=", "=", spec)
    if not element or not spec:
        return ""
    parts = spec.split(":")
    orbital = parts[0].lower()
    qualifiers = []
    for part in parts[1:]:
        if not part:
            continue
        # Normalize r = 2, r=2.0, zaxis=x, etc.
        if "=" in part:
            k, v = part.split("=", 1)
            qualifiers.append((k.lower(), v.lower()))
        else:
            qualifiers.append((part.lower(), ""))
    qtext = "" if not qualifiers else ":" + ":".join(
        f"{k}={v}" if v else k for k, v in sorted(qualifiers)
    )
    return f"{element}:{orbital}{qtext}"


def parse_projections(lines: list[str]) -> Counter[str]:
    projections: Counter[str] = Counter()
    in_block = False
    for raw in lines:
        line = strip_comment(raw)
        low = line.lower()
        if not line:
            continue
        if re.match(r"^begin\s+projections\b", low):
            in_block = True
            continue
        if re.match(r"^end\s+projections\b", low):
            in_block = False
            continue
        if not in_block:
            continue
        # Accept both "Ag: s; p; d" and "Ag:s".
        if ":" not in line:
            projections[canonical_projection("GLOBAL", line)] += 1
            continue
        element, rest = line.split(":", 1)
        for spec in rest.split(";"):
            item = canonical_projection(element, spec)
            if item:
                projections[item] += 1
    return projections


def parse_win(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    params: dict[str, Any] = {}
    for raw in lines:
        line = strip_comment(raw)
        if not line or line.lower().startswith(("begin ", "end ")):
            continue
        # Match key = value or key value.
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|:)\s*(.+)$", line)
        if not m:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$", line)
        if m:
            key = m.group(1).lower()
            val = m.group(2).strip()
            params[key] = parse_scalar(val)
    projections = parse_projections(lines)
    return {"path": str(path), "params": params, "projections": projections}


def find_best_win(run_path: Path, material: str) -> Path | None:
    candidates: list[Path] = []
    if run_path.is_file() and run_path.suffix == ".win":
        return run_path
    if not run_path.exists():
        return None
    # Prefer exact seed name.
    candidates.extend(run_path.rglob(f"{material}.win"))
    if not candidates:
        candidates.extend(run_path.rglob("*.win"))
    if not candidates:
        return None

    def score(p: Path) -> tuple[int, int, str]:
        s = str(p)
        pri = 100
        if "/artifacts/attempt_" in s:
            pri = 0
        elif "/workflow/run_dir/" in s:
            pri = 1
        elif p.name == f"{material}.win":
            pri = 2
        return (pri, len(p.parts), s)

    return sorted(candidates, key=score)[0]


def find_eig_near_win(win_path: Path, material: str) -> Path | None:
    for name in (f"{material}.eig", win_path.with_suffix(".eig").name):
        p = win_path.with_name(name)
        if p.exists():
            return p
    # broader but local
    for parent in [win_path.parent, *win_path.parents[:3]]:
        hits = list(parent.glob("*.eig"))
        if hits:
            return hits[0]
    return None


def eig_range(eig_path: Path | None) -> tuple[float | None, float | None]:
    if eig_path is None or not eig_path.exists():
        return None, None
    vals: list[float] = []
    for line in eig_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                vals.append(float(parts[2].replace("D", "E")))
            except ValueError:
                pass
    return (min(vals), max(vals)) if vals else (None, None)

def read_eig_by_kpoint(eig_path: Path | None) -> dict[int, dict[int, float]]:
    """Return {kpoint_index: {band_index: energy_eV}} from a Wannier90 .eig file."""
    if eig_path is None or not eig_path.exists():
        return {}

    result: dict[int, dict[int, float]] = {}
    for line in eig_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            band = int(parts[0])
            kpt = int(parts[1])
            energy = float(parts[2].replace("D", "E"))
        except ValueError:
            continue
        result.setdefault(kpt, {})[band] = energy
    return result


def numeric_param(params: dict[str, Any], key: str) -> float | None:
    value = params.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def energy_in_window(
    energy: float,
    lower: float | None,
    upper: float | None,
    tol: float = 1e-8,
) -> bool:
    if lower is not None and energy < lower - tol:
        return False
    if upper is not None and energy > upper + tol:
        return False
    return True


def window_mask(
    params: dict[str, Any],
    eig_by_kpoint: dict[int, dict[int, float]],
    *,
    kind: str,
) -> set[tuple[int, int]]:
    """
    Return selected (kpoint, band) pairs for either the outer or frozen window.

    kind='outer' uses dis_win_min/dis_win_max.
    kind='frozen' uses dis_froz_min/dis_froz_max.
    """
    if kind == "outer":
        lower = numeric_param(params, "dis_win_min")
        upper = numeric_param(params, "dis_win_max")
    elif kind == "frozen":
        lower = numeric_param(params, "dis_froz_min")
        upper = numeric_param(params, "dis_froz_max")
    else:
        raise ValueError(f"unknown window kind: {kind}")

    # For the outer window, omitted bounds mean unbounded.
    # For frozen windows, both omitted means no explicit frozen window.
    if kind == "frozen" and lower is None and upper is None:
        return set()

    selected: set[tuple[int, int]] = set()
    for kpt, bands in eig_by_kpoint.items():
        for band, energy in bands.items():
            if energy_in_window(energy, lower, upper):
                selected.add((kpt, band))
    return selected


def set_jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float | None:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def compare_window_masks(
    p1: dict[str, Any],
    p2: dict[str, Any],
    eig_path1: Path | None,
    eig_path2: Path | None,
) -> dict[str, Any]:
    eig1 = read_eig_by_kpoint(eig_path1)
    eig2 = read_eig_by_kpoint(eig_path2)

    if not eig1 or not eig2:
        return {
            "available": False,
            "outer_similarity": None,
            "frozen_similarity": None,
            "combined_similarity": None,
            "notes": ["missing or unreadable .eig file"],
        }

    # This assumes paired runs use the same k-point/band indexing.
    outer1 = window_mask(p1, eig1, kind="outer")
    outer2 = window_mask(p2, eig2, kind="outer")
    frozen1 = window_mask(p1, eig1, kind="frozen")
    frozen2 = window_mask(p2, eig2, kind="frozen")

    outer_sim = set_jaccard(outer1, outer2)
    frozen_sim = set_jaccard(frozen1, frozen2)

    if outer_sim is None or frozen_sim is None:
        combined = None
    else:
        combined = 0.4 * outer_sim + 0.6 * frozen_sim

    return {
        "available": True,
        "outer_similarity": outer_sim,
        "frozen_similarity": frozen_sim,
        "combined_similarity": combined,
        "outer_selected_count1": len(outer1),
        "outer_selected_count2": len(outer2),
        "frozen_selected_count1": len(frozen1),
        "frozen_selected_count2": len(frozen2),
    }

def fmt_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"

def multiset_jaccard(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


def projector_count(proj: Counter[str]) -> int | None:
    total = 0
    unknown = False
    for item, n in proj.items():
        orbital = item.split(":", 1)[1].split(":", 1)[0].lower() if ":" in item else item.lower()
        deg = ORBITAL_DEGENERACY.get(orbital)
        if deg is None:
            unknown = True
            continue
        total += n * deg
    return None if unknown else total


def close(a: Any, b: Any, tol: float = 1e-8) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def projection_diff(a: Counter[str], b: Counter[str]) -> dict[str, list[dict[str, Any]]]:
    only1 = []
    only2 = []
    for k in sorted(set(a) | set(b)):
        if a[k] > b[k]:
            only1.append({"projection": k, "extra_count": a[k] - b[k]})
        elif b[k] > a[k]:
            only2.append({"projection": k, "extra_count": b[k] - a[k]})
    return {"run1_extra": only1, "run2_extra": only2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_json", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."), help="Root directory for relative run paths; default current directory.")
    ap.add_argument("--out", type=Path, default=Path("win_choice_comparison.csv"))
    ap.add_argument("--details", type=Path, default=Path("win_choice_comparison.details.json"))
    args = ap.parse_args()

    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    rows = []
    details = {}
    for material, record in case.items():
        if not isinstance(record, list) or len(record) < 4:
            continue
        run1 = args.root / record[2]
        run2 = args.root / record[3]
        win1 = find_best_win(run1, material)
        win2 = find_best_win(run2, material)
        if win1 is None or win2 is None:
            rows.append({
                "material": material,
                "win1": str(win1) if win1 else "MISSING",
                "win2": str(win2) if win2 else "MISSING",
                "projection_similarity": "",
                "projections_equal": "",
                "outer_window_similarity": "",
                "frozen_window_similarity": "",
                "window_similarity": "",
                "summary": "missing .win file",
            })
            continue
        d1 = parse_win(win1)
        d2 = parse_win(win2)
        p1, p2 = d1["params"], d2["params"]
        proj1, proj2 = d1["projections"], d2["projections"]

        eig_path1 = find_eig_near_win(win1, material)
        eig_path2 = find_eig_near_win(win2, material)
        wmask = compare_window_masks(p1, p2, eig_path1, eig_path2)
        psim = multiset_jaccard(proj1, proj2)
        peq = proj1 == proj2
        extra_changed = {k: [p1.get(k), p2.get(k)] for k in DIS_EXTRA_KEYS if not close(p1.get(k), p2.get(k))}
        summary_bits = []
        if peq:
            summary_bits.append("same projections")
        else:
            summary_bits.append(f"projection similarity {psim:.3f}")
        window_sim = wmask["combined_similarity"]

        if window_sim is None:
            summary_bits.append("window similarity unavailable")
        elif math.isclose(float(window_sim), 1.0, rel_tol=0.0, abs_tol=1e-8):
            summary_bits.append("same window band masks")
        else:
            summary_bits.append(f"window mask similarity {float(window_sim):.3f}")
        if extra_changed:
            summary_bits.append("different disentanglement optimizer params")
        window_strict_equal = all(close(p1.get(k), p2.get(k)) for k in WINDOW_KEYS)

        row = {
            "material": material,
            "metric1": record[0],
            "metric2": record[1],
            "win1": str(win1),
            "win2": str(win2),
            "projection_similarity": f"{psim:.6f}",
            "projections_equal": peq,
            "projector_count1": projector_count(proj1),
            "projector_count2": projector_count(proj2),
            "window_strict_equal": window_strict_equal,
            "outer_window_similarity": fmt_float(wmask["outer_similarity"]),
            "frozen_window_similarity": fmt_float(wmask["frozen_similarity"]),
            "window_similarity": fmt_float(wmask["combined_similarity"]),
            "dis_extra_changed": json.dumps(extra_changed, sort_keys=True),
            "summary": "; ".join(summary_bits),
        }
        rows.append(row)
        details[material] = {
            "record": record,
            "win1": str(win1),
            "win2": str(win2),
            "windows1": {k: p1.get(k) for k in WINDOW_KEYS + DIS_EXTRA_KEYS},
            "windows2": {k: p2.get(k) for k in WINDOW_KEYS + DIS_EXTRA_KEYS},
            "window_mask_compare": wmask,
            "eig_path1": str(eig_path1) if eig_path1 else None,
            "eig_path2": str(eig_path2) if eig_path2 else None,
            "projections1": dict(sorted(proj1.items())),
            "projections2": dict(sorted(proj2.items())),
            "projection_compare": projection_diff(proj1, proj2),
            "projection_similarity": psim,
            "projector_count1": projector_count(proj1),
            "projector_count2": projector_count(proj2),
            "dis_extra_changed": extra_changed,
        }

    fieldnames = [
        "material",
        "metric1",
        "metric2",
        "win1",
        "win2",
        "projection_similarity",
        "projections_equal",
        "projector_count1",
        "projector_count2",
        "window_strict_equal",
        "outer_window_similarity",
        "frozen_window_similarity",
        "window_similarity",
        "dis_extra_changed",
        "summary",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.details.write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

