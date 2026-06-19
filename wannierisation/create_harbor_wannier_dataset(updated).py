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
DEFAULT_PSEUDO_DIR = ROOT / "input_packages" / "pseudos_sssp_efficiency"


DOCKERFILE = """FROM wannier-qe-gemini-base:0.46.0

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


GEMINIIGNORE = """# Keep large QE binary/checkpoint payloads out of Gemini CLI startup context.
# The files still exist under /app/material for QE/pw2wannier90 commands.
material/qe_save/out/aiida.wfc*
material/qe_save/out/aiida.save/wfc*.dat
material/qe_save/out/aiida.save/charge-density.dat
"""


SAFE_CAT = """#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        material/nscf/input/nscf.in|/app/material/nscf/input/nscf.in|\
        material/scf/input/scf.in|/app/material/scf/input/scf.in|\
        material/nscf/output/aiida.out|/app/material/nscf/output/aiida.out|\
        material/scf/output/aiida.out|/app/material/scf/output/aiida.out|\
        material/nscf/output/data-file.xml|/app/material/nscf/output/data-file.xml|\
        material/scf/output/data-file.xml|/app/material/scf/output/data-file.xml|\
        material/qe_save/logs/*.out|/app/material/qe_save/logs/*.out|\
        material/qe_save/out/aiida.xml|/app/material/qe_save/out/aiida.xml|\
        material/qe_save/out/aiida.save/data-file-schema.xml|/app/material/qe_save/out/aiida.save/data-file-schema.xml)
            if [ -f "$arg" ] && [ "$(wc -c < "$arg")" -gt 20000 ]; then
                {
                    echo "Refusing to print large QE file: $arg"
                    echo "Use grep/head/tail/sed -n for small excerpts, or parse it with a script that prints a compact summary."
                    echo "For copying, use cp rather than cat."
                } >&2
                exit 2
            fi
            ;;
    esac
done

exec /bin/cat "$@"
"""


SAFE_FILTER_COMMON = r'''
is_large_qe_text_arg() {
    case "$1" in
        material/nscf/input/nscf.in|/app/material/nscf/input/nscf.in|\
        material/scf/input/scf.in|/app/material/scf/input/scf.in|\
        material/nscf/output/aiida.out|/app/material/nscf/output/aiida.out|\
        material/scf/output/aiida.out|/app/material/scf/output/aiida.out|\
        material/nscf/output/data-file.xml|/app/material/nscf/output/data-file.xml|\
        material/scf/output/data-file.xml|/app/material/scf/output/data-file.xml|\
        material/qe_save/logs/*.out|/app/material/qe_save/logs/*.out|\
        material/qe_save/out/aiida.xml|/app/material/qe_save/out/aiida.xml|\
        material/qe_save/out/aiida.save/data-file-schema.xml|/app/material/qe_save/out/aiida.save/data-file-schema.xml)
            [ -f "$1" ] && [ "$(wc -c < "$1")" -gt 20000 ]
            return
            ;;
    esac
    return 1
}

touches_large_qe_text() {
    for arg in "$@"; do
        is_large_qe_text_arg "$arg" && return 0
    done
    return 1
}

stdout_is_regular_file() {
    [ -e /proc/self/fd/1 ] && [ -f /proc/self/fd/1 ]
}

run_with_chat_cap() {
    real_cmd="$1"
    shift
    if touches_large_qe_text "$@" && ! stdout_is_regular_file; then
        tmp="$(mktemp)"
        set +e
        "$real_cmd" "$@" > "$tmp"
        status=$?
        set -e
        max_lines="${SAFE_QE_TEXT_MAX_LINES:-80}"
        line_count="$(wc -l < "$tmp" | tr -d ' ')"
        if [ "$line_count" -gt "$max_lines" ]; then
            /usr/bin/sed -n "1,${max_lines}p" "$tmp"
            echo "[safe-output-cap] Suppressed $((line_count - max_lines)) additional line(s) from large QE text output."
            echo "[safe-output-cap] Redirect to a file for full output, then inspect compact excerpts."
        else
            /bin/cat "$tmp"
        fi
        rm -f "$tmp"
        exit "$status"
    fi
    exec "$real_cmd" "$@"
}
'''


SAFE_LS = """#!/usr/bin/env bash
set -euo pipefail

compact_aiida_save() {
    dir="$1"
    echo "Compact listing for large QE save directory: $dir"
    printf "file_count: "
    find "$dir" -maxdepth 1 -type f | wc -l
    printf "wfc_file_count: "
    find "$dir" -maxdepth 1 -type f -name 'wfc*.dat' | wc -l
    printf "total_size: "
    du -sh "$dir" 2>/dev/null | awk '{print $1}' || true
    echo "sample_files:"
    /bin/ls -1 "$dir" | sed -n '1,30p'
    echo "Full ls output for this directory is suppressed; use find/count/head for compact queries."
}

compact_recursive_listing() {
    target="$1"
    echo "Recursive ls output suppressed for large QE workspace: $target"
    echo "Compact tree, excluding QE wavefunction/checkpoint payloads:"
    find "$target" -maxdepth 5 -print \
        | /usr/bin/grep -Ev '(^|/)material/qe_save/out/aiida\\.save/wfc[0-9]+\\.dat$|(^|/)material/qe_save/out/aiida\\.wfc[0-9]+$|(^|/)material/qe_save/out/aiida\\.save/charge-density\\.dat$' \
        | sed -n '1,200p'
    echo "Use targeted find/grep/head queries for more detail."
}

has_recursive=0
targets=()
for arg in "$@"; do
    case "$arg" in
        -R|--recursive|-*R*)
            has_recursive=1
            ;;
        -*)
            ;;
        *)
            targets+=("$arg")
            ;;
    esac
done

if [ "$has_recursive" -eq 1 ]; then
    if [ "${#targets[@]}" -eq 0 ]; then
        targets=(".")
    fi
    for target in "${targets[@]}"; do
        case "$target" in
            .|./|/app|/app/|material|material/|/app/material|/app/material/|\
            material/qe_save*|/app/material/qe_save*)
                if [ -d "$target" ]; then
                    compact_recursive_listing "$target"
                    exit 0
                fi
                ;;
        esac
    done
fi

for arg in "$@"; do
    case "$arg" in
        material/qe_save/out/aiida.save|/app/material/qe_save/out/aiida.save)
            if [ -d "$arg" ]; then
                compact_aiida_save "$arg"
                exit 0
            fi
            ;;
        material/qe_save/out/aiida.save/wfc*.dat|/app/material/qe_save/out/aiida.save/wfc*.dat)
            dir="$(dirname "$arg")"
            if [ -d "$dir" ]; then
                compact_aiida_save "$dir"
                exit 0
            fi
            ;;
    esac
done

exec /bin/ls "$@"
"""


SAFE_GREP = """#!/usr/bin/env bash
set -euo pipefail
""" + SAFE_FILTER_COMMON + """

touches_qe_upf=0
dangerous_pattern=0

for arg in "$@"; do
    case "$arg" in
        material/qe_save/out/aiida.save/*.UPF|/app/material/qe_save/out/aiida.save/*.UPF)
            touches_qe_upf=1
            ;;
        *PP_PSWFC*|*Wavefunction*|*WAVEFUNCTION*|*PP_BETA*|*PP_R*|*PP_RAB*)
            dangerous_pattern=1
            ;;
    esac
done

if [ "$touches_qe_upf" -eq 1 ] && [ "$dangerous_pattern" -eq 1 ]; then
    {
        echo "Refusing grep that may print large UPF radial arrays from qe_save."
        echo "Use a small parser that prints only orbital labels/counts, or limit output with rg -m/head."
    } >&2
    exit 2
fi

run_with_chat_cap /usr/bin/grep "$@"
"""


SAFE_SED = """#!/usr/bin/env bash
set -euo pipefail
""" + SAFE_FILTER_COMMON + """
run_with_chat_cap /usr/bin/sed "$@"
"""


SAFE_AWK = """#!/usr/bin/env bash
set -euo pipefail
""" + SAFE_FILTER_COMMON + """
run_with_chat_cap /usr/bin/awk "$@"
"""


SAFE_HEAD = """#!/usr/bin/env bash
set -euo pipefail
""" + SAFE_FILTER_COMMON + """
run_with_chat_cap /usr/bin/head "$@"
"""


SAFE_TAIL = """#!/usr/bin/env bash
set -euo pipefail
""" + SAFE_FILTER_COMMON + """
run_with_chat_cap /usr/bin/tail "$@"
"""


SAFE_RG = """#!/usr/bin/env bash
set -euo pipefail
""" + SAFE_FILTER_COMMON + """
run_with_chat_cap /usr/bin/rg "$@"
"""


TEST_SH = """#!/usr/bin/env bash
set -euo pipefail

preserve_artifacts() {
    mkdir -p /logs/artifacts
    if [ -d /app/artifacts ]; then
        cp -a /app/artifacts/. /logs/artifacts/ 2>/dev/null || true
    fi
    if [ -f /app/report.json ]; then
        cp /app/report.json /logs/artifacts/report.json 2>/dev/null || true
    fi
    if [ -f /app/REPORT.md ]; then
        cp /app/REPORT.md /logs/artifacts/REPORT.md 2>/dev/null || true
    fi
}

trap preserve_artifacts EXIT
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


def required_pseudos_from_qe_input(path: Path) -> list[str]:
    pseudos: list[str] = []
    in_species = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() == "ATOMIC_SPECIES":
            in_species = True
            continue
        if not in_species:
            continue
        keyword = line.split()[0].upper()
        if keyword in {"ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS"}:
            break
        parts = line.split()
        if len(parts) >= 3:
            pseudos.append(parts[2])
    return pseudos


def copy_required_pseudos(agent_material: Path, material_dir: Path, pseudo_dir: Path | None) -> dict[str, list[str]]:
    required = sorted(set(required_pseudos_from_qe_input(agent_material / "nscf" / "input" / "nscf.in")))
    copied: list[str] = []
    missing: list[str] = []
    if not required:
        return {"required": [], "copied": [], "missing": []}

    target_dir = material_dir / "pseudo"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        source = pseudo_dir / name if pseudo_dir is not None else None
        if source is not None and source.exists():
            shutil.copy2(source, target_dir / name)
            copied.append(name)
        else:
            missing.append(name)

    write_text(target_dir / "required_pseudos.txt", "\n".join(required) + "\n")
    if missing:
        write_text(target_dir / "missing_pseudos.txt", "\n".join(missing) + "\n")
    return {"required": required, "copied": copied, "missing": missing}


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

If `/app/material/pseudo/` contains UPF files, use that directory when rerunning
SCF/NSCF instead of downloading pseudopotentials.

The hidden verifier reads `/tests/reference` after the agent finishes.
"""


def base_instruction_text(prompt_path: Path, existing_instruction_path: Path) -> str:
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").rstrip()
    if existing_instruction_path.exists():
        text = existing_instruction_path.read_text(encoding="utf-8")
        return text.split("\n## Harbor grading note", 1)[0].rstrip()
    raise FileNotFoundError(
        f"Missing prompt {prompt_path} and no existing instruction {existing_instruction_path}"
    )


def instruction_text(base_instruction: str) -> str:
    return base_instruction.rstrip() + """

If `material/pseudo/` contains UPF files, those are the pseudopotentials that
match the supplied QE inputs. Prefer rerunning SCF/NSCF with
`pseudo_dir = './pseudo/'` from a workflow run directory that symlinks or copies
`material/pseudo/`, especially when the supplied QE 6.3 XML cannot be consumed
directly by the installed `pw2wannier90.x`.

## Recommended QE-save workflow

When `material/qe_save/out/aiida.save` exists, use it directly as the DFT-side
input to `pw2wannier90.x`; do not inspect or print the wavefunction files.
Do not spend the run doing open-ended analysis before creating files. Start by
creating `workflow/run_dir`, writing a first `<seed>.win`, and running the
QE-save workflow below. Keep any pre-work to a few compact metadata/log
queries.

The usual workflow is:

```bash
seed="<task seedname>"
mkdir -p workflow/run_dir
cp -a material/qe_save/out workflow/run_dir/out
cp material/pseudo/*.UPF workflow/run_dir/ 2>/dev/null || true
cd workflow/run_dir

# Write ${seed}.win with the task's required num_wann/num_bands, projections,
# kpoint_path/bands_plot settings if desired, and disentanglement windows.
wannier90.x -pp "${seed}"

cat > "${seed}.pw2wan" <<EOF
&inputpp
  outdir = './out'
  prefix = 'aiida'
  seedname = '${seed}'
  write_mmn = .true.
  write_amn = .true.
  write_eig = .true.
/
EOF
pw2wannier90.x -in "${seed}.pw2wan"
wannier90.x "${seed}"
```

Only rerun SCF/NSCF if this save-tree workflow fails for a recorded technical
reason. Use compact scripts to summarize QE logs/XML when choosing projections
or windows; keep the full save tree as file input rather than chat output.

Keep terminal output compact. Do not print full QE inputs, XML files, logs, or
K-point lists into the chat transcript with commands like `cat` on
`material/nscf/input/nscf.in` or `material/qe_save/logs/nscf.out`. Extract only
the needed lines with `grep`, `head`, `tail`, `sed -n`, or small scripts, and
redirect generated full K-point lists or workflow files directly to files.
Also avoid full listings of `material/qe_save/out/aiida.save` and avoid
`grep -A`/`grep -B` on UPF wavefunction or `PP_PSWFC` blocks, because those
include large binary/checkpoint listings or long radial numeric tables.

## Harbor grading note

The hidden verifier compares your final Wannier Hamiltonian against withheld
DFT reference bands. The best/final attempt must include `<seed>_hr.dat` in
`artifacts/attempt_<N>/`; without that file the run receives reward 0 because
there is no Hamiltonian to evaluate.

Before returning a final response, verify that the final artifact exists:

```bash
test -s artifacts/attempt_1/run_manifest.json
test -s artifacts/attempt_1/${seed}_hr.dat
ls -lh artifacts/attempt_1/*_hr.dat
```

Do not report `status: "success"` or `executed_successfully: true` unless
`artifacts/attempt_1/<seed>_hr.dat` exists and is non-empty. Do not run the
final workflow in the background with `nohup` or `&` and then return before it
finishes. When collecting artifacts, do not rely on `cp <seed>.* ...` because
that does not copy `<seed>_hr.dat`; copy `<seed>_hr.dat` explicitly or use
`cp <seed>* ...`.

For example, after a successful Wannier90 run:

```bash
mkdir -p artifacts/attempt_1
cp workflow/run_dir/${seed}_hr.dat artifacts/attempt_1/
cp workflow/run_dir/${seed}.win workflow/run_dir/${seed}.wout workflow/run_dir/${seed}.eig workflow/run_dir/${seed}.chk workflow/run_dir/${seed}.nnkp artifacts/attempt_1/ 2>/dev/null || true
```

If `<seed>_hr.dat` is missing or empty after your attempts, return
`status: "partial"` or `status: "failed"`, not `status: "success"`.

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
    pseudo_dir: Path | None,
    overwrite: bool,
) -> None:
    task_dir = output_dir / material
    agent_material = agent_dir / material
    reference_material = reference_dir / material
    prompt_path = prompt_dir / material / "prompt.md"
    base_instruction = base_instruction_text(prompt_path, task_dir / "instruction.md")
    clean_or_create(task_dir, overwrite)

    reference_meta = read_json(reference_material / "reference_metadata.json")

    write_text(task_dir / "instruction.md", instruction_text(base_instruction))
    write_text(task_dir / "task.toml", task_toml(material))
    write_text(task_dir / "environment" / "Dockerfile", DOCKERFILE)
    write_text(task_dir / "environment" / ".geminiignore", GEMINIIGNORE)
    write_text(task_dir / "environment" / "safe_cat", SAFE_CAT, mode=0o755)
    write_text(task_dir / "environment" / "safe_ls", SAFE_LS, mode=0o755)
    write_text(task_dir / "environment" / "safe_grep", SAFE_GREP, mode=0o755)
    write_text(task_dir / "environment" / "safe_sed", SAFE_SED, mode=0o755)
    write_text(task_dir / "environment" / "safe_awk", SAFE_AWK, mode=0o755)
    write_text(task_dir / "environment" / "safe_head", SAFE_HEAD, mode=0o755)
    write_text(task_dir / "environment" / "safe_tail", SAFE_TAIL, mode=0o755)
    write_text(task_dir / "environment" / "safe_rg", SAFE_RG, mode=0o755)
    write_text(task_dir / "environment" / "README.md", task_readme(material))
    copytree(agent_material, task_dir / "environment" / "material")
    pseudo_status = copy_required_pseudos(agent_material, task_dir / "environment" / "material", pseudo_dir)
    write_text(
        task_dir / "environment" / "material" / "pseudo" / "pseudo_manifest.json",
        json.dumps(pseudo_status, indent=2) + "\n",
    )

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
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=DEFAULT_PSEUDO_DIR,
        help="Directory containing UPF files to bundle under material/pseudo.",
    )
    parser.add_argument(
        "--no-pseudos",
        action="store_true",
        help="Do not bundle pseudopotentials into the agent-visible material package.",
    )
    parser.add_argument("--materials", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fermi_energies = read_json(args.fermi_energies)
    materials = material_ids(args.agent_dir, args.materials)
    pseudo_dir = None if args.no_pseudos else args.pseudo_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for material in materials:
        create_task(
            material,
            args.output_dir,
            args.agent_dir,
            args.reference_dir,
            args.prompt_dir,
            fermi_energies,
            pseudo_dir,
            args.overwrite,
        )
    write_dataset_readme(args.output_dir, materials)
    print(f"Wrote {len(materials)} Harbor tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
