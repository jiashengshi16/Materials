#!/usr/bin/env python3
"""Print Harbor commands for DeepSeek runs with prior self-debug context.

By default this uses every material with report pairs under
jobsDeepseekProTerminus2/deepseek_pro_debug_reviews. To restrict the run, edit
MATERIALS or pass one or more --material values.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import sys

import generate_harbor_num_wann_order_command as harbor_generator
import generate_harbor_self_debug_context_command as self_debug_generator


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "openai/deepseek-v4-pro"

# Hardcoded workflow selector.
# - "chemically similar": existing workflow; candidate-material self-debug reports
#   from include_only_candidates.csv are copied as context.
# - "codex_self_review": only Codex next-run recommendations are copied as context.
WORKFLOW = "codex_self_review"
SUPPORTED_WORKFLOWS = {"chemically similar", "codex_self_review"}

DEEPSEEK_SELF_DEBUG_REVIEWS_ROOT = (
    self_debug_generator.ROOT
    / "jobsDeepseekProTerminus2"
    / "deepseek_pro_debug_reviews"
)
DEFAULT_CODEX_NEXT_RUN_DIAGNOSES = (
    self_debug_generator.ROOT
    / "jobsDeepseekProTerminus2InstructionTest"
    / "codex_next_run_diagnoses.md"
)
DEFAULT_CANDIDATE_RUN_ERROR_TABLE = (
    self_debug_generator.ROOT
    / "jobsDeepseekProTerminus2Candidates"
    / "include_only_candidates.csv"
)
DEFAULT_JOBS_ROOT = (
    self_debug_generator.ROOT / "jobsDeepseekProTerminus2ControlledSelfDebugContext"
    if WORKFLOW == "codex_self_review"
    else self_debug_generator.ROOT / "jobsDeepseekProTerminus2SelfDebugContext"
)
DEFAULT_SELF_DEBUG_REVIEWS_ROOT = (
    DEEPSEEK_SELF_DEBUG_REVIEWS_ROOT
)

# Leave empty to use all materials that have DeepSeek self-debug reports.
MATERIALS: list[str] = [
    "Al4Sc2",
    "Al18Co4",
    "Li4O6Si2",
    "Mg2O10Ti4",
    "Si6Y10",
]

NEXT_RUN_TRACE_WRAPPER_NAME = "trace_next_run_file_access.sh"
NEXT_RUN_TRACE_VERIFIER_NAME = "verify_next_run_context_access.py"
NEXT_RUN_TRACE_WRAPPER_APP_PATH = "/app/trace_next_run_file_access.sh"
LOCKED_RUNNER_NAME = "locked_wannier_runner.py"
LOCKED_RUNNER_APP_PATH = "/app/locked_wannier_runner.py"
LOCKED_COMMAND_WRAPPER_NAME = "locked_command_wrapper.sh"
LOCKED_COMMAND_WRAPPER_APP_PATH = "/app/locked_command_wrapper.sh"
LOCKED_BIN_APP_DIR = "/app/locked_bin"
LOCKED_DENIED_COMMANDS = [
    "wannier90.x",
    "pw2wannier90.x",
    "pw.x",
    "rm",
    "kill",
    "pkill",
    "killall",
]
NEXT_RUN_TRACE_ARTIFACTS = [
    "/app/workflow/next_run_file_trace.log",
    "/app/workflow/NEXT_RUN_CONTEXT_SUMMARY.json",
    "/app/workflow/recipe_request.json",
    "/app/workflow/LOCKED_RECIPE.json",
    "/app/workflow/locked_runner.log",
]


def next_run_trace_wrapper_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
TRACE_PATH=/app/workflow/next_run_file_trace.log
mkdir -p /app/workflow
: > "$TRACE_PATH"
if ! command -v strace >/dev/null 2>&1; then
  echo "ERROR: strace is not installed in the task image; cannot enforce next-run context reads" >&2
  echo "ERROR: strace_missing" > "$TRACE_PATH"
  exit 127
fi
exec strace -f \
  -e trace=openat,open,read,close,stat,newfstatat,access \
  -s 300 \
  -o "$TRACE_PATH" \
  "$@"
"""


def terminus_login_trace_profile_script() -> str:
    return """# Route mutable workflow commands through the locked runner policy.
# Disable terminal-generated interrupts; the locked runner owns timeouts/failure.
stty intr undef 2>/dev/null || true
trap '' INT

if [ -d /app/locked_bin ]; then
  case ":${PATH}:" in
    *:/app/locked_bin:*) ;;
    *) export PATH="/app/locked_bin:${PATH}" ;;
  esac
fi
if [ -x /app/locked_command_wrapper.sh ] \
  && [ "${LOCKED_WANNIER_RUNNER_ACTIVE:-}" != "1" ]; then
  wannier90.x() { /app/locked_bin/wannier90.x "$@"; }
  pw2wannier90.x() { /app/locked_bin/pw2wannier90.x "$@"; }
  pw.x() { /app/locked_bin/pw.x "$@"; }
  rm() { /app/locked_bin/rm "$@"; }
  kill() { /app/locked_bin/kill "$@"; }
  pkill() { /app/locked_bin/pkill "$@"; }
  killall() { /app/locked_bin/killall "$@"; }
fi

# Auto-start Terminus login shells under the configured file-access tracer.
if [ -n "${HARBOR_AGENT_COMMAND_WRAPPER:-}" ] \
  && [ -z "${HARBOR_AGENT_COMMAND_WRAPPER_ACTIVE:-}" ] \
  && [ -x "${HARBOR_AGENT_COMMAND_WRAPPER:-}" ]; then
  export HARBOR_AGENT_COMMAND_WRAPPER_ACTIVE=1
  exec "${HARBOR_AGENT_COMMAND_WRAPPER}" /bin/bash --login
fi
"""


def next_run_trace_verifier_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return data


def trace_has_path_access(trace_text: str, app_path: str) -> bool:
    escaped = re.escape(app_path)
    return re.search(r"\\b(openat|open|stat|newfstatat|access)\\([^\\n]*" + escaped, trace_text) is not None


def read_bytes_by_path(trace_text: str) -> dict[str, int]:
    fd_paths: dict[tuple[str, str], str] = {}
    totals: dict[str, int] = {}
    open_re = re.compile(
        r"^\\s*(?P<pid>\\d+)\\s+(?:openat|open)\\([^\\n]*?\\\"(?P<path>/app/next_run_context/[^\\\"]+)\\\"[^\\n]*\\)\\s+=\\s+(?P<fd>\\d+)"
    )
    read_re = re.compile(
        r"^\\s*(?P<pid>\\d+)\\s+read\\((?P<fd>\\d+),.*\\)\\s+=\\s+(?P<count>-?\\d+)"
    )
    close_re = re.compile(r"^\\s*(?P<pid>\\d+)\\s+close\\((?P<fd>\\d+)\\)\\s+=\\s+0")

    for line in trace_text.splitlines():
        open_match = open_re.match(line)
        if open_match:
            fd_paths[(open_match.group("pid"), open_match.group("fd"))] = open_match.group("path")
            continue
        read_match = read_re.match(line)
        if read_match:
            count = int(read_match.group("count"))
            path = fd_paths.get((read_match.group("pid"), read_match.group("fd")))
            if path and count > 0:
                totals[path] = totals.get(path, 0) + count
            continue
        close_match = close_re.match(line)
        if close_match:
            fd_paths.pop((close_match.group("pid"), close_match.group("fd")), None)
    return totals


def verify(index_path: Path, summary_path: Path, trace_path: Path) -> list[str]:
    errors: list[str] = []
    if not index_path.is_file():
        return [f"missing index.json: {index_path}"]
    if not summary_path.is_file():
        return [f"missing NEXT_RUN_CONTEXT_SUMMARY.json: {summary_path}"]
    if not trace_path.is_file():
        return [f"missing next_run_file_trace.log: {trace_path}"]

    index = load_json(index_path)
    summary = load_json(summary_path)
    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    if "trace_wrapper_not_invoked" in trace_text:
        errors.append("trace wrapper was not invoked; the Terminus shell was not run under strace")
    if "strace_missing" in trace_text:
        errors.append("strace is missing in the task image")
    if "read(" not in trace_text:
        errors.append("trace contains no read(2) syscalls")

    required_paths = [
        "/app/next_run_context/index.json",
        index.get("required_bundle_path"),
        index.get("raw_source_path"),
    ]
    required_paths = [path for path in required_paths if isinstance(path, str) and path]
    byte_totals = read_bytes_by_path(trace_text)
    for required_path in required_paths:
        if not trace_has_path_access(trace_text, required_path):
            errors.append(f"no OS trace evidence of opening/stat/access for {required_path}")
        if byte_totals.get(required_path, 0) <= 0:
            errors.append(f"no positive read(2) bytes recorded for {required_path}")

    if summary.get("target_material") != index.get("target_material"):
        errors.append(
            f"summary target_material={summary.get('target_material')!r} "
            f"does not match index target_material={index.get('target_material')!r}"
        )
    if summary.get("bundle_path") != index.get("required_bundle_path"):
        errors.append(
            f"summary bundle_path={summary.get('bundle_path')!r} "
            f"does not match index required_bundle_path={index.get('required_bundle_path')!r}"
        )
    if summary.get("index_path") != "/app/next_run_context/index.json":
        errors.append(
            f"summary index_path={summary.get('index_path')!r} "
            "does not match /app/next_run_context/index.json"
        )
    if summary.get("read_complete_bundle") is not True:
        errors.append("summary read_complete_bundle is not true")

    print(json.dumps({
        "required_paths": required_paths,
        "read_bytes": {path: byte_totals.get(path, 0) for path in required_paths},
    }, indent=2, sort_keys=True))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="/app/next_run_context/index.json")
    parser.add_argument("--summary", default="/app/workflow/NEXT_RUN_CONTEXT_SUMMARY.json")
    parser.add_argument("--trace", default="/app/workflow/next_run_file_trace.log")
    args = parser.parse_args()
    errors = verify(Path(args.index), Path(args.summary), Path(args.trace))
    if errors:
        print("NEXT_RUN_CONTEXT_ACCESS_VERIFICATION_FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("NEXT_RUN_CONTEXT_ACCESS_VERIFICATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def locked_command_wrapper_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")"

find_real_command() {
  local candidate
  local path_part
  local wrapper_real
  wrapper_real="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"
  IFS=':' read -ra path_parts <<< "${PATH:-}"
  for path_part in "${path_parts[@]}"; do
    [ -n "$path_part" ] || continue
    [ "$path_part" = "/app/locked_bin" ] && continue
    candidate="${path_part}/${name}"
    [ -x "$candidate" ] || continue
    [ "$(readlink -f "$candidate" 2>/dev/null || printf '%s' "$candidate")" = "$wrapper_real" ] && continue
    printf '%s\\n' "$candidate"
    return 0
  done
  return 1
}

if [ "${LOCKED_WANNIER_RUNNER_ACTIVE:-}" = "1" ]; then
  real_command="$(find_real_command || true)"
  if [ -z "${real_command:-}" ]; then
    echo "LOCKED_WORKFLOW_POLICY_ERROR: could not locate real ${name}" >&2
    exit 127
  fi
  exec "$real_command" "$@"
fi

mkdir -p /app/workflow
{
  printf '%s\tDENIED\t%s\t' "$(date -Is 2>/dev/null || date)" "$name"
  printf '%q ' "$@"
  printf '\\n'
} >> /app/workflow/locked_command_denials.log
echo "LOCKED_WORKFLOW_POLICY_DENIED: ${name} may only be run by /app/locked_wannier_runner.py" >&2
exit 126
"""


def locked_runner_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


APP = Path("/app")
MATERIAL_DIR = APP / "material"
WORKFLOW_DIR = APP / "workflow"
ARTIFACTS_DIR = APP / "artifacts"
RECIPE_REQUEST_PATH = WORKFLOW_DIR / "recipe_request.json"
LOCKED_RECIPE_PATH = WORKFLOW_DIR / "LOCKED_RECIPE.json"
NEXT_SUMMARY_PATH = WORKFLOW_DIR / "NEXT_RUN_CONTEXT_SUMMARY.json"
LOG_PATH = WORKFLOW_DIR / "locked_runner.log"
RUNNER_STATE_PATH = WORKFLOW_DIR / "locked_runner_state.json"


RECIPE_TABLE: dict[str, dict[str, Any]] = {
    "Al4Sc2": {
        "num_wann": 36,
        "num_bands": 51,
        "target_end": 36,
        "default_windows": [-50.0, 30.0, -50.0, 17.25],
        "window_bounds": {"dis_win_min": [-60.0, -35.0], "dis_win_max": [20.0, 40.0], "dis_froz_min": [-60.0, -35.0], "dis_froz_max": [15.0, 18.0]},
        "projection_variants": {
            "reference": ["Al:l=0;l=2", "Sc:l=0;l=2"],
            "fallback_1": ["Al:s;p", "Sc:s;p;d", "Sc:s:r=2"],
        },
        "allowed_modes": ["reference", "explicit"],
    },
    "Li4O6Si2": {
        "num_wann": 52,
        "num_bands": 84,
        "target_end": 52,
        "default_windows": [-42.0, 38.0, -40.0, 20.2],
        "window_bounds": {"dis_win_min": [-50.0, -30.0], "dis_win_max": [30.0, 45.0], "dis_froz_min": [-50.0, -30.0], "dis_froz_max": [18.0, 21.0]},
        "projection_variants": {
            "reference": ["Li:s;p", "Li:s", "O:s;p", "Si:s;p"],
            "fallback_1": ["Li:s;p", "O:s;p", "Si:s;p"],
        },
        "allowed_modes": ["reference", "explicit"],
    },
    "Mg2O10Ti4": {
        "num_wann": 88,
        "num_bands": 168,
        "target_end": 88,
        "default_windows": [-52.0, 45.0, -52.0, 19.5],
        "window_bounds": {"dis_win_min": [-60.0, -40.0], "dis_win_max": [35.0, 55.0], "dis_froz_min": [-60.0, -40.0], "dis_froz_max": [18.0, 20.0]},
        "projection_variants": {
            "reference": [],
            "fallback_1": ["Mg:s;p", "O:s;p", "Ti:s;p;d", "Ti:s:r=2"],
        },
        "allowed_modes": ["scdm", "reference", "explicit"],
        "default_mode": "scdm",
    },
    "Al18Co4": {
        "num_wann": 124,
        "num_bands": 183,
        "target_end": 124,
        "default_windows": [-86.0, 35.0, -86.0, 20.6],
        "window_bounds": {"dis_win_min": [-95.0, -70.0], "dis_win_max": [25.0, 45.0], "dis_froz_min": [-95.0, -70.0], "dis_froz_max": [18.0, 22.0]},
        "projection_variants": {
            "reference": ["Al:s;p", "Co:s;p;d", "Co:s;p:r=2"],
            "fallback_1": ["Al:s;p", "Co:s;p;d"],
        },
        "allowed_modes": ["reference", "explicit"],
    },
    "Si6Y10": {
        "num_wann": 154,
        "num_bands": 201,
        "target_end": 154,
        "default_windows": [-35.0, 30.0, -35.0, 22.242],
        "window_bounds": {"dis_win_min": [-45.0, -25.0], "dis_win_max": [25.0, 40.0], "dis_froz_min": [-45.0, -25.0], "dis_froz_max": [20.0, 23.0]},
        "projection_variants": {
            "reference": ["Si:s;p", "Y:s;p;d", "Y:s;p"],
            "fallback_1": ["Si:s;p", "Y:s;p;d"],
        },
        "allowed_modes": ["reference", "explicit"],
    },
}


def log(message: str) -> None:
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\\n")
    print(message, flush=True)


def configure_interrupt_policy() -> None:
    # DeepSeek was repeatedly sending Ctrl-C and rerunning the allowed runner.
    # Ignore terminal SIGINT so the runner and its Wannier/QE children keep ownership.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def read_runner_state() -> dict[str, Any] | None:
    if not RUNNER_STATE_PATH.is_file():
        return None
    try:
        data = json.loads(RUNNER_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unknown", "reason": "unreadable runner state"}
    return data if isinstance(data, dict) else {"status": "unknown", "reason": "invalid runner state"}


def write_runner_state(status: str, **extra: Any) -> None:
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "status": status,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    tmp_path = RUNNER_STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    tmp_path.replace(RUNNER_STATE_PATH)


def runner_state_process_alive(state: dict[str, Any]) -> bool:
    try:
        pid = int(state.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refuse_runner_rerun(state: dict[str, Any]) -> int:
    status = str(state.get("status") or "unknown")
    active = runner_state_process_alive(state)
    message = (
        "locked runner has already been started; refusing to rerun because "
        "reruns wipe workflow/run_dir and cause thrashing"
    )
    log(f"REFUSE_RERUN previous_status={status} active={active}")
    if not active and status != "success" and not (APP / "report.json").is_file():
        try:
            material = material_id()
        except Exception:
            material = "unknown"
        fail(material, message)
    print(f"LOCKED_WORKFLOW_POLICY_DENIED: {message}", file=sys.stderr)
    return 0 if status == "success" else 1


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_locked_recipe(recipe: dict[str, Any]) -> None:
    # Persist the current effective recipe after every accepted mutation.
    LOCKED_RECIPE_PATH.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )


def material_id() -> str:
    metadata = read_json(MATERIAL_DIR / "metadata.json")
    value = metadata.get("material_id") or metadata.get("formula")
    if not isinstance(value, str) or not value:
        raise ValueError("material metadata does not contain material_id/formula")
    return value


def validate_context(material: str) -> None:
    summary = read_json(NEXT_SUMMARY_PATH)
    if summary.get("target_material") != material:
        raise ValueError("NEXT_RUN_CONTEXT_SUMMARY.json target_material mismatch")
    if summary.get("bundle_path") != "/app/next_run_context/ALL_NEXT_RUN_RECOMMENDATIONS.md":
        raise ValueError("NEXT_RUN_CONTEXT_SUMMARY.json bundle_path mismatch")
    if summary.get("index_path") != "/app/next_run_context/index.json":
        raise ValueError("NEXT_RUN_CONTEXT_SUMMARY.json index_path mismatch")
    if summary.get("read_complete_bundle") is not True:
        raise ValueError("NEXT_RUN_CONTEXT_SUMMARY.json did not confirm complete bundle read")


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_recipe(material: str, request: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    if request.get("material_id") not in {None, material}:
        raise ValueError("recipe_request material_id does not match task material")
    if request.get("rerun_dft", False) is not False:
        raise ValueError("rerun_dft is not allowed in the locked workflow")
    if request.get("use_exclude_bands", False) is not False:
        raise ValueError("exclude_bands is not allowed in the locked workflow")

    num_wann = int(request.get("num_wann", table["num_wann"]))
    num_bands = int(request.get("num_bands", table["num_bands"]))
    target_end = int(request.get("target_dft_band_end", table["target_end"]))
    if num_wann != table["num_wann"] or num_bands != table["num_bands"] or target_end != table["target_end"]:
        raise ValueError("num_wann, num_bands, and target band end are fixed by the task")

    mode = str(request.get("projection_mode") or table.get("default_mode") or "reference")
    if mode == "reference":
        variant = "reference"
    elif mode == "explicit":
        variant = str(request.get("projection_variant") or "reference")
    elif mode == "scdm":
        variant = "reference"
    else:
        raise ValueError(f"unsupported projection_mode {mode!r}")
    if mode not in table["allowed_modes"]:
        raise ValueError(f"projection_mode {mode!r} is not allowed for {material}")
    if variant not in table["projection_variants"]:
        raise ValueError(f"projection_variant {variant!r} is not allowed for {material}")

    requested_windows = request.get("windows", {})
    if requested_windows is None:
        requested_windows = {}
    if not isinstance(requested_windows, dict):
        raise ValueError("windows must be a JSON object")
    dmin, dmax, fmin, fmax = table["default_windows"]
    windows = {
        "dis_win_min": finite_float(requested_windows.get("dis_win_min", dmin), "dis_win_min"),
        "dis_win_max": finite_float(requested_windows.get("dis_win_max", dmax), "dis_win_max"),
        "dis_froz_min": finite_float(requested_windows.get("dis_froz_min", fmin), "dis_froz_min"),
        "dis_froz_max": finite_float(requested_windows.get("dis_froz_max", fmax), "dis_froz_max"),
    }
    for key, value in windows.items():
        lo, hi = table["window_bounds"][key]
        if value < lo or value > hi:
            raise ValueError(f"{key}={value} is outside locked bounds [{lo}, {hi}]")
    if not (windows["dis_win_min"] <= windows["dis_froz_min"] <= windows["dis_froz_max"] <= windows["dis_win_max"]):
        raise ValueError("energy windows must satisfy dis_win_min <= dis_froz_min <= dis_froz_max <= dis_win_max")

    return {
        "material_id": material,
        "seedname": material,
        "num_wann": num_wann,
        "num_bands": num_bands,
        "target_dft_band_start": 1,
        "target_dft_band_end": target_end,
        "projection_mode": mode,
        "projection_variant": variant,
        "projections": list(table["projection_variants"][variant]),
        "windows": windows,
        "rerun_dft": False,
        "use_exclude_bands": False,
        "allowed_repairs": [
            "projection_count_fallback_variant_after_wannier90_pp_failure",
            "outer_window_expand_from_eig_for_fewer_states",
            "frozen_window_lower_from_eig_when_target_plus_one_is_frozen",
            "one_disentanglement_tolerance_relaxation_after_nonconvergence",
        ],
    }


def parse_nscf_input(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\\n".join(lines)
    nbnd_match = re.search(r"\\bnbnd\\s*=\\s*(\\d+)", text, flags=re.IGNORECASE)
    if not nbnd_match:
        raise ValueError("could not parse nbnd from nscf.in")

    atoms: list[tuple[str, float, float, float]] = []
    cell: list[list[float]] = []
    kpoints: list[list[float]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        upper = stripped.upper()
        if upper.startswith("ATOMIC_POSITIONS"):
            index += 1
            while index < len(lines):
                parts = lines[index].split()
                if not parts or parts[0].upper() in {"K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES", "OCCUPATIONS"} or lines[index].lstrip().startswith("&"):
                    break
                if len(parts) >= 4:
                    atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
                index += 1
            continue
        if upper.startswith("CELL_PARAMETERS"):
            for offset in range(1, 4):
                parts = lines[index + offset].split()
                cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
            index += 4
            continue
        if upper.startswith("K_POINTS") and "CRYSTAL" in upper:
            count = int(lines[index + 1].split()[0])
            for offset in range(count):
                parts = lines[index + 2 + offset].split()
                kpoints.append([float(parts[0]), float(parts[1]), float(parts[2])])
            index += 2 + count
            continue
        index += 1

    if not atoms:
        raise ValueError("could not parse ATOMIC_POSITIONS from nscf.in")
    if len(cell) != 3:
        raise ValueError("could not parse CELL_PARAMETERS from nscf.in")
    if not kpoints:
        raise ValueError("could not parse crystal K_POINTS from nscf.in")
    return {"nbnd": int(nbnd_match.group(1)), "atoms": atoms, "cell": cell, "kpoints": kpoints}


def infer_mp_grid(kpoints: list[list[float]]) -> list[int]:
    grid: list[int] = []
    for dim in range(3):
        values = sorted({round(point[dim] % 1.0, 10) for point in kpoints})
        grid.append(len(values))
    if grid[0] * grid[1] * grid[2] != len(kpoints):
        raise ValueError(f"could not infer rectangular mp_grid from {len(kpoints)} kpoints: {grid}")
    return grid


def write_win(path: Path, recipe: dict[str, Any], nscf: dict[str, Any], *, dis_conv_tol: str = "1.0d-8", conv_tol: str = "1.0d-8") -> None:
    windows = recipe["windows"]
    mp_grid = infer_mp_grid(nscf["kpoints"])
    lines: list[str] = [
        f"num_wann = {recipe['num_wann']}",
        f"num_bands = {recipe['num_bands']}",
        "num_iter = 1000",
        "dis_num_iter = 1000",
        f"conv_tol = {conv_tol}",
        f"dis_conv_tol = {dis_conv_tol}",
        "write_hr = .true.",
        "guiding_centres = .true.",
        f"mp_grid = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}",
        f"dis_win_min = {windows['dis_win_min']:.8f}",
        f"dis_win_max = {windows['dis_win_max']:.8f}",
        f"dis_froz_min = {windows['dis_froz_min']:.8f}",
        f"dis_froz_max = {windows['dis_froz_max']:.8f}",
    ]
    if recipe["projection_mode"] == "scdm":
        lines.extend(["auto_projections = .true.", "scdm_entanglement = erfc"])
    else:
        lines.append("begin projections")
        lines.extend(f"  {projection}" for projection in recipe["projections"])
        lines.append("end projections")
    lines.extend(["begin unit_cell_cart", "Ang"])
    lines.extend(f"  {row[0]: .12f} {row[1]: .12f} {row[2]: .12f}" for row in nscf["cell"])
    lines.extend(["end unit_cell_cart", "begin atoms_cart", "Ang"])
    lines.extend(f"  {atom[0]} {atom[1]: .12f} {atom[2]: .12f} {atom[3]: .12f}" for atom in nscf["atoms"])
    lines.extend(["end atoms_cart", "begin kpoints"])
    lines.extend(f"  {point[0]: .12f} {point[1]: .12f} {point[2]: .12f}" for point in nscf["kpoints"])
    lines.append("end kpoints")
    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def install_qe_save(run_dir: Path) -> None:
    candidates = [
        MATERIAL_DIR / "qe_save" / "out",
        MATERIAL_DIR / "qe_save",
        MATERIAL_DIR / "out",
        MATERIAL_DIR,
    ]
    for candidate in candidates:
        if (candidate / "aiida.save").is_dir():
            out_dir = run_dir / "out"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(candidate, out_dir, symlinks=True)
            return
    raise ValueError("no usable aiida.save tree found under material")


def copy_pseudos(run_dir: Path) -> None:
    pseudo_dir = MATERIAL_DIR / "pseudo"
    if not pseudo_dir.is_dir():
        return
    for pseudo in pseudo_dir.glob("*.UPF"):
        shutil.copy2(pseudo, run_dir / pseudo.name)


def run_command(argv: list[str], cwd: Path, log_name: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LOCKED_WANNIER_RUNNER_ACTIVE"] = "1"
    log(f"RUN {' '.join(argv)}")
    output_path = cwd / log_name
    with output_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
            raise TimeoutError(f"{' '.join(argv)} timed out after {timeout_sec} seconds")
    log(f"EXIT {return_code} {' '.join(argv)}")
    return subprocess.CompletedProcess(argv, return_code, "", output_path.read_text(encoding="utf-8", errors="replace"))


def run_wannier_final(seed: str, run_dir: Path, timeout_sec: int = 7200) -> int:
    env = os.environ.copy()
    env["LOCKED_WANNIER_RUNNER_ACTIVE"] = "1"
    log(f"RUN wannier90.x {seed}")
    output_path = run_dir / f"{seed}.wannier90.log"
    with output_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            ["wannier90.x", seed],
            cwd=run_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start = time.monotonic()
        while True:
            return_code = process.poll()
            if return_code is not None:
                log(f"EXIT {return_code} wannier90.x {seed}")
                return return_code
            if time.monotonic() - start > timeout_sec:
                process.kill()
                process.wait()
                raise TimeoutError(f"wannier90.x {seed} exceeded locked wall timeout")
            time.sleep(30)


def write_pw2wan(path: Path, seed: str, mode: str) -> None:
    extra = "  scdm_proj = .true.\\n" if mode == "scdm" else ""
    path.write_text(
        "&inputpp\\n"
        "  outdir = './out'\\n"
        "  prefix = 'aiida'\\n"
        f"  seedname = '{seed}'\\n"
        "  write_mmn = .true.\\n"
        "  write_amn = .true.\\n"
        "  write_eig = .true.\\n"
        f"{extra}"
        "/\\n",
        encoding="utf-8",
    )


def eig_by_k(path: Path) -> dict[int, dict[int, float]]:
    values: dict[int, dict[int, float]] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            band = int(parts[0])
            kpt = int(parts[1])
            energy = float(parts[2])
        except ValueError:
            continue
        values.setdefault(kpt, {})[band] = energy
    return values


def apply_eig_repairs(recipe: dict[str, Any], table: dict[str, Any], eig_path: Path) -> list[str]:
    repairs: list[str] = []
    values = eig_by_k(eig_path)
    if not values:
        return repairs
    target = recipe["target_dft_band_end"]
    target_energies = [bands[target] for bands in values.values() if target in bands]
    next_energies = [bands[target + 1] for bands in values.values() if target + 1 in bands]
    bounds = table["window_bounds"]
    windows = recipe["windows"]
    if target_energies:
        needed_outer = max(target_energies) + 1.0
        if needed_outer > windows["dis_win_max"]:
            new_value = min(bounds["dis_win_max"][1], needed_outer)
            if new_value > windows["dis_win_max"]:
                windows["dis_win_max"] = new_value
                repairs.append("expanded dis_win_max from eig target-band energy")
    if next_energies:
        safe_frozen = min(next_energies) - 0.03
        if safe_frozen < windows["dis_froz_max"]:
            new_value = max(bounds["dis_froz_max"][0], safe_frozen)
            if new_value < windows["dis_froz_max"]:
                windows["dis_froz_max"] = new_value
                repairs.append("lowered dis_froz_max below target+1 eig energy")
    return repairs


def collect_artifacts(seed: str, run_dir: Path, recipe: dict[str, Any], status: str, notes: list[str]) -> dict[str, Any]:
    attempt_dir = ARTIFACTS_DIR / "attempt_1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "win": f"{seed}.win",
        "wout": f"{seed}.wout",
        "eig": f"{seed}.eig",
        "chk": f"{seed}.chk",
        "nnkp": f"{seed}.nnkp",
        "hr": f"{seed}_hr.dat",
    }
    for filename in files.values():
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, attempt_dir / filename)
    for extra in [f"{seed}.amn", f"{seed}.mmn", f"{seed}.pw2wan"]:
        source = run_dir / extra
        if source.is_file():
            shutil.copy2(source, attempt_dir / extra)
    hr_exists = (attempt_dir / f"{seed}_hr.dat").is_file() and (attempt_dir / f"{seed}_hr.dat").stat().st_size > 0
    executed = status == "success" and hr_exists
    manifest = {
        "material_id": recipe["material_id"],
        "seedname": seed,
        "attempt": "attempt_1",
        "status": "success" if executed else status,
        "executed_successfully": bool(executed),
        "workflow_entrypoint": "workflow/run.sh",
        "target_dft_band_start": 1,
        "target_dft_band_end": recipe["target_dft_band_end"],
        "num_wann": recipe["num_wann"],
        "num_bands": recipe["num_bands"],
        "projections": recipe["projections"],
        "wannier_parameters": {
            "num_bands": recipe["num_bands"],
            "num_wann": recipe["num_wann"],
            **recipe["windows"],
        },
        "files": files,
        "dft_reference": {
            "target_dft_band_start": 1,
            "target_dft_band_end": recipe["target_dft_band_end"],
            "source": "material/qe_save/out/aiida.save",
        },
        "commands": [
            "wannier90.x -pp <seed>",
            "pw2wannier90.x -in <seed>.pw2wan",
            "wannier90.x <seed>",
        ],
        "notes": notes,
    }
    (attempt_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    report = {
        "status": manifest["status"],
        "task_complete": bool(executed),
        "executed_successfully": bool(executed),
        "material_id": recipe["material_id"],
        "seedname": seed,
        "run_manifest_path": "artifacts/attempt_1/run_manifest.json",
        "target_dft_band_start": 1,
        "target_dft_band_end": recipe["target_dft_band_end"],
        "num_wann": recipe["num_wann"],
        "num_bands": recipe["num_bands"],
        "notes": notes,
    }
    (APP / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    (APP / "REPORT.md").write_text(
        "# Locked Wannier Runner Report\\n\\n"
        f"- Material: {recipe['material_id']}\\n"
        f"- Status: {manifest['status']}\\n"
        f"- Projection mode: {recipe['projection_mode']} / {recipe['projection_variant']}\\n"
        f"- Windows: {json.dumps(recipe['windows'], sort_keys=True)}\\n"
        f"- Notes: {'; '.join(notes) if notes else 'none'}\\n",
        encoding="utf-8",
    )
    return manifest


def write_decisions(recipe: dict[str, Any], notes: list[str]) -> None:
    (WORKFLOW_DIR / "DECISIONS.md").write_text(
        "# Locked Workflow Decisions\\n\\n"
        "The agent was restricted to recipe proposal only. The runner authored and executed the workflow.\\n\\n"
        f"- Material: {recipe['material_id']}\\n"
        f"- num_wann/num_bands: {recipe['num_wann']} / {recipe['num_bands']}\\n"
        f"- Target DFT bands: 1-{recipe['target_dft_band_end']}\\n"
        f"- Projections: {recipe['projection_mode']} {recipe['projections']}\\n"
        f"- Energy windows: {json.dumps(recipe['windows'], sort_keys=True)}\\n"
        f"- Repairs: {'; '.join(notes) if notes else 'none'}\\n",
        encoding="utf-8",
    )


def fail(material: str, message: str) -> int:
    log(f"FAILED {message}")
    write_runner_state("failed", message=message)
    table = RECIPE_TABLE.get(material, {"num_wann": None, "num_bands": None, "target_end": None, "default_windows": [None, None, None, None]})
    recipe = {
        "material_id": material,
        "seedname": material,
        "num_wann": table.get("num_wann"),
        "num_bands": table.get("num_bands"),
        "target_dft_band_end": table.get("target_end"),
        "projections": [],
        "projection_mode": "none",
        "projection_variant": "none",
        "windows": {
            "dis_win_min": table.get("default_windows", [None, None, None, None])[0],
            "dis_win_max": table.get("default_windows", [None, None, None, None])[1],
            "dis_froz_min": table.get("default_windows", [None, None, None, None])[2],
            "dis_froz_max": table.get("default_windows", [None, None, None, None])[3],
        },
    }
    run_dir = WORKFLOW_DIR / "run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    collect_artifacts(material, run_dir, recipe, "failed", [message])
    return 1


def main() -> int:
    configure_interrupt_policy()
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_state = read_runner_state()
    if existing_state is not None:
        return refuse_runner_rerun(existing_state)
    LOG_PATH.write_text("", encoding="utf-8")
    write_runner_state("running")
    try:
        material = material_id()
        if material not in RECIPE_TABLE:
            return fail(material, "material is not in the locked recipe table")
        table = RECIPE_TABLE[material]
        validate_context(material)
        request = read_json(RECIPE_REQUEST_PATH)
        recipe = normalize_recipe(material, request, table)
        write_locked_recipe(recipe)
        notes: list[str] = []

        nscf_path = MATERIAL_DIR / "nscf" / "input" / "nscf.in"
        nscf = parse_nscf_input(nscf_path)
        if nscf["nbnd"] != recipe["num_bands"]:
            raise ValueError(f"nscf.in nbnd={nscf['nbnd']} does not match locked num_bands={recipe['num_bands']}")

        run_dir = WORKFLOW_DIR / "run_dir"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        install_qe_save(run_dir)
        copy_pseudos(run_dir)
        seed = recipe["seedname"]
        run_script = WORKFLOW_DIR / "run.sh"
        run_script.write_text(
            "#!/usr/bin/env bash\\nset -euo pipefail\\ncd /app/workflow/run_dir\\n"
            f"wannier90.x -pp {seed}\\n"
            f"pw2wannier90.x -in {seed}.pw2wan\\n"
            f"wannier90.x {seed}\\n",
            encoding="utf-8",
        )
        run_script.chmod(0o755)

        pp_ok = False
        variants_to_try = [recipe["projection_variant"]]
        if recipe["projection_variant"] != "fallback_1" and "fallback_1" in table["projection_variants"] and recipe["projection_mode"] != "scdm":
            variants_to_try.append("fallback_1")
        for variant in variants_to_try:
            recipe["projection_variant"] = variant
            recipe["projections"] = list(table["projection_variants"][variant])
            write_locked_recipe(recipe)
            write_win(run_dir / f"{seed}.win", recipe, nscf)
            result = run_command(["wannier90.x", "-pp", seed], run_dir, f"{seed}.pp.log", 600)
            if result.returncode == 0:
                pp_ok = True
                if variant != variants_to_try[0]:
                    notes.append(f"allowed repair: switched projections to {variant} after -pp failure")
                break
            notes.append(f"wannier90 -pp failed for projection variant {variant}")
        if not pp_ok:
            write_decisions(recipe, notes)
            collect_artifacts(seed, run_dir, recipe, "failed", notes)
            write_runner_state("failed", message="wannier90 -pp failed")
            return 1

        write_pw2wan(run_dir / f"{seed}.pw2wan", seed, recipe["projection_mode"])
        pw2 = run_command(["pw2wannier90.x", "-in", f"{seed}.pw2wan"], run_dir, f"{seed}.pw2wannier90.log", 3600)
        if pw2.returncode != 0:
            notes.append("pw2wannier90.x failed; no arbitrary DFT rerun is allowed")
            write_decisions(recipe, notes)
            collect_artifacts(seed, run_dir, recipe, "failed", notes)
            write_runner_state("failed", message="pw2wannier90.x failed")
            return 1

        repairs = apply_eig_repairs(recipe, table, run_dir / f"{seed}.eig")
        if repairs:
            notes.extend(f"allowed repair: {repair}" for repair in repairs)
            write_locked_recipe(recipe)
            write_win(run_dir / f"{seed}.win", recipe, nscf)
            run_command(["wannier90.x", "-pp", seed], run_dir, f"{seed}.pp_after_window_repair.log", 600)

        return_code = run_wannier_final(seed, run_dir)
        wout_text = (run_dir / f"{seed}.wout").read_text(encoding="utf-8", errors="replace") if (run_dir / f"{seed}.wout").is_file() else ""
        if return_code != 0 or not (run_dir / f"{seed}_hr.dat").is_file():
            if "disentanglement" in wout_text.lower() and "conver" in wout_text.lower():
                notes.append("allowed repair: relaxed disentanglement tolerance once after nonconvergence")
                write_win(run_dir / f"{seed}.win", recipe, nscf, dis_conv_tol="1.0d-4", conv_tol="1.0d-6")
                run_wannier_final(seed, run_dir)
        status = "success" if (run_dir / f"{seed}_hr.dat").is_file() and (run_dir / f"{seed}_hr.dat").stat().st_size > 0 else "failed"
        if status != "success":
            notes.append("final Hamiltonian was not produced")
        write_locked_recipe(recipe)
        write_decisions(recipe, notes)
        collect_artifacts(seed, run_dir, recipe, status, notes)
        log(f"COMPLETE status={status}")
        write_runner_state(status)
        return 0 if status == "success" else 1
    except Exception as exc:
        try:
            material = material_id()
        except Exception:
            material = "unknown"
        return fail(material, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
"""


def locked_runner_instruction_appendix(material: str) -> str:
    return f"""

# Locked DeepSeek Execution Contract

For this DeepSeek run, you are not the workflow executor. You are the recipe
proposer only.

After reading `/app/next_run_context/ALL_NEXT_RUN_RECOMMENDATIONS.md` and
writing `workflow/NEXT_RUN_CONTEXT_SUMMARY.json`, write exactly one proposed
recipe file:

`workflow/recipe_request.json`

The recipe must be valid JSON. Use only this schema:

```json
{{
  "material_id": "{material}",
  "num_wann": null,
  "num_bands": null,
  "target_dft_band_end": null,
  "projection_mode": "reference | explicit | scdm",
  "projection_variant": "reference | fallback_1",
  "windows": {{
    "dis_win_min": null,
    "dis_win_max": null,
    "dis_froz_min": null,
    "dis_froz_max": null
  }},
  "use_exclude_bands": false,
  "rerun_dft": false,
  "rationale": []
}}
```

Then run exactly:

```bash
/app/locked_wannier_runner.py
```

Do not run `wannier90.x`, `pw2wannier90.x`, `pw.x`, `rm`, `kill`, `pkill`, or
`killall` yourself. Do not edit `.win`, `.pw2wan`, generated Wannier files, or
files under `material/` yourself. Do not change kmesh, `num_wann`, `num_bands`,
target bands, projections, windows, or DFT inputs after deterministic preflight.
Do not send `C-c`, `Ctrl-C`, `SIGINT`, terminal interrupt keys, or EOF/control
keys to the runner or to a productive `wannier90.x`/`pw2wannier90.x` process
because it is taking time. Do not rerun `/app/locked_wannier_runner.py`; it is a
single-shot executor.

The locked runner owns all execution, generated files, deterministic checks,
and the only allowed repair table:
- switch to the locked fallback projection variant after a `wannier90.x -pp`
  parser/count failure;
- adjust only frozen/outer windows from the generated `.eig`;
- relax disentanglement tolerance once after a final Wannier90 nonconvergence;
- fail with reports instead of improvising any other change.

After `/app/locked_wannier_runner.py` exits, inspect `report.json` and return
the final JSON status. If the runner fails, report its failure. Do not attempt a
manual rescue path.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate num_wann-ordered Harbor DeepSeek runs with copied "
            "self-debug reports in each task context."
        )
    )
    parser.add_argument("--dataset", type=Path, default=harbor_generator.DEFAULT_DATASET)
    parser.add_argument(
        "--self-debug-reviews-root",
        type=Path,
        default=DEFAULT_SELF_DEBUG_REVIEWS_ROOT,
        help="Root containing per-material self_debug_report.md/json folders.",
    )
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        help="Material to run. Repeat to select multiple materials.",
    )
    parser.add_argument(
        "--target-success-runs",
        type=int,
        default=2,
        help=(
            "Top each selected material up to this many successful Harbor runs. "
            "Ignored when --target-runs is specified. Default: 2."
        ),
    )
    parser.add_argument(
        "--target-runs",
        type=int,
        default=None,
        help=(
            "Run each selected material this many total times, regardless of "
            "success or failure. This replaces --target-success-runs behavior."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of material jobs to run concurrently. Default: 1.",
    )
    parser.add_argument(
        "--success-wave-timeout-sec",
        type=int,
        default=4500,
        help="Wall timeout for each target-success wave. Default: 4500.",
    )
    parser.add_argument(
        "--success-wave-kill-after-sec",
        type=int,
        default=30,
        help="Seconds to wait after SIGTERM before SIGKILL. Default: 30.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first batch containing a failed Harbor run.",
    )
    parser.add_argument(
        "--materials-only",
        action="store_true",
        help="Print just the selected material names in num_wann order.",
    )
    parser.add_argument(
        "--no-docker-prune-after-batch",
        action="store_true",
        help="Do not print docker system/builder prune commands after each batch/wave.",
    )
    parser.add_argument(
        "--include-candidate-self-debug-reports",
        action="store_true",
        default=True,
        help="Also copy reports for candidate_material rows from --candidate-run-error-table.",
    )
    parser.add_argument(
        "--no-include-candidate-self-debug-reports",
        dest="include_candidate_self_debug_reports",
        action="store_false",
        help="Do not copy candidate_material reports from --candidate-run-error-table.",
    )
    parser.add_argument(
        "--candidate-self-debug-reports-only",
        action="store_true",
        default=True,
        help=(
            "Copy only candidate_material reports, not reports for the same "
            "target material. Implies --include-candidate-self-debug-reports."
        ),
    )
    parser.add_argument(
        "--include-same-material-self-debug-reports",
        dest="candidate_self_debug_reports_only",
        action="store_false",
        help="Also copy reports for the target material itself.",
    )
    parser.add_argument(
        "--candidate-run-error-table",
        type=Path,
        default=DEFAULT_CANDIDATE_RUN_ERROR_TABLE,
    )
    parser.add_argument(
        "--candidate-self-debug-reviews-root",
        type=Path,
        default=DEFAULT_SELF_DEBUG_REVIEWS_ROOT,
    )
    parser.add_argument(
        "--next-run-diagnoses",
        type=Path,
        default=None,
        help=(
            "Codex-reviewed next-run diagnosis markdown. In codex_self_review "
            "workflow, defaults to jobsDeepseekProTerminus2InstructionTest/"
            "codex_next_run_diagnoses.md."
        ),
    )
    return parser.parse_args()


def material_names_with_reports(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and self_debug_generator.self_debug_reports_for_material(path.name, root)
    }


def material_names_with_next_run_recommendations(path: Path) -> set[str]:
    """Find target materials that have per-run sections in the Codex diagnosis."""
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"^###\s+\d{4}\s+`([^_`\s]+)__", text, flags=re.MULTILINE))


def existing_run_counts(jobs_root: Path, valid_materials: set[str]) -> Counter[str]:
    """Count existing completed Harbor job directories, regardless of status."""
    counts: Counter[str] = Counter()
    if not jobs_root.is_dir():
        return counts

    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue

        material: str | None = None
        for diagnostics_path in job_dir.rglob("diagnostics.json"):
            if diagnostics_path.parent.name != "verifier":
                continue
            if "randprojections" in diagnostics_path.parts:
                continue
            try:
                data = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                candidate = data.get("material") or data.get("material_from_folder")
                if candidate in valid_materials:
                    material = candidate
                    break

        if material is None:
            for candidate in valid_materials:
                if job_dir.name.endswith(f"__{candidate}") or f"__{candidate}__" in job_dir.name:
                    material = candidate
                    break

        if material is not None:
            counts[material] += 1
    return counts


def candidate_materials_from_include_only_csv(path: Path) -> dict[str, list[str]]:
    """Read target_material,candidate_material rows in the exact include-only CSV."""
    if not path.is_file():
        raise SystemExit(f"candidate include-only CSV does not exist: {path}")

    import csv

    candidates_by_target: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "candidate_material" not in reader.fieldnames:
            raise SystemExit(f"{path} must contain a candidate_material column")
        target_column = (
            "target_material"
            if "target_material" in reader.fieldnames
            else "material"
            if "material" in reader.fieldnames
            else None
        )
        if target_column is None:
            raise SystemExit(f"{path} must contain target_material or material column")

        for row in reader:
            target = (row.get(target_column) or "").strip()
            candidate = (row.get("candidate_material") or "").strip()
            if not target and not candidate:
                continue
            if not target or not candidate:
                continue
            candidates = candidates_by_target.setdefault(target, [])
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates_by_target


def filter_candidate_reports(
    candidates_by_material: dict[str, list[str]],
    reviews_root: Path,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    filtered: dict[str, list[str]] = {}
    missing: list[str] = []
    for target, candidates in sorted(candidates_by_material.items()):
        for candidate in candidates:
            reports = self_debug_generator.self_debug_reports_for_material(
                candidate,
                reviews_root,
            )
            if not reports:
                missing.append(f"{target}->{candidate}")
                continue
            filtered.setdefault(target, []).append(candidate)
    no_usable_candidates = sorted(
        target
        for target in candidates_by_material
        if not filtered.get(target)
    )
    return filtered, missing, no_usable_candidates


def preview_list(values: list[str], *, limit: int = 12) -> str:
    if not values:
        return "none"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(shown) + suffix


def inject_next_run_trace_tools_into_dockerfile(dockerfile_text: str) -> str:
    denied_commands = " ".join(shlex.quote(name) for name in LOCKED_DENIED_COMMANDS)
    install_snippet = (
        "RUN if command -v apt-get >/dev/null 2>&1; then "
        "apt-get update && apt-get install -y --no-install-recommends strace && "
        "rm -rf /var/lib/apt/lists/*; "
        "elif command -v apk >/dev/null 2>&1; then apk add --no-cache strace; "
        "elif command -v dnf >/dev/null 2>&1; then dnf install -y strace && dnf clean all; "
        "else echo 'WARNING: no known package manager for installing strace' >&2; fi\n"
    )
    copy_snippet = (
        f"COPY {NEXT_RUN_TRACE_WRAPPER_NAME} /app/{NEXT_RUN_TRACE_WRAPPER_NAME}\n"
        f"COPY {NEXT_RUN_TRACE_VERIFIER_NAME} /app/{NEXT_RUN_TRACE_VERIFIER_NAME}\n"
        f"COPY {LOCKED_RUNNER_NAME} {LOCKED_RUNNER_APP_PATH}\n"
        f"COPY {LOCKED_COMMAND_WRAPPER_NAME} {LOCKED_COMMAND_WRAPPER_APP_PATH}\n"
        f"RUN chmod +x /app/{NEXT_RUN_TRACE_WRAPPER_NAME} "
        f"/app/{NEXT_RUN_TRACE_VERIFIER_NAME} "
        f"{LOCKED_RUNNER_APP_PATH} {LOCKED_COMMAND_WRAPPER_APP_PATH} && "
        f"mkdir -p {LOCKED_BIN_APP_DIR} && "
        f"for name in {denied_commands}; do "
        f"ln -sf {LOCKED_COMMAND_WRAPPER_APP_PATH} {LOCKED_BIN_APP_DIR}/$name; "
        "done && "
        "mkdir -p /app/workflow && "
        "printf 'ERROR: trace_wrapper_not_invoked\\n' > "
        "/app/workflow/next_run_file_trace.log\n"
    )
    profile_lines = " ".join(
        shlex.quote(line)
        for line in terminus_login_trace_profile_script().splitlines()
    )
    profile_hook = (
        "RUN mkdir -p /etc/profile.d && printf '%s\\n' "
        f"{profile_lines} > /etc/profile.d/harbor-agent-trace.sh\n"
    )

    if (
        "apt-get install -y --no-install-recommends strace" not in dockerfile_text
        and "apk add --no-cache strace" not in dockerfile_text
        and "dnf install -y strace" not in dockerfile_text
    ):
        lines = dockerfile_text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.lstrip().upper().startswith("FROM "):
                lines.insert(index + 1, install_snippet)
                dockerfile_text = "".join(lines)
                break
        else:
            dockerfile_text = install_snippet + dockerfile_text

    additions = ""
    if f"COPY {NEXT_RUN_TRACE_WRAPPER_NAME} /app/{NEXT_RUN_TRACE_WRAPPER_NAME}" not in dockerfile_text:
        additions += copy_snippet
    elif f"COPY {LOCKED_RUNNER_NAME} {LOCKED_RUNNER_APP_PATH}" not in dockerfile_text:
        additions += copy_snippet
    if "harbor-agent-trace.sh" not in dockerfile_text:
        additions += profile_hook
    if not additions:
        return dockerfile_text

    marker = "COPY material /app/material\n"
    if marker in dockerfile_text:
        return dockerfile_text.replace(marker, additions + marker, 1)
    return dockerfile_text + "\n" + additions


def install_next_run_trace_tools(tasks: list[tuple[int, str, Path]]) -> None:
    for _num_wann, material, task_dir in tasks:
        environment_dir = task_dir / "environment"
        wrapper_path = environment_dir / NEXT_RUN_TRACE_WRAPPER_NAME
        wrapper_path.write_text(next_run_trace_wrapper_script(), encoding="utf-8")
        wrapper_path.chmod(0o755)

        verifier_path = environment_dir / NEXT_RUN_TRACE_VERIFIER_NAME
        verifier_path.write_text(next_run_trace_verifier_script(), encoding="utf-8")
        verifier_path.chmod(0o755)

        runner_path = environment_dir / LOCKED_RUNNER_NAME
        runner_path.write_text(locked_runner_script(), encoding="utf-8")
        runner_path.chmod(0o755)

        command_wrapper_path = environment_dir / LOCKED_COMMAND_WRAPPER_NAME
        command_wrapper_path.write_text(locked_command_wrapper_script(), encoding="utf-8")
        command_wrapper_path.chmod(0o755)

        dockerfile_path = environment_dir / "Dockerfile"
        dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
        dockerfile_path.write_text(
            inject_next_run_trace_tools_into_dockerfile(dockerfile_text),
            encoding="utf-8",
        )

        instruction_path = task_dir / "instruction.md"
        instruction_text = instruction_path.read_text(encoding="utf-8")
        if "# Locked DeepSeek Execution Contract" not in instruction_text:
            instruction_text += locked_runner_instruction_appendix(material)
            instruction_path.write_text(instruction_text, encoding="utf-8")


def deepseek_harbor_args(cli: argparse.Namespace) -> argparse.Namespace:
    trace_wrapper_path = (
        NEXT_RUN_TRACE_WRAPPER_APP_PATH
        if WORKFLOW == "codex_self_review"
        else self_debug_generator.TRACE_WRAPPER_APP_PATH
    )
    trace_artifacts = (
        list(NEXT_RUN_TRACE_ARTIFACTS)
        if WORKFLOW == "codex_self_review"
        else []
    )
    return argparse.Namespace(
        dataset=cli.dataset,
        agent="terminus-2",
        model=MODEL,
        n_concurrent=1,
        batch_size=cli.batch_size,
        stop_on_error=cli.stop_on_error,
        docker_prune_after_batch=not cli.no_docker_prune_after_batch,
        docker_prune_after_material=False,
        delete_after_run=True,
        extra_arg=[
            "--agent-env",
            f"{self_debug_generator.DEFAULT_TRACE_AGENT_WRAPPER_ENV}="
            f"{trace_wrapper_path}",
            "--agent-timeout-multiplier",
            "1.1",
            "--max-retries",
            "2",
            "--retry-include",
            "AgentSetupTimeoutError",
            "--retry-include",
            "NonZeroAgentExitCodeError",
        ],
        artifact=trace_artifacts,
        no_default_artifacts=False,
        save_generated_qe_save=False,
        jobs_root=cli.jobs_root,
        target_success_runs=cli.target_success_runs,
        validate_new_success=False,
        max_attempts_per_needed_success=0,
        delete_failed_attempt_folders=False,
        success_wave_timeout_sec=cli.success_wave_timeout_sec,
        success_wave_kill_after_sec=cli.success_wave_kill_after_sec,
        success_roots=[cli.jobs_root],
        include_result_dir_name=[],
        least_success_first=False,
        no_gemini_cached_defaults=True,
        gemini_ipv4_first=False,
        no_gemini_run_timeout=True,
        no_gemini_host_network=True,
        no_gemini_file_trace=WORKFLOW == "codex_self_review",
        trace_agent_wrapper_env_name=self_debug_generator.DEFAULT_TRACE_AGENT_WRAPPER_ENV,
    )


def selected_materials(cli: argparse.Namespace) -> set[str]:
    explicit = {name.strip() for name in [*MATERIALS, *cli.material] if name.strip()}
    if explicit:
        return explicit

    if WORKFLOW == "codex_self_review":
        diagnoses_path = cli.next_run_diagnoses or DEFAULT_CODEX_NEXT_RUN_DIAGNOSES
        return material_names_with_next_run_recommendations(diagnoses_path)

    if cli.candidate_self_debug_reports_only:
        candidates = candidate_materials_from_include_only_csv(
            cli.candidate_run_error_table.expanduser().resolve()
        )
        return set(candidates)

    return material_names_with_reports(cli.self_debug_reviews_root)


def main() -> None:
    cli = parse_args()
    if WORKFLOW not in SUPPORTED_WORKFLOWS:
        raise SystemExit(
            f"Unsupported WORKFLOW={WORKFLOW!r}; choose one of "
            f"{', '.join(sorted(SUPPORTED_WORKFLOWS))}"
        )
    if cli.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if cli.target_runs is not None and cli.target_runs < 1:
        raise SystemExit("--target-runs must be at least 1")
    if cli.target_success_runs < 1:
        raise SystemExit("--target-success-runs must be at least 1")
    if cli.success_wave_timeout_sec < 1:
        raise SystemExit("--success-wave-timeout-sec must be at least 1")
    if cli.success_wave_kill_after_sec < 0:
        raise SystemExit("--success-wave-kill-after-sec cannot be negative")

    cli.dataset = cli.dataset.expanduser().resolve()
    cli.jobs_root = cli.jobs_root.expanduser().resolve()
    cli.self_debug_reviews_root = cli.self_debug_reviews_root.expanduser().resolve()
    cli.candidate_run_error_table = cli.candidate_run_error_table.expanduser().resolve()
    cli.candidate_self_debug_reviews_root = (
        cli.candidate_self_debug_reviews_root.expanduser().resolve()
    )
    if cli.next_run_diagnoses is not None:
        cli.next_run_diagnoses = cli.next_run_diagnoses.expanduser().resolve()

    self_debug_generator.SELF_DEBUG_REVIEWS_ROOT = cli.self_debug_reviews_root

    if WORKFLOW == "codex_self_review":
        cli.include_candidate_self_debug_reports = False
        cli.candidate_self_debug_reports_only = False
        if cli.next_run_diagnoses is None:
            cli.next_run_diagnoses = DEFAULT_CODEX_NEXT_RUN_DIAGNOSES
        if not cli.next_run_diagnoses.is_file():
            raise SystemExit(f"Codex next-run diagnosis file does not exist: {cli.next_run_diagnoses}")

    if cli.candidate_self_debug_reports_only:
        cli.include_candidate_self_debug_reports = True

    include_same_material_reports = (
        WORKFLOW != "codex_self_review"
        and not cli.candidate_self_debug_reports_only
    )

    candidate_materials_by_material = None
    if cli.include_candidate_self_debug_reports:
        candidate_materials_by_material = candidate_materials_from_include_only_csv(
            cli.candidate_run_error_table
        )

    requested = selected_materials(cli)
    if not requested:
        if WORKFLOW == "codex_self_review":
            raise SystemExit(
                "No materials selected. Add names to MATERIALS, pass --material, "
                f"or add per-material sections to {cli.next_run_diagnoses}."
            )
        raise SystemExit(
            "No materials selected. Add names to MATERIALS, pass --material, "
            f"or create reports under {cli.self_debug_reviews_root}."
        )

    tasks = harbor_generator.dataset_tasks(cli.dataset, include_materials=requested)
    found = {material for _num_wann, material, _source in tasks}
    missing_dataset_materials = sorted(requested - found)

    skipped_missing_target_reports: list[str] = []
    if include_same_material_reports:
        with_reports = material_names_with_reports(cli.self_debug_reviews_root)
        skipped_missing_target_reports = sorted(found - with_reports)
        tasks = [
            task
            for task in tasks
            if task[1] not in set(skipped_missing_target_reports)
        ]
        found = {material for _num_wann, material, _source in tasks}

    skipped_missing_candidate_links: list[str] = []
    skipped_no_usable_candidate_materials: list[str] = []
    if candidate_materials_by_material is not None:
        candidate_materials_by_material = {
            material: candidates
            for material, candidates in candidate_materials_by_material.items()
            if material in found
        }
        (
            candidate_materials_by_material,
            skipped_missing_candidate_links,
            skipped_no_usable_candidate_materials,
        ) = filter_candidate_reports(
            candidate_materials_by_material,
            cli.candidate_self_debug_reviews_root,
        )
        if cli.candidate_self_debug_reports_only:
            skipped_no_usable_candidate_materials = sorted(
                set(skipped_no_usable_candidate_materials)
                | (found - set(candidate_materials_by_material))
            )
            skipped_no_usable_set = set(skipped_no_usable_candidate_materials)
            tasks = [
                task
                for task in tasks
                if task[1] not in skipped_no_usable_set
            ]
            found = {material for _num_wann, material, _source in tasks}

    if cli.materials_only:
        print(" ".join(material for _num_wann, material, _source in tasks))
        return

    args = deepseek_harbor_args(cli)
    skipped_materials = sorted(
        set(missing_dataset_materials)
        | set(skipped_missing_target_reports)
        | set(skipped_no_usable_candidate_materials)
    )
    requested_run_slots_skipped = (
        len(skipped_materials) * cli.target_runs
        if cli.target_runs is not None
        else len(skipped_materials) * cli.target_success_runs
    )
    print("# DeepSeek self-debug context skip summary")
    print(f"# Workflow: {WORKFLOW}")
    if cli.next_run_diagnoses is not None:
        print(f"# Codex next-run diagnoses: {cli.next_run_diagnoses}")
    print(f"# Target materials skipped: {len(skipped_materials)}")
    print(f"# Requested run slots skipped: {requested_run_slots_skipped}")
    print(f"# Missing dataset target materials: {len(missing_dataset_materials)}")
    print(f"# Missing same-material report targets: {len(skipped_missing_target_reports)}")
    print(f"# Candidate links skipped for missing report pairs: {len(skipped_missing_candidate_links)}")
    print(f"# Target materials skipped with no usable candidate reports: {len(skipped_no_usable_candidate_materials)}")
    print(f"# Skipped target preview: {preview_list(skipped_materials)}")
    print(f"# Skipped candidate-link preview: {preview_list(skipped_missing_candidate_links)}")
    print(': "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running}"')
    print(
        'export OPENAI_BASE_URL="${OPENAI_BASE_URL:-'
        + DEFAULT_DEEPSEEK_BASE_URL
        + '}"'
    )

    repeats_by_material: dict[str, int] | None = None
    if cli.target_runs is not None:
        counts = existing_run_counts(cli.jobs_root, valid_materials=found)
        repeats_by_material = {}
        pending_tasks = []
        for task in tasks:
            _num_wann, material, _source = task
            existing = counts[material]
            needed = max(0, cli.target_runs - existing)
            print(f"# {material}: existing={existing}, target={cli.target_runs}, new={needed}")
            if needed:
                repeats_by_material[material] = needed
                pending_tasks.append(task)
        tasks = pending_tasks
        args.target_success_runs = None
    else:
        excluded = harbor_generator.DEFAULT_EXCLUDED_RESULT_DIR_NAMES
        success_counts = self_debug_generator.successful_run_counts(
            [cli.jobs_root],
            valid_materials=found,
            excluded_dir_names=excluded | {"case_files"},
        )
        tasks = [
            task
            for task in tasks
            if success_counts[task[1]] < cli.target_success_runs
        ]

    if not tasks:
        print("# Every selected material already has the requested number of runs.")
        print("true")
        return

    augmented_dataset, augmented_tasks = self_debug_generator.materialize_self_debug_context_dataset(
        cli.dataset,
        tasks,
        include_same_material_reports=include_same_material_reports,
        candidate_materials_by_material=candidate_materials_by_material,
        candidate_self_debug_reviews_root=(
            cli.candidate_self_debug_reviews_root
            if cli.include_candidate_self_debug_reports
            else None
        ),
        next_run_diagnoses_path=cli.next_run_diagnoses,
    )
    if WORKFLOW == "codex_self_review":
        install_next_run_trace_tools(augmented_tasks)
    args.dataset = augmented_dataset
    if repeats_by_material is not None:
        augmented_tasks = [
            task
            for task in augmented_tasks
            for _repeat in range(repeats_by_material[task[1]])
        ]

    if cli.target_runs is not None:
        self_debug_generator.print_ordered_commands(args, augmented_tasks)
    else:
        self_debug_generator.print_target_success_loop(args, augmented_tasks)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
