#!/usr/bin/env python3
"""Print Harbor commands for DeepSeek recipe-only materials runs.

This keeps the input surface from generate_harbor_qwen_materials_command.py:
edit MATERIALS or pass --dataset/--target-runs/--batch-size/--stop-on-error.

The generated tasks follow the controlled DeepSeek execution path: Terminus and
DeepSeek may inspect the original task/material inputs and write a recipe JSON;
a deterministic verifier hook runs Wannier90 afterward. Unlike the self-debug
context generator, this script does not copy prior recommendations and does not
embed per-material projection/window answers in the runner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import sys
import tomllib

import generate_harbor_num_wann_order_command as harbor_generator
import generate_harbor_deepseek_self_debug_context_command as controlled


MATERIALS = [
    "Al18Co4",
    "Al4Sc2",
    "Li4O6Si2",
    "Si6Y10",
    "Mg2O10Ti4",
]

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "openai/deepseek-v4-pro"
DEFAULT_JOBS_ROOT = harbor_generator.ROOT / "jobsDeepseekProTerminus2Controlled"
DEFAULT_AUGMENTED_DATASET_PARENT = harbor_generator.ROOT / "harbor_datasets"

LOCKED_RUNNER_NAME = controlled.LOCKED_RUNNER_NAME
LOCKED_RUNNER_APP_PATH = controlled.LOCKED_RUNNER_APP_PATH
LOCKED_COMMAND_WRAPPER_NAME = controlled.LOCKED_COMMAND_WRAPPER_NAME
LOCKED_COMMAND_WRAPPER_APP_PATH = controlled.LOCKED_COMMAND_WRAPPER_APP_PATH
LOCKED_BIN_APP_DIR = controlled.LOCKED_BIN_APP_DIR
LOCKED_DENIED_COMMANDS = controlled.LOCKED_DENIED_COMMANDS
LOCKED_RUNNER_VERIFIER_HOOK_MARKER = controlled.LOCKED_RUNNER_VERIFIER_HOOK_MARKER

DEFAULT_RECIPE_AGENT_TIMEOUT_SEC = 1200
DEFAULT_SUCCESS_WAVE_TIMEOUT_SEC = 4800
CONTROLLED_ARTIFACTS = [
    "/app/workflow/recipe_request.json",
    "/app/workflow/LOCKED_RECIPE.json",
    "/app/workflow/DECISIONS.md",
    "/app/workflow/locked_runner.log",
    "/app/workflow/locked_runner_state.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate num_wann-ordered Harbor runs using DeepSeek V4 Pro with "
            "Terminus recipe-only planning and a locked verifier runner."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=harbor_generator.DEFAULT_DATASET,
    )
    parser.add_argument(
        "--target-runs",
        type=int,
        default=None,
        help=(
            "Run each material this many times, regardless of success or failure. "
            "When specified, this replaces --target-success-runs behavior."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of material jobs to run concurrently (default: 1).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first batch containing a failed Harbor run.",
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=DEFAULT_JOBS_ROOT,
        help=f"Harbor jobs root (default: {DEFAULT_JOBS_ROOT}).",
    )
    parser.add_argument(
        "--recipe-agent-timeout-sec",
        type=int,
        default=DEFAULT_RECIPE_AGENT_TIMEOUT_SEC,
        help=(
            "Agent timeout for DeepSeek's recipe-writing phase. The verifier "
            f"runner executes after that phase. Default: {DEFAULT_RECIPE_AGENT_TIMEOUT_SEC}."
        ),
    )
    parser.add_argument(
        "--success-wave-timeout-sec",
        type=int,
        default=DEFAULT_SUCCESS_WAVE_TIMEOUT_SEC,
        help=f"Wall timeout for each generated Harbor run. Default: {DEFAULT_SUCCESS_WAVE_TIMEOUT_SEC}.",
    )
    parser.add_argument(
        "--success-wave-kill-after-sec",
        type=int,
        default=30,
        help="Seconds to wait after SIGTERM before SIGKILL. Default: 30.",
    )
    return parser.parse_args()


def existing_run_counts(jobs_root: Path, valid_materials: set[str]) -> Counter[str]:
    """Count existing completed Harbor runs, regardless of verifier status."""
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
                if (
                    job_dir.name.endswith(f"__{candidate}")
                    or f"__{candidate}__" in job_dir.name
                ):
                    material = candidate
                    break

        if material is not None:
            counts[material] += 1

    return counts


def task_timeout_sec(task_dir: Path, section: str) -> int:
    task_toml = task_dir / "task.toml"
    data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    timeout_sec = data.get(section, {}).get("timeout_sec")
    if type(timeout_sec) is not int or timeout_sec < 1:
        raise ValueError(
            f"{task_toml} must define [{section}].timeout_sec as a positive integer"
        )
    return timeout_sec


def common_task_timeout_sec(
    tasks: list[tuple[int, str, Path]],
    section: str,
) -> int:
    timeouts = {
        task_timeout_sec(task_dir, section)
        for _num_wann, _material, task_dir in tasks
    }
    if len(timeouts) != 1:
        raise ValueError(
            f"selected tasks have different [{section}].timeout_sec values: "
            f"{sorted(timeouts)}"
        )
    return timeouts.pop()


def generic_locked_runner_script() -> str:
    return r"""#!/usr/bin/env python3
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
INSTRUCTION_PATH = APP / "instruction.md"
RECIPE_REQUEST_PATH = WORKFLOW_DIR / "recipe_request.json"
LOCKED_RECIPE_PATH = WORKFLOW_DIR / "LOCKED_RECIPE.json"
LOG_PATH = WORKFLOW_DIR / "locked_runner.log"
RUNNER_STATE_PATH = WORKFLOW_DIR / "locked_runner_state.json"
RUNNER_EXECUTOR_ENV = "HARBOR_LOCKED_RUNNER_EXECUTOR"
RUNNER_EXECUTOR_VALUE = "harbor_verifier"


def log(message: str) -> None:
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")
    print(message, flush=True)


def configure_interrupt_policy() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_runner_state(status: str, **extra: Any) -> None:
    state = {
        "status": status,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    tmp_path = RUNNER_STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(RUNNER_STATE_PATH)


def read_runner_state() -> dict[str, Any] | None:
    if not RUNNER_STATE_PATH.is_file():
        return None
    try:
        data = json.loads(RUNNER_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unknown", "reason": "unreadable runner state"}
    return data if isinstance(data, dict) else {"status": "unknown", "reason": "invalid runner state"}


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
    print(f"LOCKED_WORKFLOW_POLICY_DENIED: {message}", file=sys.stderr)
    return 0 if status == "success" else 1


def material_id() -> str:
    metadata = read_json(MATERIAL_DIR / "metadata.json")
    value = metadata.get("material_id") or metadata.get("formula")
    if not isinstance(value, str) or not value:
        raise ValueError("material metadata does not contain material_id/formula")
    return value


def instruction_text() -> str:
    if not INSTRUCTION_PATH.is_file():
        return ""
    return INSTRUCTION_PATH.read_text(encoding="utf-8", errors="replace")


def expected_from_instruction() -> dict[str, int | None]:
    text = instruction_text()
    num_wann = None
    num_bands = None
    target_end = None
    match = re.search(r"\bnum_wann\s*=\s*(\d+)\b", text)
    if match:
        num_wann = int(match.group(1))
    match = re.search(r"\bnum_bands\s*=\s*(\d+)\b", text)
    if match:
        num_bands = int(match.group(1))
    match = re.search(r"Target DFT bands\s*`?1\s*-\s*(\d+)`?", text, flags=re.IGNORECASE)
    if match:
        target_end = int(match.group(1))
    if target_end is None:
        match = re.search(r"target(?:ed)?(?:\s+DFT)?\s+bands?\s+1\s*[-:]\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            target_end = int(match.group(1))
    if target_end is None:
        target_end = num_wann
    return {"num_wann": num_wann, "num_bands": num_bands, "target_end": target_end}


def parse_nscf_input(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\n".join(lines)
    nbnd_match = re.search(r"\bnbnd\s*=\s*(\d+)", text, flags=re.IGNORECASE)
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


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_recipe(material: str, request: dict[str, Any], expected: dict[str, int | None], nscf: dict[str, Any]) -> dict[str, Any]:
    if request.get("material_id") not in {None, material}:
        raise ValueError("recipe_request material_id does not match task material")
    if request.get("rerun_dft", False) is not False:
        raise ValueError("rerun_dft is not allowed in the locked workflow")
    if request.get("use_exclude_bands", False) is not False:
        raise ValueError("exclude_bands is not allowed in the locked workflow")

    num_wann = int(request.get("num_wann"))
    num_bands = int(request.get("num_bands"))
    target_end = int(request.get("target_dft_band_end"))
    if expected["num_wann"] is not None and num_wann != expected["num_wann"]:
        raise ValueError(f"num_wann={num_wann} does not match task num_wann={expected['num_wann']}")
    if expected["num_bands"] is not None and num_bands != expected["num_bands"]:
        raise ValueError(f"num_bands={num_bands} does not match task num_bands={expected['num_bands']}")
    if expected["target_end"] is not None and target_end != expected["target_end"]:
        raise ValueError(f"target_dft_band_end={target_end} does not match task target_end={expected['target_end']}")
    if num_bands != nscf["nbnd"]:
        raise ValueError(f"num_bands={num_bands} does not match nscf.in nbnd={nscf['nbnd']}")
    if num_wann < 1 or num_wann > num_bands:
        raise ValueError("num_wann must be between 1 and num_bands")
    if target_end < 1 or target_end > num_bands:
        raise ValueError("target_dft_band_end must be between 1 and num_bands")

    projections = request.get("projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError("projections must be a non-empty JSON list")
    normalized_projections: list[str] = []
    for item in projections:
        if not isinstance(item, str):
            raise ValueError("every projection must be a string")
        projection = item.strip()
        if not projection:
            raise ValueError("projection strings cannot be empty")
        if len(projection) > 160:
            raise ValueError(f"projection line is too long: {projection[:80]!r}")
        if re.search(r"random|placeholder|dummy", projection, flags=re.IGNORECASE):
            raise ValueError(f"projection line looks non-deterministic or placeholder-like: {projection!r}")
        normalized_projections.append(projection)

    requested_windows = request.get("windows")
    if not isinstance(requested_windows, dict):
        raise ValueError("windows must be a JSON object")
    windows = {
        "dis_win_min": finite_float(requested_windows.get("dis_win_min"), "dis_win_min"),
        "dis_win_max": finite_float(requested_windows.get("dis_win_max"), "dis_win_max"),
        "dis_froz_min": finite_float(requested_windows.get("dis_froz_min"), "dis_froz_min"),
        "dis_froz_max": finite_float(requested_windows.get("dis_froz_max"), "dis_froz_max"),
    }
    for key, value in windows.items():
        if value < -250.0 or value > 250.0:
            raise ValueError(f"{key}={value} is outside broad sanity bounds [-250, 250] eV")
    if not (windows["dis_win_min"] <= windows["dis_froz_min"] <= windows["dis_froz_max"] <= windows["dis_win_max"]):
        raise ValueError("energy windows must satisfy dis_win_min <= dis_froz_min <= dis_froz_max <= dis_win_max")

    seed = str(request.get("seedname") or material)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", seed):
        raise ValueError(f"unsafe seedname: {seed!r}")

    return {
        "material_id": material,
        "seedname": seed,
        "num_wann": num_wann,
        "num_bands": num_bands,
        "target_dft_band_start": 1,
        "target_dft_band_end": target_end,
        "projections": normalized_projections,
        "windows": windows,
        "rerun_dft": False,
        "use_exclude_bands": False,
        "rationale": request.get("rationale") if isinstance(request.get("rationale"), list) else [],
    }


def write_locked_recipe(recipe: dict[str, Any]) -> None:
    LOCKED_RECIPE_PATH.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_win(path: Path, recipe: dict[str, Any], nscf: dict[str, Any]) -> None:
    windows = recipe["windows"]
    mp_grid = infer_mp_grid(nscf["kpoints"])
    lines: list[str] = [
        f"num_wann = {recipe['num_wann']}",
        f"num_bands = {recipe['num_bands']}",
        "num_iter = 1000",
        "dis_num_iter = 1000",
        "conv_tol = 1.0d-8",
        "dis_conv_tol = 1.0d-8",
        "write_hr = .true.",
        "guiding_centres = .true.",
        f"mp_grid = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}",
        f"dis_win_min = {windows['dis_win_min']:.8f}",
        f"dis_win_max = {windows['dis_win_max']:.8f}",
        f"dis_froz_min = {windows['dis_froz_min']:.8f}",
        f"dis_froz_max = {windows['dis_froz_max']:.8f}",
        "begin projections",
    ]
    lines.extend(f"  {projection}" for projection in recipe["projections"])
    lines.extend(["end projections", "begin unit_cell_cart", "Ang"])
    lines.extend(f"  {row[0]: .12f} {row[1]: .12f} {row[2]: .12f}" for row in nscf["cell"])
    lines.extend(["end unit_cell_cart", "begin atoms_cart", "Ang"])
    lines.extend(f"  {atom[0]} {atom[1]: .12f} {atom[2]: .12f} {atom[3]: .12f}" for atom in nscf["atoms"])
    lines.extend(["end atoms_cart", "begin kpoints"])
    lines.extend(f"  {point[0]: .12f} {point[1]: .12f} {point[2]: .12f}" for point in nscf["kpoints"])
    lines.append("end kpoints")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pw2wan(path: Path, seed: str) -> None:
    path.write_text(
        "&inputpp\n"
        "  outdir = './out'\n"
        "  prefix = 'aiida'\n"
        f"  seedname = '{seed}'\n"
        "  write_mmn = .true.\n"
        "  write_amn = .true.\n"
        "  write_eig = .true.\n"
        "/\n",
        encoding="utf-8",
    )


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
            process.wait()
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
        last_heartbeat = start
        while True:
            return_code = process.poll()
            if return_code is not None:
                log(f"EXIT {return_code} wannier90.x {seed}")
                return return_code
            now = time.monotonic()
            if now - last_heartbeat >= 60:
                wout_path = run_dir / f"{seed}.wout"
                if wout_path.is_file():
                    try:
                        lines = wout_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        tail = lines[-1].strip() if lines else ""
                        size = wout_path.stat().st_size
                        log(f"STILL_RUNNING wannier90.x {seed} elapsed_sec={int(now - start)} wout_size={size} last_wout_line={tail[:180]!r}")
                    except OSError as exc:
                        log(f"STILL_RUNNING wannier90.x {seed} elapsed_sec={int(now - start)} wout_read_error={exc}")
                else:
                    log(f"STILL_RUNNING wannier90.x {seed} elapsed_sec={int(now - start)} wout_missing=true")
                last_heartbeat = now
            if now - start > timeout_sec:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TimeoutError(f"wannier90.x {seed} exceeded locked wall timeout")
            time.sleep(30)


def collect_artifacts(seed: str, run_dir: Path, recipe: dict[str, Any], status: str, notes: list[str]) -> None:
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
    for extra in [f"{seed}.amn", f"{seed}.mmn", f"{seed}.pw2wan", f"{seed}.pp.log", f"{seed}.pw2wannier90.log", f"{seed}.wannier90.log"]:
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
    (attempt_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    (APP / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (APP / "REPORT.md").write_text(
        "# Locked Wannier Runner Report\n\n"
        f"- Material: {recipe['material_id']}\n"
        f"- Status: {manifest['status']}\n"
        f"- Projections: {json.dumps(recipe['projections'])}\n"
        f"- Windows: {json.dumps(recipe['windows'], sort_keys=True)}\n"
        f"- Notes: {'; '.join(notes) if notes else 'none'}\n",
        encoding="utf-8",
    )


def write_decisions(recipe: dict[str, Any], notes: list[str]) -> None:
    rationale = recipe.get("rationale") or []
    rationale_text = "\n".join(f"- {item}" for item in rationale if isinstance(item, str))
    (WORKFLOW_DIR / "DECISIONS.md").write_text(
        "# Locked Workflow Decisions\n\n"
        "DeepSeek proposed the recipe. The locked runner authored and executed the workflow from that recipe.\n\n"
        f"- Material: {recipe['material_id']}\n"
        f"- num_wann/num_bands: {recipe['num_wann']} / {recipe['num_bands']}\n"
        f"- Target DFT bands: 1-{recipe['target_dft_band_end']}\n"
        f"- Projections: {json.dumps(recipe['projections'])}\n"
        f"- Energy windows: {json.dumps(recipe['windows'], sort_keys=True)}\n"
        f"- Runner notes: {'; '.join(notes) if notes else 'none'}\n\n"
        "## Agent Rationale\n\n"
        f"{rationale_text if rationale_text else '- none supplied'}\n",
        encoding="utf-8",
    )


def fail(material: str, message: str) -> int:
    log(f"FAILED {message}")
    write_runner_state("failed", message=message)
    recipe = {
        "material_id": material,
        "seedname": material,
        "num_wann": None,
        "num_bands": None,
        "target_dft_band_end": None,
        "projections": [],
        "windows": {
            "dis_win_min": None,
            "dis_win_max": None,
            "dis_froz_min": None,
            "dis_froz_max": None,
        },
        "rationale": [],
    }
    run_dir = WORKFLOW_DIR / "run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    collect_artifacts(material, run_dir, recipe, "failed", [message])
    return 1


def main() -> int:
    if os.environ.get(RUNNER_EXECUTOR_ENV) != RUNNER_EXECUTOR_VALUE:
        print(
            "LOCKED_WORKFLOW_POLICY_DENIED: locked_wannier_runner.py is deferred "
            "to Harbor's verifier and may not be run directly by the agent",
            file=sys.stderr,
        )
        return 126
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
        nscf = parse_nscf_input(MATERIAL_DIR / "nscf" / "input" / "nscf.in")
        expected = expected_from_instruction()
        request = read_json(RECIPE_REQUEST_PATH)
        recipe = normalize_recipe(material, request, expected, nscf)
        write_locked_recipe(recipe)
        notes: list[str] = []

        run_dir = WORKFLOW_DIR / "run_dir"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        install_qe_save(run_dir)
        copy_pseudos(run_dir)
        seed = recipe["seedname"]
        write_win(run_dir / f"{seed}.win", recipe, nscf)
        write_pw2wan(run_dir / f"{seed}.pw2wan", seed)

        run_script = WORKFLOW_DIR / "run.sh"
        run_script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\ncd /app/workflow/run_dir\n"
            f"wannier90.x -pp {shlex_quote(seed)}\n"
            f"pw2wannier90.x -in {shlex_quote(seed + '.pw2wan')}\n"
            f"wannier90.x {shlex_quote(seed)}\n",
            encoding="utf-8",
        )
        run_script.chmod(0o755)

        result = run_command(["wannier90.x", "-pp", seed], run_dir, f"{seed}.pp.log", 600)
        if result.returncode != 0:
            notes.append("wannier90.x -pp failed for the proposed projection recipe")
            write_decisions(recipe, notes)
            collect_artifacts(seed, run_dir, recipe, "failed", notes)
            write_runner_state("failed", message="wannier90 -pp failed")
            return 1

        pw2 = run_command(["pw2wannier90.x", "-in", f"{seed}.pw2wan"], run_dir, f"{seed}.pw2wannier90.log", 3600)
        if pw2.returncode != 0:
            notes.append("pw2wannier90.x failed for the proposed recipe")
            write_decisions(recipe, notes)
            collect_artifacts(seed, run_dir, recipe, "failed", notes)
            write_runner_state("failed", message="pw2wannier90.x failed")
            return 1

        try:
            return_code = run_wannier_final(seed, run_dir)
        except TimeoutError as exc:
            notes.append(str(exc))
            write_decisions(recipe, notes)
            collect_artifacts(seed, run_dir, recipe, "failed", notes)
            log("COMPLETE status=failed")
            write_runner_state("failed", message=str(exc))
            return 1

        status = "success" if return_code == 0 and (run_dir / f"{seed}_hr.dat").is_file() and (run_dir / f"{seed}_hr.dat").stat().st_size > 0 else "failed"
        if status != "success":
            notes.append("final Hamiltonian was not produced")
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


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def locked_runner_instruction_appendix(material: str) -> str:
    return f"""

# Locked DeepSeek Execution Contract

For this DeepSeek run, you are not the workflow executor. You are the recipe
proposer only. YOU HAVE {DEFAULT_RECIPE_AGENT_TIMEOUT_SEC} SECONDS to propose a recipe and write it to `workflow/recipe_request.json`.

Use the original task instructions and the supplied files under `/app/material`
to decide the Wannierisation recipe yourself. You may inspect compact metadata/log snippets, but keep terminal
output small.

Write exactly one proposed recipe file:

`workflow/recipe_request.json`

The recipe must be valid JSON. Use only this schema:

```json
{{
  "material_id": "{material}",
  "seedname": "{material}",
  "num_wann": null,
  "num_bands": null,
  "target_dft_band_end": null,
  "projections": [],
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

`use_exclude_bands` must always be false. DO NOT SET IT TO TRUE. 
All four window fields must be numeric. Do not leave any window value as null.

`projections` must contain the actual Wannier90 projection lines you choose,
for example strings in the syntax you would place between `begin projections`
and `end projections`. The runner will not choose projections or windows for
you. Use standard Wannier90 projection syntax, such as `Si:l=0;l=1` or
`Y:l=0;l=1;l=2`. Use coordinate-centered projections only as
`f=x,y,z:l=...` for fractional coordinates or `c=x,y,z:l=...` for Cartesian
coordinates. Do not use atom-index projection selectors such as `Si=1:l=...` 
or `Y=1,2:l=...`; this workflow writes atom labels as repeated species names,
 and Wannier90 will not recognize those selectors here.
 Your projection lines must generate exactly `num_wann` usable
projections. If they generate fewer or more than `num_wann`, the runner will
fail. Count projections by angular momentum multiplicity: 
`l=0` gives 1 function per selected site, `l=1` gives 3, `l=2` gives 5, and `l=3` 
gives 7. Do not multiply these counts by the number of beta projectors in the 
UPF unless you use an explicitly supported radial-projector syntax.

Window values must be in eV, because the runner writes them directly into the 
Wannier90 `.win` file. If you parse eigenvalues in Hartree, convert them to eV 
before writing the recipe. `dis_win_min` and `dis_win_max` must contain at 
least `num_wann` states at every k-point. `dis_froz_max` must not freeze 
more than `num_wann` states at any k-point; keep it below the minimum energy 
of band `num_wann + 1` across k-points, with margin.

Do not run `/app/locked_wannier_runner.py`. Direct agent-side calls are
rejected. Harbor's verifier will run that deterministic executor after you
exit, before grading. This prevents agent tokens from being spent while
Wannier90 is computing.

Do not run `wannier90.x`, `pw2wannier90.x`, `pw.x`, `rm`, `kill`, `pkill`, or
`killall` yourself. Do not edit `.win`, `.pw2wan`, generated Wannier files, or
files under `material/` yourself. Do not send `C-c`, `Ctrl-C`, `SIGINT`,
terminal interrupt keys, or EOF/control keys. Do not poll for `report.json`,
do not inspect execution logs, and do not attempt any manual rescue path.

The locked runner will author `.win` and `.pw2wan` from your recipe, copy the
provided QE save tree into `workflow/run_dir`, run `wannier90.x -pp`,
`pw2wannier90.x`, and `wannier90.x`, then collect artifacts and reports.
If your recipe is invalid or the commands fail, the attempt should fail rather
than be silently corrected. The runner only performs broad JSON validation
before execution; it will not repair projection syntax or projection counts.

After writing `workflow/recipe_request.json`, stop. Return only a concise final
JSON status like:

```json
{{
  "status": "recipe_submitted",
  "task_complete": true,
  "recipe_path": "workflow/recipe_request.json",
  "runner": "deferred_to_harbor_verifier"
}}
```
"""


def upsert_locked_runner_instruction_appendix(instruction_text: str, material: str) -> str:
    marker = "# Locked DeepSeek Execution Contract"
    if marker in instruction_text:
        instruction_text = instruction_text.split(marker, 1)[0].rstrip() + "\n"
    return instruction_text + locked_runner_instruction_appendix(material)


def inject_locked_tools_into_dockerfile(dockerfile_text: str) -> str:
    denied_commands = " ".join(shlex.quote(name) for name in LOCKED_DENIED_COMMANDS)
    copy_snippet = (
        f"COPY {LOCKED_RUNNER_NAME} {LOCKED_RUNNER_APP_PATH}\n"
        f"COPY {LOCKED_COMMAND_WRAPPER_NAME} {LOCKED_COMMAND_WRAPPER_APP_PATH}\n"
        f"COPY instruction.md /app/instruction.md\n"
        f"RUN chmod +x {LOCKED_RUNNER_APP_PATH} {LOCKED_COMMAND_WRAPPER_APP_PATH} && "
        f"mkdir -p {LOCKED_BIN_APP_DIR} && "
        f"for name in {denied_commands}; do "
        f"ln -sf {LOCKED_COMMAND_WRAPPER_APP_PATH} {LOCKED_BIN_APP_DIR}/$name; "
        "done\n"
    )
    profile_lines = " ".join(
        shlex.quote(line)
        for line in controlled.terminus_login_trace_profile_script().splitlines()
    )
    profile_hook = (
        "RUN mkdir -p /etc/profile.d && printf '%s\\n' "
        f"{profile_lines} > /etc/profile.d/harbor-agent-trace.sh\n"
    )

    additions = ""
    if f"COPY {LOCKED_RUNNER_NAME} {LOCKED_RUNNER_APP_PATH}" not in dockerfile_text:
        additions += copy_snippet
    if (
        "COPY instruction.md /app/instruction.md" not in dockerfile_text
        and "COPY instruction.md /app/instruction.md" not in additions
    ):
        additions += "COPY instruction.md /app/instruction.md\n"
    if "harbor-agent-trace.sh" not in dockerfile_text:
        additions += profile_hook
    if not additions:
        return dockerfile_text

    marker = "COPY material /app/material\n"
    if marker in dockerfile_text:
        return dockerfile_text.replace(marker, additions + marker, 1)
    return dockerfile_text + "\n" + additions


def materialize_controlled_dataset(
    source_dataset: Path,
    tasks: list[tuple[int, str, Path]],
) -> tuple[Path, list[tuple[int, str, Path]]]:
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    target_dataset = (
        DEFAULT_AUGMENTED_DATASET_PARENT
        / f"{source_dataset.name}__deepseek_controlled_recipe__{timestamp}__pid{os.getpid()}"
    )
    target_dataset.mkdir(parents=True, exist_ok=False)

    def link_or_copy(src: str, dst: str) -> None:
        if Path(src).name == "Dockerfile":
            shutil.copy2(src, dst)
            return
        try:
            os.link(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(src, dst)

    runner_text = generic_locked_runner_script()
    compile(runner_text, LOCKED_RUNNER_NAME, "exec")

    augmented_tasks: list[tuple[int, str, Path]] = []
    for num_wann, material, source_task in tasks:
        target_task = target_dataset / material
        target_task.mkdir(parents=True, exist_ok=False)

        for child in sorted(source_task.iterdir()):
            if child.name == "instruction.md":
                continue
            if child.name == "environment":
                target_environment = target_task / "environment"
                shutil.copytree(
                    child,
                    target_environment,
                    copy_function=link_or_copy,
                    ignore=shutil.ignore_patterns(
                        LOCKED_RUNNER_NAME,
                        LOCKED_COMMAND_WRAPPER_NAME,
                    ),
                )
                continue
            link = target_task / child.name
            link.symlink_to(child.resolve(), target_is_directory=child.is_dir())

        instruction_text = (source_task / "instruction.md").read_text(encoding="utf-8")
        instruction_text = upsert_locked_runner_instruction_appendix(
            instruction_text,
            material,
        )
        (target_task / "instruction.md").write_text(instruction_text, encoding="utf-8")
        (target_task / "environment" / "instruction.md").write_text(
            instruction_text,
            encoding="utf-8",
        )

        runner_path = target_task / "environment" / LOCKED_RUNNER_NAME
        runner_path.write_text(runner_text, encoding="utf-8")
        runner_path.chmod(0o755)

        command_wrapper_path = target_task / "environment" / LOCKED_COMMAND_WRAPPER_NAME
        command_wrapper_path.write_text(
            controlled.locked_command_wrapper_script(),
            encoding="utf-8",
        )
        command_wrapper_path.chmod(0o755)

        dockerfile_path = target_task / "environment" / "Dockerfile"
        dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
        replacement_dockerfile = dockerfile_path.with_name(
            f"{dockerfile_path.name}.tmp"
        )
        replacement_dockerfile.write_text(
            inject_locked_tools_into_dockerfile(dockerfile_text),
            encoding="utf-8",
        )
        replacement_dockerfile.replace(dockerfile_path)

        controlled.ensure_local_tests_dir(target_task)
        test_script_path = target_task / "tests" / "test.sh"
        if not test_script_path.is_file():
            raise FileNotFoundError(
                f"expected Harbor verifier script does not exist: {test_script_path}"
            )
        test_script_text = test_script_path.read_text(encoding="utf-8")
        test_script_path.write_text(
            controlled.inject_locked_runner_verifier_hook(test_script_text),
            encoding="utf-8",
        )
        test_script_path.chmod(0o755)

        augmented_tasks.append((num_wann, material, target_task))

    lines = [
        "# DeepSeek Controlled Recipe Dataset",
        "",
        f"Source dataset: `{source_dataset}`",
        f"Task count: {len(augmented_tasks)}",
        "",
        "Each task preserves the original input prompt and material package, then",
        "adds a recipe-only DeepSeek contract plus a generic locked verifier runner.",
        "No prior self-debug reports or per-material answer table are copied.",
        "",
    ]
    (target_dataset / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return target_dataset, augmented_tasks


def deepseek_harbor_args(
    cli: argparse.Namespace,
    tasks: list[tuple[int, str, Path]],
) -> argparse.Namespace:
    agent_timeout_sec = common_task_timeout_sec(tasks, "agent")
    verifier_timeout_sec = common_task_timeout_sec(tasks, "verifier")
    return argparse.Namespace(
        dataset=cli.dataset,
        agent="terminus-2",
        model=MODEL,
        n_concurrent=1,
        batch_size=cli.batch_size,
        stop_on_error=cli.stop_on_error,
        docker_prune_after_batch=True,
        docker_prune_after_material=False,
        delete_after_run=True,
        extra_arg=[
            "--agent-timeout-multiplier",
            f"{cli.recipe_agent_timeout_sec / agent_timeout_sec:.6g}",
            "--verifier-timeout-multiplier",
            f"{cli.success_wave_timeout_sec / verifier_timeout_sec:.6g}",
            "--max-retries",
            "2",
            "--retry-include",
            "AgentSetupTimeoutError",
            "--retry-include",
            "NonZeroAgentExitCodeError",
        ],
        artifact=CONTROLLED_ARTIFACTS,
        no_default_artifacts=False,
        save_generated_qe_save=False,
        jobs_root=cli.jobs_root,
        target_success_runs=2 if cli.target_runs is None else None,
        target_runs=cli.target_runs,
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
    )


def main() -> None:
    cli = parse_args()
    materials = [name.strip() for name in MATERIALS if name.strip()]

    if not materials:
        raise SystemExit(
            "MATERIALS is empty. Edit this new script and add up to 10 directory names."
        )
    if len(materials) != len(set(materials)):
        raise SystemExit("MATERIALS contains duplicate entries.")
    if cli.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if cli.target_runs is not None and cli.target_runs < 1:
        raise SystemExit("--target-runs must be at least 1")
    if cli.recipe_agent_timeout_sec < 1:
        raise SystemExit("--recipe-agent-timeout-sec must be at least 1")
    if cli.success_wave_timeout_sec < 1:
        raise SystemExit("--success-wave-timeout-sec must be at least 1")
    if cli.success_wave_kill_after_sec < 0:
        raise SystemExit("--success-wave-kill-after-sec cannot be negative")

    cli.dataset = cli.dataset.expanduser().resolve()
    cli.jobs_root = cli.jobs_root.expanduser().resolve()

    requested = set(materials)
    tasks = harbor_generator.dataset_tasks(
        cli.dataset,
        include_materials=requested,
    )
    found = {material for _num_wann, material, _source in tasks}
    missing = sorted(requested - found)
    if missing:
        raise SystemExit(
            "Unknown material directory/directories: " + ", ".join(missing)
        )

    print("# DeepSeek controlled materials run")
    print("# Input task surface: generate_harbor_qwen_materials_command.py")
    print("# Execution path: Terminus recipe proposal, locked verifier execution")
    print("# Prior self-debug context: not copied")
    print(': "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running}"')
    print(
        'export OPENAI_BASE_URL="${OPENAI_BASE_URL:-'
        + DEFAULT_DEEPSEEK_BASE_URL
        + '}"'
    )

    repeats_by_material: dict[str, int] | None = None
    if cli.target_runs is not None:
        counts = existing_run_counts(
            cli.jobs_root,
            valid_materials=requested,
        )
        repeats_by_material = {}
        pending_tasks = []
        for task in tasks:
            _num_wann, material, _source = task
            existing = counts[material]
            needed = max(0, cli.target_runs - existing)
            print(
                f"# {material}: existing={existing}, "
                f"target={cli.target_runs}, new={needed}"
            )
            if needed:
                repeats_by_material[material] = needed
                pending_tasks.append(task)
        tasks = pending_tasks

    if not tasks:
        print("# Every material already has the requested number of runs.")
        print("true")
        return

    augmented_dataset, augmented_tasks = materialize_controlled_dataset(
        cli.dataset,
        tasks,
    )
    args = deepseek_harbor_args(cli, augmented_tasks)
    args.dataset = augmented_dataset

    if repeats_by_material is not None:
        augmented_tasks = [
            task
            for task in augmented_tasks
            for _repeat in range(repeats_by_material[task[1]])
        ]
        harbor_generator.print_ordered_commands(args, augmented_tasks)
    else:
        harbor_generator.print_target_success_loop(args, augmented_tasks)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
