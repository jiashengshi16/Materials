#!/usr/bin/env python3
"""Run DeepSeek/Terminus2 on prior-run Wannierisation evidence.

No argparse. Edit MATERIALS below, then run:

    scripts/run_deepseek_pro_self_debug_reviews.py

Requires Harbor and DEEPSEEK_API_KEY or OPENAI_API_KEY.
OPENAI_BASE_URL defaults to https://api.deepseek.com.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARBOR_DATASET_ROOT = ROOT / "harbor_datasets" / "wannier_200"

# Hardcoded experiment controls.
# Choose either "chemically similar" or "list".
MATERIAL_SELECTION_MODE = "list"

MATERIALS = [

'Al18Co4',
'Al4Sc2',
'Li4O6Si2',
'Si6Y10',
'Mg2O10Ti4',
"Al4Mn2O8",
"C4O12Sr4",
'B2Ta',
'RuTi',
'Ag2Y',
'NNb',
'C2Cu2O6'
]

# MATERIALS = [
#     "Al12Ni4",
#     "Al4Mn2O8",
#     "Al4O8Zn2",
#     'Al4Y2',
#     "Al8Zr4",
#     "Al4Sc2",
# "Kr2",
# 'Au2Y',
# 'Ag2Sc',
# 'Ag2Y',
# 'B2Mn',
# 'B2Ta',
# 'B2Ti',
# 'C2Cu2O6',
# 'C4O12Sr4',
# 'C2Cd2O6',
# 'O2Sr',
# 'Br2V',
# 'Cl2Ti',
# 'Mg4O12Se4',
# 'Li4O6Si2',
# 'F4Ni2',
# 'Co2F4',
# 'Cr6Ga2',
# 'Mo6Si2',
# 'Al2Mo6',
# 'Ga2Mo6',
# 'B8H16O16',
# 'Ne',
# 'Hf6Si4',
# 'Hf4Si2',
# 'Si6Y10',
# 'O2Pd2',
# 'Co2O8W2',
# 'CTi',
# 'Hf4Ni4',
# 'Pt4Y4',
# 'Hg3O3',
# 'O2Pb2',
# 'Ru4S8',
# 'Co4S8',
# 'FeTi',
# 'RuZr',
# 'RhSc',
# 'FLi',
# 'BrNa',
# 'Ar2',
# 'AgSc',
# ]

"""
export OPENAI_API_KEY="sk-your-new-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
"""
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
def normalize_terminus_model(model: str) -> str:
    aliases = {
        "deepseek-v4-pro": "openai/deepseek-v4-pro",
        "deepseek-v4-flash": "openai/deepseek-v4-flash",
    }
    return aliases.get(model, model)


MODEL = normalize_terminus_model(
    os.environ.get("DEEPSEEK_MODEL", "openai/deepseek-v4-pro")
)
TERMINUS_AGENT = "terminus-2"
HARBOR_BIN = "harbor"
HARBOR_REVIEW_IMAGE = os.environ.get(
    "DEEPSEEK_REVIEW_DOCKER_IMAGE",
    "wannier-qe-gemini-base:0.46.0",
)
MAX_CONCURRENT_DEEPSEEK = 12
OUTPUT_ROOT = ROOT / "jobsDeepseekProTerminus2InstructionTest" / "deepseek_pro_debug_reviews"
RUN_ROOTS = [
    ROOT / "jobsDeepseekProTerminus2Controlled",
]
NUM_WANN_JOB_RE = re.compile(
    r"^num_wann_ordered__(?P<timestamp>.+?)__pid(?P<pid>\d+)__"
    r"(?P<middle>.+?)__num_wann_(?P<num_wann>\d+)__(?P<material>.+)$"
)


@dataclass(frozen=True)
class TrialCase:
    material: str
    job_dir: Path
    trial_dir: Path
    attempt_dir: Path
    case_id: str
    job_metadata: dict[str, Any]
    manifest: dict[str, Any]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_present(path: Path) -> Any | None:
    """Return parsed JSON when available and valid; otherwise return None."""
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def read_json_object_if_present(path: Path) -> dict[str, Any]:
    data = read_json_if_present(path)
    return data if isinstance(data, dict) else {}


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_file_if_present(src: Path, dst: Path) -> bool:
    """Copy a file when it exists. Missing files are recorded by the caller, not fatal."""
    if not src.is_file():
        return False
    try:
        copy_file(src, dst)
    except OSError:
        return False
    return True


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_attempt_file(attempt: Path, material: str, suffix: str) -> Path:
    exact = attempt / f"{material}{suffix}"
    if exact.exists():
        return exact
    matches = sorted(attempt.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"Could not resolve *{suffix} for {material} in {attempt}")


def has_attempt_file(attempt: Path, material: str, suffix: str) -> bool:
    exact = attempt / f"{material}{suffix}"
    if exact.exists():
        return True
    return bool(sorted(attempt.glob(f"*{suffix}")))


def optional_attempt_evidence_files(attempt: Path, material: str) -> list[Path]:
    """Return optional text-like artifacts that can improve forensic reviews."""
    patterns = (
        f"{material}.amn",
        f"{material}.eig",
        f"{material}.mmn",
        f"{material}.nnkp",
        f"{material}_hr.dat",
        f"{material}.pp.log",
        f"{material}.pw2wan",
        f"{material}.wannier90.log",
        "*.amn",
        "*.mmn",
        "*.pp.log",
        "*.pw2wan",
        "*pw2wan*.log",
        "*.wannier90.log",
        "*.werr",
        "*.err",
    )
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in sorted(attempt.glob(pattern)):
            if path.is_file():
                files[path] = None
    return list(files)


def optional_workflow_evidence_files(trial_dir: Path) -> list[tuple[Path, Path]]:
    """Return workflow-contract and runner artifacts for forensic review staging."""
    relative_paths = (
        "artifacts/app/workflow/recipe_request.json",
        "artifacts/app/workflow/compile_recipe_report.json",
        "artifacts/app/workflow/locked_runner.log",
        "artifacts/app/workflow/LOCKED_RECIPE.json",
        "artifacts/app/workflow/DECISIONS.md",
        "artifacts/app/workflow/locked_runner_state.json",
        "artifacts/logs/artifacts/REPORT.md",
        "artifacts/logs/artifacts/report.json",
        "artifacts/manifest.json",
        "config.json",
        "exception.txt",
        "result.json",
        "trial.log",
        "verifier/test-stdout.txt",
        "verifier/test-stderr.txt",
    )
    files: list[tuple[Path, Path]] = []
    for relative in relative_paths:
        src = trial_dir / relative
        if src.is_file():
            files.append((src, Path(relative)))
    return files


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_job_name(job_dir: Path) -> dict[str, Any]:
    match = NUM_WANN_JOB_RE.match(job_dir.name)
    if not match:
        return {
            "job_folder": job_dir.name,
            "job_timestamp": None,
            "pid": None,
            "job_middle": None,
            "ordinal": None,
            "attempt_from_folder": None,
            "num_wann_from_folder": None,
            "material_from_folder": None,
        }

    groups = match.groupdict()
    middle = groups["middle"]
    ordinal = int(middle) if middle.isdigit() else None
    attempt_match = re.search(r"(?:^|__)attempt_(\d+)(?:__|$)", middle)
    attempt_from_folder = int(attempt_match.group(1)) if attempt_match else None

    return {
        "job_folder": job_dir.name,
        "job_timestamp": groups["timestamp"],
        "pid": int(groups["pid"]),
        "job_middle": middle,
        "ordinal": ordinal,
        "attempt_from_folder": attempt_from_folder,
        "num_wann_from_folder": int(groups["num_wann"]),
        "material_from_folder": groups["material"],
    }


def case_id_for(job_metadata: dict[str, Any], trial_dir: Path) -> str:
    run_root = job_metadata.get("run_root") or "unknown_root"
    timestamp = job_metadata.get("job_timestamp") or "unknown_time"
    pid = job_metadata.get("pid")
    ordinal = job_metadata.get("ordinal")
    middle = job_metadata.get("job_middle")
    num_wann = job_metadata.get("num_wann_from_folder")
    middle_label = (
        f"ordinal_{ordinal:04d}"
        if isinstance(ordinal, int)
        else str(middle or "ordinal_unknown")
    )

    parts = [
        str(run_root),
        str(timestamp),
        f"pid{pid}" if isinstance(pid, int) else "pid_unknown",
        middle_label,
        f"num_wann_{num_wann:03d}" if isinstance(num_wann, int) else "num_wann_unknown",
        trial_dir.name,
    ]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "__".join(parts)).strip("_")


def candidate_trial_dirs(job_dir: Path) -> list[Path]:
    # If the job itself looks like a trial, use it.
    if any(
        (job_dir / name).exists()
        for name in ("artifacts", "agent", "verifier")
    ):
        return [job_dir]

    # Otherwise, look for immediate child trial directories.
    trials = [
        path
        for path in sorted(job_dir.iterdir())
        if path.is_dir()
        and any(
            (path / name).exists()
            for name in ("artifacts", "agent", "verifier")
        )
    ]

    # Absolute last resort: still return the job directory.
    return trials or [job_dir]


def trial_attempt_dir(trial_dir: Path) -> Path | None:
    """Find the best available artifact directory without requiring a fixed layout."""

    candidates = [
        trial_dir / "artifacts" / "attempt_1",
        trial_dir / "artifacts" / "logs" / "artifacts" / "attempt_1",
        trial_dir / "attempt_1",
        trial_dir / "artifacts",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Search recursively for any attempt_1 directory.
    matches = sorted(
        path for path in trial_dir.rglob("attempt_1")
        if path.is_dir()
    )
    if matches:
        return matches[0]

    # Last resort: use the trial directory itself.
    # This allows a case to exist even if no artifact directory was produced.
    return trial_dir


def read_json_object(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return data


def validate_manifest_material(manifest: dict[str, Any], material: str, manifest_path: Path) -> None:
    manifest_material = manifest.get("material_id") or manifest.get("material")
    if isinstance(manifest_material, str) and manifest_material == material:
        return

    seedname = manifest.get("seedname")
    if isinstance(seedname, str) and seedname == material:
        return

    if isinstance(manifest_material, str):
        raise SystemExit(
            f"Manifest material mismatch for {material}: {manifest_path} "
            f"contains {manifest_material!r}"
        )


def find_trial_cases(material: str) -> list[TrialCase]:
    cases: list[TrialCase] = []

    for run_root in RUN_ROOTS:
        if not run_root.is_dir():
            continue

        for job_dir in sorted(path for path in run_root.glob("num_wann_ordered*") if path.is_dir()):
            job_metadata = {
                **parse_job_name(job_dir),
                "run_root": run_root.name,
                "run_root_path": display_path(run_root),
            }
            if job_metadata.get("material_from_folder") != material:
                continue

            for trial_dir in candidate_trial_dirs(job_dir):
                attempt_dir = trial_attempt_dir(trial_dir)
                if attempt_dir is None:
                    continue
                manifest_path = attempt_dir / "run_manifest.json"
                manifest = read_json_object_if_present(manifest_path)
                if not manifest_path.is_file():
                    print(
                        f"Warning for {material}: missing run_manifest.json at {manifest_path}; "
                        "continuing with empty metadata."
                    )
                elif not manifest:
                    print(
                        f"Warning for {material}: run_manifest.json is invalid or not a JSON object "
                        f"at {manifest_path}; continuing with empty metadata."
                    )
                else:
                    try:
                        validate_manifest_material(manifest, material, manifest_path)
                    except SystemExit as exc:
                        print(f"Warning for {material}: {exc}; continuing anyway.")

                if not has_attempt_file(attempt_dir, material, ".win"):
                    print(
                        f"Warning for {material}: missing .win file in {attempt_dir}; "
                        "continuing with the remaining evidence."
                    )

                cases.append(
                    TrialCase(
                        material=material,
                        job_dir=job_dir,
                        trial_dir=trial_dir,
                        attempt_dir=attempt_dir,
                        case_id=case_id_for(job_metadata, trial_dir),
                        job_metadata=job_metadata,
                        manifest=manifest,
                    )
                )

    if not cases:
        print(
            f"No discoverable trial folders for {material}: "
            f"under {', '.join(display_path(root) for root in RUN_ROOTS)}"
        )
        return []

    return cases


def dataset_task_instruction_path(material: str) -> Path | None:
    material_dir = HARBOR_DATASET_ROOT / material

    candidates = [
        material_dir / "instruction.md",
        material_dir / "instructions.md",
        material_dir / "prompt.md",
        material_dir / "task.md",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(material_dir.glob("*instruction*.md"))
    if len(matches) == 1:
        return matches[0]

    matches = sorted(material_dir.glob("*instructions*.md"))
    if len(matches) == 1:
        return matches[0]

    return None


def first_user_message_from_trajectory(trial_dir: Path) -> str | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.is_file():
        return None

    try:
        data = read_json(trajectory_path)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    steps = data.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("source") != "user":
                continue
            message = step.get("message")
            if isinstance(message, str) and message.strip():
                return message.rstrip() + "\n"

    messages = data.get("messages")
    if isinstance(messages, list):
        for message_record in messages:
            if not isinstance(message_record, dict):
                continue
            if message_record.get("role") not in {"user", "human"}:
                continue
            message = message_record.get("content")
            if isinstance(message, str) and message.strip():
                return message.rstrip() + "\n"

    return None


def original_task_instructions(case: TrialCase) -> tuple[str, str]:
    trajectory_prompt = first_user_message_from_trajectory(case.trial_dir)
    if trajectory_prompt is not None:
        return trajectory_prompt, "agent/trajectory.json:first user message"

    instruction_path = dataset_task_instruction_path(case.material)
    if instruction_path is not None:
        try:
            return instruction_path.read_text(encoding="utf-8"), display_path(instruction_path)
        except OSError:
            pass

    return (
        "Original task instructions were not available for this case.\n",
        "not available",
    )


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "material_id",
        "seedname",
        "attempt",
        "status",
        "executed_successfully",
        "target_dft_band_start",
        "target_dft_band_end",
        "num_wann",
        "num_bands",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def terminus_instruction_text(case: TrialCase) -> str:
    return (
        prompt_text(case)
        + """

## Terminus2 workspace note

You are running inside a Harbor Terminus2 task. The review evidence is staged in
the current working directory, with the same paths named above:

- `case_files/case_metadata.json` if present
- `case_files/verifier/diagnostics.json` if present
- `case_files/artifacts/attempt_1/...`
- `case_files/workflow_contract/...`
- `case_files/agent/...`
- `case_files/original_task_instructions.md` if present

Use shell tools to inspect those files directly. Do not use any embedded
summary as a substitute for reading the files.

Write the final reports to both of these locations:

- `self_debug_report.md` and `self_debug_report.json` in the current working directory
- `/logs/artifacts/self_debug_report.md` and `/logs/artifacts/self_debug_report.json`

The `/logs/artifacts` copies are required so Harbor can return the reports to
the host script.
"""
    )


def harbor_task_toml() -> str:
    return f"""schema_version = "1.3"

[agent]
timeout_sec = 7200
user = "root"

[verifier]
timeout_sec = 900
user = "root"

[environment]
network_mode = "public"
docker_image = "{HARBOR_REVIEW_IMAGE}"
workdir = "/app"
cpus = 8
memory_mb = 32768
storage_mb = 20480
"""


def harbor_task_dir_for_case(case_dir: Path) -> Path:
    return case_dir / "harbor_task"


def prepare_harbor_task_files(case_dir: Path, instruction: str) -> None:
    """Stage this review case as a minimal Harbor task for Terminus2."""
    task_dir = harbor_task_dir_for_case(case_dir)
    if task_dir.exists():
        shutil.rmtree(task_dir)

    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)

    write_text(task_dir / "instruction.md", instruction)
    write_text(task_dir / "task.toml", harbor_task_toml())
    write_text(environment_dir / "prompt.md", instruction)
    test_path = task_dir / "tests" / "test.sh"
    write_text(test_path, "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n")
    test_path.chmod(0o755)

    environment_case_files = environment_dir / "case_files"
    shutil.copytree(case_dir / "case_files", environment_case_files)


def prompt_text(case: TrialCase) -> str:
    material = case.material
    num_wann = case.job_metadata.get("num_wann_from_folder")
    target_start = case.manifest.get("target_dft_band_start")
    target_end = case.manifest.get("target_dft_band_end")

    return f"""# DeepSeek Pro Self-Debug Review: {material}

You are reviewing a Wannierisation trajectory for `{material}`.
This exact case is:

- case id: `{case.case_id}`
- source run root: `{case.job_metadata.get("run_root")}`
- source job folder: `{case.job_dir.name}`
- source trial folder: `{case.trial_dir.name}`
- `num_wann` from the job folder: `{num_wann}`
- target DFT bands from the run manifest: `{target_start}` to `{target_end}`

If another case directory exists for the same material, treat it as a separate
old run or separate Wannierisation. Diagnose only the current case directory.

This is forensic analysis only. Do not rerun QE, do not produce a new
Wannierisation, and do not browse the internet. Use only files in this case
directory. Do not read files outside this case directory, even if trajectory
logs mention outside paths.

Your job is to reconstruct, as closely as the logs allow, the old decision chain 
using the original task instructions and old run logs. Treat 
`case_files/original_task_instructions.md` as the task prompt that was available 
to the old model during the original run.

Evaluate the trajectory fairly. If the old run made scientifically reasonable
choices given the information available at the time, say so and do not force a
critique. Only identify avoidable mistakes when the task materials, trajectory, 
logs, or final diagnostics provide evidence that a specific choice was not ideal/poor, 
contradicted by later run output, or led to an avoidable failure or high RMSE.
For any such issue, explain the evidence-backed diagnosis using only information
that is present in the staged case files. 

Read these files if present:

- `case_files/case_metadata.json` if present
- `case_files/verifier/diagnostics.json` if present
- `case_files/artifacts/attempt_1/{material}.win` if present
- `case_files/artifacts/attempt_1/{material}.wout` if present
- `case_files/artifacts/attempt_1/{material}.amn` if present
- `case_files/artifacts/attempt_1/{material}.eig` if present
- `case_files/artifacts/attempt_1/{material}.mmn` if present
- `case_files/artifacts/attempt_1/{material}.nnkp` if present
- `case_files/artifacts/attempt_1/{material}_hr.dat` if present
- `case_files/artifacts/attempt_1/{material}.pp.log` if present
- `case_files/artifacts/attempt_1/{material}.wannier90.log` if present
- `case_files/artifacts/attempt_1/*.pw2wan`, `*pw2wan*.log`, `*.werr`,
  or `*.err` if present
- `case_files/artifacts/attempt_1/run_manifest.json` if present
- `case_files/workflow_contract/artifacts/app/workflow/recipe_request.json` if present
- `case_files/workflow_contract/artifacts/app/workflow/compile_recipe_report.json` if present
- `case_files/workflow_contract/artifacts/app/workflow/locked_runner.log` if present
- `case_files/workflow_contract/artifacts/app/workflow/LOCKED_RECIPE.json` if present
- `case_files/workflow_contract/artifacts/app/workflow/DECISIONS.md` if present
- `case_files/workflow_contract/artifacts/app/workflow/locked_runner_state.json` if present
- `case_files/workflow_contract/artifacts/logs/artifacts/REPORT.md` if present
- `case_files/workflow_contract/artifacts/logs/artifacts/report.json` if present
- `case_files/workflow_contract/artifacts/manifest.json` if present
- `case_files/workflow_contract/config.json` if present
- `case_files/workflow_contract/exception.txt` if present
- `case_files/workflow_contract/result.json` if present
- `case_files/workflow_contract/trial.log` if present
- `case_files/workflow_contract/verifier/test-stdout.txt` or `test-stderr.txt` if present
- `case_files/agent/trajectory.json` if present
- `case_files/agent/gemini-cli.trajectory.jsonl` if present
- `case_files/agent/gemini-cli.txt` if present
- `case_files/original_task_instructions.md` if present

The per-run verifier diagnostics are allowed only as scalar final outcome
metrics for this specific run. They do not provide the hidden reference recipe
or the DeepSeek/reference RMSE ratio. Do not use the verifier diagnostics to
infer hidden reference settings, reference-only methods, or a DeepSeek/reference
RMSE ratio. The diagnosis must obey `case_files/original_task_instructions.md`
when judging what the old model could know or control.

Do not handwave from aggregate statistics. The core diagnosis must
come from this material's `.win`, `.wout`, `.amn`, `.eig`, `.mmn`, `.nnkp`,
`_hr.dat`, preprocessor/pw2wannier/wannier90/error logs, workflow-contract
files, run manifest, trajectory, and per-run verifier diagnostics, if present.

Use the workflow-contract files to distinguish what the old model could control
from what the locked runner hard-coded. Do not recommend or criticize the old
model for failing to set a field unless `recipe_request.json`, the original
task instructions, or the runner logs show that field was actually available to
the old model.

Write exactly these two files:

- `self_debug_report.md`
- `self_debug_report.json`

The Markdown report must be step-by-step and specific. For each substantive
decision in the old trajectory, judge whether it was good, bad, mixed, or
uncertain, and explain why USING CONCRETE EVIDENCE from `.win`, `.wout`,
manifest notes, workflow-contract files, trajectory reasoning, and final error
metrics. Cite the file paths you used, and include line numbers when you have
them from grep, rg, nl, or similar inspection. Cover at least:

1. projection choice
2. `num_wann` / target-band handling;
3. `num_bands` / band-pool handling;
4. disentanglement outer and frozen windows;
5. response to Wannier90 warnings or iteration caps;
6. localization quality from WF spreads and spread components;
7. whether the old run accepted a result it should have rejected, while
   explicitly separating old-model responsibility from locked-runner behavior;

If the run shows evidence of avoidable issues, also cover:

8. the most likely specific failure chain.

Do not produce next-run recommendations, future playbooks, anti-loop rules, or
operational retry plans in this review unless you are very confident what to do. 
Your job is mainly diagnosis: what failed, where it failed, why the evidence
supports that diagnosis, and what remains uncertain.

For every decision review, answer all of these forensic questions:

- What exactly did the old run decide or claim?
- What evidence did the old run use at the time?
- What later evidence in `.wout`, verifier diagnostics, or trajectory contradicts
  or weakens that decision?
- Was the mistake avoidable without seeing the hidden reference recipe?
- Which exact step failed, if any: recipe writing, compile/preflight,
  `wannier90.x -pp`, `pw2wannier90.x`, final `wannier90.x` disentanglement,
  final `wannier90.x` localization, artifact collection, verifier scoring, or
  old-model interpretation?
- What is the confidence level for any causal claim: `proven`,
  `strongly_supported`, `plausible_but_unproven`, or `unsupported`?
- What remains uncertain because the logs do not contain enough information?

Be especially suspicious of:

- accepting a result after checking only that `<seed>_hr.dat` exists;
- judging localization by average/total spread while ignoring max WF spread,
  spread outliers, final gradient, or iteration limits;
- declaring projections "excellent" or windows "robust" merely because they are
  chemically plausible;
- padding `num_wann` with duplicate same-site, same-angular-momentum
  projections without evidence that the channels are linearly independent;
- abandoning a physically motivated projection after a syntax, stale-file, or
  workflow error and falling back to `random`;
- claiming a root cause merely because it is a common failure mode;
- using a chemically plausible story as proof when the logs only show
  correlation.

For each diagnosis claim, cite the specific evidence that supports it. If the
files only show a symptom (for example high RMSE, spread outliers, SVD warning,
or nonconvergence) but do not identify the root cause, say "symptom observed;
root cause not proven" rather than inventing one.

Before calling a projection or window decision "bad", do the relevant arithmetic
from the available files when possible:

- projection count must equal `num_wann`;
- outer window per-k-point count must be at least `num_wann`;
- frozen window per-k-point count must be at most `num_wann`;
- coordinate-center claims must distinguish `f=` fractional coordinates from
  `c=` Cartesian coordinates;
- any claim about allowed recipe controls must be checked against staged
  workflow-contract files.

Be explicit about uncertainty. Do not claim causal proof when the files only
support diagnostic correlation. But make concrete judgments where evidence is
strong: projections, unconverged disentanglement, huge WF spreads,
fragile windows, or mismatch between claimed rationale and observed output.
Use "plausible but unproven" rather than "good" when a choice is chemically
reasonable but the run output shows poor localization or band interpolation.
The goal is to find avoidable scientific decision errors, not to assign credit
for parameters that merely look conventional.

The JSON report must have this shape:

```json
{{
  "material": "{material}",
  "case_id": "{case.case_id}",
  "run_root": "{case.job_metadata.get("run_root")}",
  "job_folder": "{case.job_dir.name}",
  "trial_folder": "{case.trial_dir.name}",
  "num_wann_from_job_folder": {json.dumps(num_wann)},
  "verdict": "good | mixed | bad | uncertain",
  "projection_verdict": "good | not_used | bad | uncertain",
  "decision_reviews": [
    {{
      "decision": "short name",
      "verdict": "good | mixed | bad | uncertain",
      "evidence": ["specific file-backed evidence"],
      "old_claim_or_decision": "what the old run said or did",
      "observed_failure_signal": "what later output showed",
      "failed_step": "none | recipe_writing | compile_preflight | wannier90_pp | pw2wannier90 | final_disentanglement | final_localization | artifact_collection | verifier_scoring | old_model_interpretation | locked_runner_behavior | unknown",
      "causal_confidence": "proven | strongly_supported | plausible_but_unproven | unsupported",
      "why": "specific evidence-backed explanation",
      "old_model_responsibility": "avoidable | not_avoidable | mixed | unknown",
      "uncertainty": "what remains uncertain, or null"
    }}
  ],
  "failure_chain": [
    {{
      "step": "specific failed step",
      "claim": "specific causal claim",
      "causal_confidence": "proven | strongly_supported | plausible_but_unproven | unsupported",
      "evidence": ["specific file-backed evidence"]
    }}
  ],
  "symptoms_observed": ["directly observed symptoms, not inferred causes"],
  "root_causes_supported": ["root causes with proven or strongly_supported evidence"],
  "plausible_but_unproven_causes": ["possible causes that should not be treated as facts"],
  "unsupported_or_overreaching_claims_to_avoid": ["claims not supported by staged files"],
  "workflow_constraints_relevant_to_diagnosis": ["constraints found in original task or workflow-contract files"],
  "evidence_gaps": ["missing files or missing diagnostics that limit the diagnosis"]
}}
```

If you cannot prove the exact root cause from the evidence, say so explicitly.
Do not fill the gap with a recommended recipe or future playbook.

Your final response should be a JSON object pointing to
`self_debug_report.md` and `self_debug_report.json`.
"""

def build_case(case: TrialCase) -> Path:
    material = case.material
    case_dir = OUTPUT_ROOT / material / case.case_id
    clean_dir(case_dir)
    case_files = case_dir / "case_files"

    win_dst = case_files / "artifacts" / "attempt_1" / f"{material}.win"
    try:
        copy_file(find_attempt_file(case.attempt_dir, material, ".win"), win_dst)
        win_copied = True
    except (SystemExit, FileNotFoundError, OSError):
        win_copied = False
    wout_dst = case_files / "artifacts" / "attempt_1" / f"{material}.wout"
    try:
        copy_file(find_attempt_file(case.attempt_dir, material, ".wout"), wout_dst)
        wout_copied = True
    except (SystemExit, FileNotFoundError, OSError):
        write_text(
            wout_dst,
            "No .wout file was present in the source artifacts for this case.\n",
        )
        wout_copied = False
    manifest_copied = copy_file_if_present(
        case.attempt_dir / "run_manifest.json",
        case_files / "artifacts" / "attempt_1" / "run_manifest.json",
    )

    staged_optional_artifacts: list[dict[str, str]] = []
    for src in optional_attempt_evidence_files(case.attempt_dir, material):
        dst = case_files / "artifacts" / "attempt_1" / src.name
        if not copy_file_if_present(src, dst):
            continue
        staged_optional_artifacts.append(
            {
                "source": display_path(src),
                "staged": display_path(dst),
            }
        )

    staged_workflow_contract_artifacts: list[dict[str, str]] = []
    for src, relative_dst in optional_workflow_evidence_files(case.trial_dir):
        dst = case_files / "workflow_contract" / relative_dst
        if not copy_file_if_present(src, dst):
            continue
        staged_workflow_contract_artifacts.append(
            {
                "source": display_path(src),
                "staged": display_path(dst),
            }
        )

    trajectory_copied = copy_file_if_present(
        case.trial_dir / "agent" / "trajectory.json",
        case_files / "agent" / "trajectory.json",
    )

    for optional in ("gemini-cli.trajectory.jsonl", "gemini-cli.txt"):
        src = case.trial_dir / "agent" / optional
        if src.exists():
            copy_file_if_present(src, case_files / "agent" / optional)

    diagnostics_src = case.trial_dir / "verifier" / "diagnostics.json"
    diagnostics_copied = copy_file_if_present(
        diagnostics_src,
        case_files / "verifier" / "diagnostics.json",
    )

    task_text, task_source = original_task_instructions(case)
    write_text(case_files / "original_task_instructions.md", task_text)

    metadata = {
        "material": material,
        "case_id": case.case_id,
        "run_root": case.job_metadata.get("run_root"),
        "job_folder": case.job_dir.name,
        "trial_folder": case.trial_dir.name,
        "source_job_path": display_path(case.job_dir),
        "source_trial_path": display_path(case.trial_dir),
        "source_attempt_path": display_path(case.attempt_dir),
        "job_metadata": case.job_metadata,
        "manifest_summary": manifest_summary(case.manifest),
        "original_task_instructions_source": task_source,
        "win_copied": win_copied,
        "wout_copied": wout_copied,
        "run_manifest_copied": manifest_copied,
        "trajectory_copied": trajectory_copied,
        "staged_optional_artifacts": staged_optional_artifacts,
        "staged_workflow_contract_artifacts": staged_workflow_contract_artifacts,
        "verifier_diagnostics_copied": diagnostics_copied,
        "verifier_diagnostics_source": display_path(diagnostics_src) if diagnostics_copied else None,
        "aggregate_inputs_intentionally_not_copied": [
            "jobs/num_wann_ordered_diagnostics_summary.json",
            "jobs/gemini_vs_reference_errors.xlsx",
            "jobs/gemini_failure_modes/failure_modes.csv",
        ],
    }
    write_text(case_files / "case_metadata.json", json.dumps(metadata, indent=2) + "\n")

    instruction = terminus_instruction_text(case)
    write_text(case_dir / "prompt.md", prompt_text(case))
    prepare_harbor_task_files(case_dir, instruction)
    return case_dir

def report_is_nonempty(case_dir: Path) -> bool:
    md_path = case_dir / "self_debug_report.md"
    json_path = case_dir / "self_debug_report.json"

    if not md_path.is_file() or not json_path.is_file():
        return False

    if not md_path.read_text(encoding="utf-8").strip():
        return False

    raw_json = json_path.read_text(encoding="utf-8").strip()
    if not raw_json:
        return False

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return False

    # Minimal sanity checks that this is actually the requested diagnosis.
    if not isinstance(data, dict):
        return False
    if not data.get("verdict"):
        return False
    if not data.get("decision_reviews"):
        return False
    if not isinstance(data.get("failure_chain"), list):
        return False
    if not isinstance(data.get("symptoms_observed"), list):
        return False
    if not isinstance(data.get("evidence_gaps"), list):
        return False

    return True


def deepseek_harbor_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OPENAI_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    if not env.get("OPENAI_API_KEY") and env.get("DEEPSEEK_API_KEY"):
        env["OPENAI_API_KEY"] = env["DEEPSEEK_API_KEY"]
    if not env.get("OPENAI_API_KEY") and "localhost" not in env["OPENAI_BASE_URL"]:
        raise SystemExit("DEEPSEEK_API_KEY or OPENAI_API_KEY is not set.")
    return env


def copy_harbor_reports_to_case(case_dir: Path, job_dir: Path) -> bool:
    if not job_dir.is_dir():
        return False

    candidates = sorted(
        job_dir.rglob("self_debug_report.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for md_src in candidates:
        json_src = md_src.with_name("self_debug_report.json")
        if not json_src.is_file():
            continue
        copy_file(md_src, case_dir / "self_debug_report.md")
        copy_file(json_src, case_dir / "self_debug_report.json")
        return True

    return False


def run_deepseek(case_dir: Path) -> None:
    combined_log_path = case_dir / "deepseek_stdout_stderr.txt"
    status_path = case_dir / "run_status.json"
    task_dir = harbor_task_dir_for_case(case_dir)
    harbor_jobs_dir = case_dir / "harbor_runs"
    harbor_jobs_dir.mkdir(parents=True, exist_ok=True)
    env = deepseek_harbor_env()

    attempts: list[dict[str, Any]] = []
    max_attempts = 10

    for attempt_index in range(1, max_attempts + 1):
        # Remove stale outputs so success must come from this attempt.
        for output_name in ("self_debug_report.md", "self_debug_report.json"):
            output_path = case_dir / output_name
            if output_path.exists():
                output_path.unlink()

        job_name = f"terminus2_self_debug_attempt_{attempt_index:02d}"
        job_dir = harbor_jobs_dir / job_name
        if job_dir.exists():
            shutil.rmtree(job_dir)

        command = [
            HARBOR_BIN,
            "run",
            "--yes",
            "--quiet",
            "--path",
            str(task_dir),
            "--jobs-dir",
            str(harbor_jobs_dir),
            "--job-name",
            job_name,
            "--agent",
            TERMINUS_AGENT,
            "--model",
            MODEL,
            "--n-concurrent",
            "1",
            "--agent-timeout-multiplier",
            "1.1",
            "--max-retries",
            "2",
            "--retry-include",
            "AgentSetupTimeoutError",
            "--retry-include",
            "NonZeroAgentExitCodeError",
            "--disable-verification",
        ]

        attempt_log_path = case_dir / f"deepseek_attempt_{attempt_index:02d}_harbor_stdout_stderr.txt"
        completed = subprocess.run(
            command,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        stdout_stderr = completed.stdout or ""
        attempt_log_path.write_text(stdout_stderr, encoding="utf-8")
        copied_reports = copy_harbor_reports_to_case(case_dir, job_dir)

        produced_nonempty_diagnosis = report_is_nonempty(case_dir)

        attempt_record = {
            "attempt": attempt_index,
            "command": command,
            "model": MODEL,
            "agent": TERMINUS_AGENT,
            "returncode": completed.returncode,
            "job_dir": display_path(job_dir),
            "attempt_log_path": attempt_log_path.name,
            "copied_reports_from_harbor": copied_reports,
            "self_debug_report_md_exists": (case_dir / "self_debug_report.md").is_file(),
            "self_debug_report_json_exists": (case_dir / "self_debug_report.json").is_file(),
            "produced_nonempty_diagnosis": produced_nonempty_diagnosis,
        }
        attempts.append(attempt_record)

        status = {
            "command": command,
            "agent": TERMINUS_AGENT,
            "model": MODEL,
            "max_attempts": max_attempts,
            "success": produced_nonempty_diagnosis,
            "attempts": attempts,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

        if produced_nonempty_diagnosis:
            combined_log_path.write_text(stdout_stderr, encoding="utf-8")
            return

        print(
            f"DeepSeek Terminus2 attempt {attempt_index}/{max_attempts} did not produce "
            f"a valid diagnosis for {case_dir.name}; retrying..."
        )

    combined_log_path.write_text(
        "\n\n".join(
            [
                f"===== ATTEMPT {record['attempt']} "
                f"returncode={record['returncode']} model={record['model']} =====\n"
                f"{(case_dir / record['attempt_log_path']).read_text(encoding='utf-8')}"
                for record in attempts
            ]
        ),
        encoding="utf-8",
    )

    raise SystemExit(
        f"DeepSeek Pro review failed after {max_attempts} attempts or did not write "
        f"a non-empty diagnosis: {case_dir}"
    )


def output_dir_for_case(case: TrialCase) -> Path:
    return OUTPUT_ROOT / case.material / case.case_id


def run_case(case: TrialCase) -> Path:
    case_dir = build_case(case)
    print(
        "Running DeepSeek Pro review for "
        f"{case.material} case={case.case_id} in {case_dir}"
    )
    run_deepseek(case_dir)
    return case_dir


def selected_materials() -> list[str]:
    mode = MATERIAL_SELECTION_MODE.strip().lower()

    if mode == "list":
        materials = [material.strip() for material in MATERIALS if material.strip()]
        if not materials:
            raise SystemExit(
                'MATERIALS is empty. Add material names at the top of the script '
                'or set MATERIAL_SELECTION_MODE = "chemically similar".'
            )
        if len(materials) != len(set(materials)):
            raise SystemExit("MATERIALS contains duplicate entries.")
        return materials

    if mode == "chemically similar":
        materials: set[str] = set()
        for run_root in RUN_ROOTS:
            if not run_root.is_dir():
                continue
            for job_dir in run_root.glob("num_wann_ordered*"):
                if not job_dir.is_dir():
                    continue
                material = parse_job_name(job_dir).get("material_from_folder")
                if isinstance(material, str) and material.strip():
                    materials.add(material.strip())

        if not materials:
            raise SystemExit(
                "No chemically similar candidate materials were found under "
                + ", ".join(display_path(root) for root in RUN_ROOTS)
            )
        return sorted(materials)

    raise SystemExit(
        'MATERIAL_SELECTION_MODE must be either "chemically similar" or "list".'
    )


def collect_cases() -> list[TrialCase]:
    all_cases: list[TrialCase] = []

    materials = selected_materials()
    print(f"Material selection mode: {MATERIAL_SELECTION_MODE!r} ({len(materials)} material(s)).")

    for material in materials:
        cases = find_trial_cases(material)
        if len(cases) > 1:
            print(f"Found {len(cases)} trial folders for {material}; reviewing all of them.")
        all_cases.extend(cases)

    unique_cases: list[TrialCase] = []
    output_dirs: dict[Path, TrialCase] = {}
    for case in all_cases:
        output_dir = output_dir_for_case(case)
        previous = output_dirs.get(output_dir)
        if previous is not None:
            if previous.material == case.material and previous.trial_dir == case.trial_dir:
                print(f"Skipping duplicate trial folder for {case.material}: {display_path(case.trial_dir)}")
                continue
            raise SystemExit(
                "Two trial cases resolve to the same output directory: "
                f"{previous.case_id} and {case.case_id} -> {output_dir}"
            )
        output_dirs[output_dir] = case
        unique_cases.append(case)

    return unique_cases

def main() -> None:
    if shutil.which(HARBOR_BIN) is None:
        raise SystemExit(
            f"Could not find {HARBOR_BIN!r} on PATH. Run this in the same "
            "environment where Harbor is installed."
        )
    deepseek_harbor_env()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = collect_cases()

    if not cases:
        print("No cases to review.")
        return

    max_workers = min(MAX_CONCURRENT_DEEPSEEK, len(cases))
    print(f"Running {len(cases)} DeepSeek Pro review(s) with concurrency={max_workers}.")

    failures: list[tuple[TrialCase, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_case = {pool.submit(run_case, case): case for case in cases}

        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                case_dir = future.result()
            except BaseException as exc:
                failures.append((case, exc))
                print(f"FAILED DeepSeek Pro review for {case.material} case={case.case_id}: {exc}")
                continue

            print(f"Wrote expected outputs in {case_dir}")

    if failures:
        details = "\n".join(
            f"- {case.material} case={case.case_id}: {exc}"
            for case, exc in failures
        )
        raise SystemExit(
            f"{len(failures)} DeepSeek Pro review(s) failed out of {len(cases)}:\n{details}"
        )


if __name__ == "__main__":
    main()
