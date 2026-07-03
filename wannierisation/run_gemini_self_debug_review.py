#!/usr/bin/env python3
"""Run Gemini directly on prior-run Wannierisation evidence.

No argparse. Edit MATERIALS and GEMINI_BIN below, then run:

    scripts/run_gemini_self_debug_reviews.py
"""

from __future__ import annotations

import csv
import json
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
USE_CANDIDATE_RUN_ERROR_TABLE = True
CANDIDATE_RUN_ERROR_TABLE = ROOT / "candidate_run_error_table.csv"
CANDIDATE_RUN_COLUMNS = ("run_1", "run_2", "run_3", "run_4", "run_5")
MATERIALS = [
    'Al18Co4',
    'Al4Mn2O8',
    'Al4O8Zn2',
    'B8H16O16',
    'Bi4Cl12',
    'Br18W3',
    'C2Cd2O6',
    'Ca4Mg8',
    'Cl4Li4O16',
    'Cl6Ni3',
    'FLi',
    'H8O16W4',
    'Hf10Si6',
    'Hf4Ni4',
    'Hg3O3',
    'In3Pd2',
    'K4O6S2',
    'Mg2O10Ti4',
    'Mo4S6',
    'N2Na2O6',
    'Ni4Zr4',
    'Se4Tl4',
]
MODEL = "gemini-3.1-pro-preview"
GEMINI_BIN = "gemini"
MAX_CONCURRENT_GEMINI = 12
OUTPUT_ROOT = ROOT / "jobs" / "gemini_self_debug_reviews"
JOBS_ROOT = ROOT / "jobs"
RUN_ROOTS = [
    JOBS_ROOT,
    ROOT / "reruns"
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


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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
    trials: list[Path] = []

    if (job_dir / "artifacts" / "attempt_1").is_dir():
        trials.append(job_dir)

    trials.extend(
        path
        for path in sorted(job_dir.iterdir())
        if path.is_dir() and (path / "artifacts" / "attempt_1").is_dir()
    )

    return trials


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
                attempt_dir = trial_dir / "artifacts" / "attempt_1"
                manifest_path = attempt_dir / "run_manifest.json"
                if not manifest_path.is_file():
                    raise SystemExit(f"Missing run_manifest.json for {material}: {manifest_path}")

                manifest = read_json_object(manifest_path)
                validate_manifest_material(manifest, material, manifest_path)

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
        raise SystemExit(
            f"Could not find any num_wann_ordered trial folders for {material} "
            f"under {', '.join(display_path(root) for root in RUN_ROOTS)}"
        )

    return cases


def resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def trial_case_from_trial_dir(
    material: str,
    trial_dir: Path,
    csv_row_number: int | None = None,
    csv_run_column: str | None = None,
) -> TrialCase:
    attempt_dir = trial_dir / "artifacts" / "attempt_1"
    if not attempt_dir.is_dir():
        raise SystemExit(f"Missing artifacts/attempt_1 for {material}: {trial_dir}")

    job_dir = trial_dir.parent
    manifest_path = attempt_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing run_manifest.json for {material}: {manifest_path}")

    manifest = read_json_object(manifest_path)
    validate_manifest_material(manifest, material, manifest_path)

    try:
        run_root = job_dir.parent.relative_to(ROOT)
    except ValueError:
        run_root = job_dir.parent

    job_metadata = {
        **parse_job_name(job_dir),
        "run_root": job_dir.parent.name,
        "run_root_path": display_path(job_dir.parent),
        "csv_source": (
            {
                "row_number": csv_row_number,
                "run_column": csv_run_column,
                "candidate_run_error_table": display_path(CANDIDATE_RUN_ERROR_TABLE),
            }
            if csv_row_number is not None and csv_run_column is not None
            else None
        ),
    }
    job_metadata["run_root"] = str(run_root)

    return TrialCase(
        material=material,
        job_dir=job_dir,
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        case_id=case_id_for(job_metadata, trial_dir),
        job_metadata=job_metadata,
        manifest=manifest,
    )


def collect_cases_from_candidate_run_error_table() -> list[TrialCase]:
    if not CANDIDATE_RUN_ERROR_TABLE.is_file():
        raise SystemExit(f"Missing candidate run table: {CANDIDATE_RUN_ERROR_TABLE}")

    cases: list[TrialCase] = []
    with CANDIDATE_RUN_ERROR_TABLE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            column
            for column in ("candidate_material", *CANDIDATE_RUN_COLUMNS)
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise SystemExit(
                f"{CANDIDATE_RUN_ERROR_TABLE} is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            material = (row.get("candidate_material") or "").strip()
            if not material:
                raise SystemExit(
                    f"Missing candidate_material in {CANDIDATE_RUN_ERROR_TABLE} row {row_number}"
                )

            for run_column in CANDIDATE_RUN_COLUMNS:
                run_path_text = (row.get(run_column) or "").strip()
                if not run_path_text:
                    continue

                trial_dir = resolve_existing_path(run_path_text)
                if not trial_dir.is_dir():
                    raise SystemExit(
                        f"{run_column} in {CANDIDATE_RUN_ERROR_TABLE} row {row_number} "
                        f"does not point to a directory: {trial_dir}"
                    )

                cases.append(
                    trial_case_from_trial_dir(
                        material=material,
                        trial_dir=trial_dir,
                        csv_row_number=row_number,
                        csv_run_column=run_column,
                    )
                )

    if not cases:
        raise SystemExit(f"No non-empty run_1/run_2/run_3 entries in {CANDIDATE_RUN_ERROR_TABLE}")

    return cases


def dataset_task_instruction_path(material: str) -> Path:
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

    raise SystemExit(
        f"Could not find original task instructions for {material} in {material_dir}. "
        "Expected one of: instruction.md, instructions.md, prompt.md, task.md, "
        "or a unique *instruction*.md file."
    )


def first_user_message_from_trajectory(trial_dir: Path) -> str | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.is_file():
        return None

    try:
        data = read_json(trajectory_path)
    except json.JSONDecodeError:
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
    return instruction_path.read_text(encoding="utf-8"), display_path(instruction_path)


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


def prompt_text(case: TrialCase) -> str:
    material = case.material
    num_wann = case.job_metadata.get("num_wann_from_folder")
    target_start = case.manifest.get("target_dft_band_start")
    target_end = case.manifest.get("target_dft_band_end")

    return f"""# Gemini Self-Debug Review: {material}

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
choices that led to  given the information available at the time, say so and do not force a 
critique. Only identify avoidable mistakes when the task materials, trajectory, 
logs, or final diagnostics provide evidence that a specific choice was not ideal/poor, 
contradicted by later run output, or led to an avoidable failure or high RMSE. 
For any such issue, explain what a scientifically better second try would have 
changed, using only information that would have been available from the task 
materials and old run logs. 

Read these files:

- `case_files/case_metadata.json`
- `case_files/verifier/diagnostics.json` if present
- `case_files/artifacts/attempt_1/{material}.win`
- `case_files/artifacts/attempt_1/{material}.wout`
- `case_files/artifacts/attempt_1/run_manifest.json`
- `case_files/agent/trajectory.json`
- `case_files/agent/gemini-cli.trajectory.jsonl` if present
- `case_files/agent/gemini-cli.txt` if present
- `case_files/original_task_instructions.md`

Do not read or rely on these aggregate/reference-analysis files (they are outdated):

- `jobs/num_wann_ordered_diagnostics_summary.json`
- `jobs/gemini_vs_reference_errors.xlsx`
- `jobs/gemini_failure_modes/failure_modes.csv`

The per-run verifier diagnostics are allowed only as scalar final outcome
metrics for this specific run. They do not provide the hidden reference recipe
or the Gemini/reference RMSE ratio. Do not recommend "copy the reference",
SCDM, `use_ws_distance`, or any other reference-only
setting unless the original task materials made that option available. If a
reference setting is not available under the original instructions, say so
explicitly and give a non-reference second-attempt change instead. Any proposed
“better second try” must still obey 
case_files/original_task_instructions.md. Do not propose changes that would 
violate the original task prompt, including fixed num_bands, fixed target-band 
requirements, required artifact/status rules, forbidden external lookups, or
any other original instruction constraints.

Do not handwave from aggregate statistics. The core diagnosis must
come from this material's `.win`, `.wout`, run manifest, trajectory, and
per-run verifier diagnostics, if present.

Write exactly these two files:

- `self_debug_report.md`
- `self_debug_report.json`

The Markdown report must be step-by-step and specific. For each substantive
decision in the old trajectory, judge whether it was good, bad, mixed, or
uncertain, and explain why using concrete evidence from `.win`, `.wout`,
manifest notes, trajectory reasoning, and final error metrics. Cite the file
paths you used, and include line numbers when you have them from grep, rg, nl,
or similar inspection. Cover at least:

1. projection choice
2. `num_wann` / target-band handling;
3. `num_bands` / band-pool handling;
4. disentanglement outer and frozen windows;
5. response to Wannier90 warnings or iteration caps;
6. localization quality from WF spreads and spread components;
7. whether the old run accepted a result it should have rejected;

If the run shows evidence of avoidable issues, also cover:

8. the most likely specific failure chain;
9. what should be done differently next time.

For every decision review, answer all of these forensic questions:

- What exactly did the old run decide or claim?
- What evidence did the old run use at the time?
- What later evidence in `.wout`, verifier diagnostics, or trajectory contradicts
  or weakens that decision?
- Was the mistake avoidable without seeing the hidden reference recipe?
- What concrete second-attempt change should have been tried instead, staying
  WITHIN the original task instructions?
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
- recommendations that only say "increase iterations" when the logs show a
  projection, window, validation, or workflow-decision problem.

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
  "projection_verdict": "not_used | bad | uncertain",
  "decision_reviews": [
    {{
      "decision": "short name",
      "verdict": "good | mixed | bad | uncertain",
      "evidence": ["specific file-backed evidence"],
      "old_claim_or_decision": "what the old run said or did",
      "observed_failure_signal": "what later output showed",
      "why": "specific explanation",
      "better_choice": "concrete second-attempt change if the decision was an avoidable issue, otherwise null"
    }}
  ],
  "failure_chain": ["ordered specific causes, or empty if no avoidable failure chain is supported"],
  "recommended_next_run_changes": ["specific changes, or empty if the trajectory was reasonable and no supported changes are warranted"]
}}
```

Your final response should be a concise JSON object pointing to
`self_debug_report.md` and `self_debug_report.json`.
"""

def build_case(case: TrialCase) -> Path:
    material = case.material
    case_dir = OUTPUT_ROOT / material / case.case_id
    clean_dir(case_dir)
    case_files = case_dir / "case_files"

    copy_file(
        find_attempt_file(case.attempt_dir, material, ".win"),
        case_files / "artifacts" / "attempt_1" / f"{material}.win",
    )
    copy_file(
        find_attempt_file(case.attempt_dir, material, ".wout"),
        case_files / "artifacts" / "attempt_1" / f"{material}.wout",
    )
    copy_file(
        case.attempt_dir / "run_manifest.json",
        case_files / "artifacts" / "attempt_1" / "run_manifest.json",
    )
    copy_file(case.trial_dir / "agent" / "trajectory.json", case_files / "agent" / "trajectory.json")

    for optional in ("gemini-cli.trajectory.jsonl", "gemini-cli.txt"):
        src = case.trial_dir / "agent" / optional
        if src.exists():
            copy_file(src, case_files / "agent" / optional)

    diagnostics_src = case.trial_dir / "verifier" / "diagnostics.json"
    if diagnostics_src.is_file():
        copy_file(diagnostics_src, case_files / "verifier" / "diagnostics.json")

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
        "verifier_diagnostics_copied": diagnostics_src.is_file(),
        "verifier_diagnostics_source": display_path(diagnostics_src) if diagnostics_src.is_file() else None,
        "aggregate_inputs_intentionally_not_copied": [
            "jobs/num_wann_ordered_diagnostics_summary.json",
            "jobs/gemini_vs_reference_errors.xlsx",
            "jobs/gemini_failure_modes/failure_modes.csv",
        ],
    }
    write_text(case_files / "case_metadata.json", json.dumps(metadata, indent=2) + "\n")

    write_text(case_dir / "prompt.md", prompt_text(case))
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
    if not isinstance(data.get("recommended_next_run_changes"), list):
        return False

    return True


def run_gemini(case_dir: Path) -> None:
    prompt = (case_dir / "prompt.md").read_text(encoding="utf-8")
    combined_log_path = case_dir / "gemini_stdout_stderr.txt"
    status_path = case_dir / "run_status.json"

    command = [
        GEMINI_BIN,
        "--yolo",
        "--skip-trust",
        f"--model={MODEL}",
        f"--prompt={prompt}",
    ]

    attempts: list[dict[str, Any]] = []
    max_attempts = 10

    for attempt_index in range(1, max_attempts + 1):
        # Remove stale outputs so success must come from this attempt.
        for output_name in ("self_debug_report.md", "self_debug_report.json"):
            output_path = case_dir / output_name
            if output_path.exists():
                output_path.unlink()

        attempt_log_path = case_dir / f"gemini_attempt_{attempt_index:02d}_stdout_stderr.txt"

        completed = subprocess.run(
            command,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stdout_stderr = completed.stdout or ""
        attempt_log_path.write_text(stdout_stderr, encoding="utf-8")

        produced_nonempty_diagnosis = report_is_nonempty(case_dir)

        attempt_record = {
            "attempt": attempt_index,
            "returncode": completed.returncode,
            "attempt_log_path": attempt_log_path.name,
            "self_debug_report_md_exists": (case_dir / "self_debug_report.md").is_file(),
            "self_debug_report_json_exists": (case_dir / "self_debug_report.json").is_file(),
            "produced_nonempty_diagnosis": produced_nonempty_diagnosis,
        }
        attempts.append(attempt_record)

        status = {
            "command": command,
            "max_attempts": max_attempts,
            "success": produced_nonempty_diagnosis,
            "attempts": attempts,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

        if produced_nonempty_diagnosis:
            combined_log_path.write_text(stdout_stderr, encoding="utf-8")
            return

        print(
            f"Gemini attempt {attempt_index}/{max_attempts} did not produce "
            f"a non-empty diagnosis for {case_dir.name}; retrying..."
        )

    combined_log_path.write_text(
        "\n\n".join(
            [
                f"===== ATTEMPT {record['attempt']} "
                f"returncode={record['returncode']} =====\n"
                f"{(case_dir / record['attempt_log_path']).read_text(encoding='utf-8')}"
                for record in attempts
            ]
        ),
        encoding="utf-8",
    )

    raise SystemExit(
        f"Gemini review failed after {max_attempts} attempts or did not write "
        f"a non-empty diagnosis: {case_dir}"
    )


def output_dir_for_case(case: TrialCase) -> Path:
    return OUTPUT_ROOT / case.material / case.case_id


def run_case(case: TrialCase) -> Path:
    case_dir = build_case(case)
    print(
        "Running Gemini review for "
        f"{case.material} case={case.case_id} in {case_dir}"
    )
    run_gemini(case_dir)
    return case_dir


def collect_cases() -> list[TrialCase]:
    all_cases: list[TrialCase] = []

    if USE_CANDIDATE_RUN_ERROR_TABLE:
        all_cases = collect_cases_from_candidate_run_error_table()
        print(
            f"Loaded {len(all_cases)} explicit trial folder(s) from "
            f"{display_path(CANDIDATE_RUN_ERROR_TABLE)}."
        )
    else:
        for material in MATERIALS:
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
    if shutil.which(GEMINI_BIN) is None:
        raise SystemExit(
            f"Could not find {GEMINI_BIN!r} on PATH. Edit GEMINI_BIN at the top of this script "
            "or run it in the same environment where Gemini CLI is installed."
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = collect_cases()

    if not cases:
        print("No cases to review.")
        return

    max_workers = min(MAX_CONCURRENT_GEMINI, len(cases))
    print(f"Running {len(cases)} Gemini review(s) with concurrency={max_workers}.")

    failures: list[tuple[TrialCase, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_case = {pool.submit(run_case, case): case for case in cases}

        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                case_dir = future.result()
            except BaseException as exc:
                failures.append((case, exc))
                print(f"FAILED Gemini review for {case.material} case={case.case_id}: {exc}")
                continue

            print(f"Wrote expected outputs in {case_dir}")

    if failures:
        details = "\n".join(
            f"- {case.material} case={case.case_id}: {exc}"
            for case, exc in failures
        )
        raise SystemExit(
            f"{len(failures)} Gemini review(s) failed out of {len(cases)}:\n{details}"
        )


if __name__ == "__main__":
    main()
