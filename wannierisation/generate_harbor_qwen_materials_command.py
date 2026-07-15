#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import generate_harbor_num_wann_order_command as harbor_generator

# MATERIALS = [
#     'Al18Co4',
#     'Al4Mn2O8',
#     'Al4O8Zn2',
#     'Al4Sc2',
#     'Al8Zr4',
#     'Ar2',
#     'Au2Sc',
#     'B2Cr',
#     'B2Hf',
#     'Bi4Cl12',
#     'C2Cd2O6',
#     'C2Cu2O6',
#     'C4O12Sr4',
#     'Cl2V',
#     'Cl4Li4O16',
#     'Co4Sc8',
#     'Cr2F4',
#     'Cr6Si2',
#     'H8O16W4',
#     'He',
#     'Hf10Si6',
#     'Hg3O3',
#     'Mg2O10Ti4',
#     'N2Na2O6',
#     'NNb',
#     'Ni4Zr4',
#     'O2Pd2',
#     'Pd4S8',
#     'RuTi',
#     wild cards here
#     'AgMg',
#     'FNa',
#     'Ne',
# ]

MATERIALS = [
'Al12Ni4',
'Al4Y2',
'Kr2',
'Au2Y',
'Ag2Sc',
'Ag2Y',
'B2Mn',
'B2Ta',
'B2Ti',
'O2Sr',
'Br2V',
'Cl2Ti',
'Mg4O12Se4',
'Li4O6Si2',
'F4Ni2',
'Co2F4',
'Cr6Ga2',
'Mo6Si2',
'Al2Mo6',
'Ga2Mo6',
'B8H16O16',
'Hf6Si4',
'Hf4Si2',
'Si6Y10',
'Co2O8W2',
'CTi',
'Hf4Ni4',
'Pt4Y4',
'O2Pb2',
'Ru4S8',
'Co4S8',
'FeTi',
'RuZr',
'RhSc',
'FLi',
'BrNa',
'AgSc',
]


#Deepseek
"""
export OPENAI_API_KEY="sk-your-new-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
"""

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

MODEL = "openai/deepseek-v4-pro"


#GLM Alibaba
# DEFAULT_DEEPSEEK_BASE_URL = (
#     "https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
#MODEL = "deepseek-v4-flash"


#GLM Z.ai
"""
export OPENAI_API_KEY="your-z-ai-api-key"
export OPENAI_BASE_URL="https://api.z.ai/api/paas/v4"
"""
# DEFAULT_DEEPSEEK_BASE_URL = (
#     "https://api.z.ai/api/paas/v4"
# )
# MODEL = "custom_openai/glm-4.5-air"


#ChatGPT

"""

"""
# DEFAULT_DEEPSEEK_BASE_URL = "https://api.openai.com/v1"
# MODEL = "gpt-5.4-mini"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate num_wann-ordered Harbor runs using OpenAI Codex."
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
    return parser.parse_args()

import json
from collections import Counter


def existing_run_counts(
    jobs_root: Path,
    valid_materials: set[str],
) -> Counter[str]:
    """Count existing completed Harbor runs, regardless of verifier status."""
    counts: Counter[str] = Counter()

    if not jobs_root.is_dir():
        return counts

    # Count each top-level Harbor job directory at most once.
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue

        material: str | None = None

        # First try the material embedded in diagnostics.json.
        for diagnostics_path in job_dir.rglob("diagnostics.json"):
            if diagnostics_path.parent.name != "verifier":
                continue
            if "randprojections" in diagnostics_path.parts:
                continue

            try:
                data = json.loads(
                    diagnostics_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(data, dict):
                candidate = (
                    data.get("material")
                    or data.get("material_from_folder")
                )
                if candidate in valid_materials:
                    material = candidate
                    break

        # Fall back to identifying the material from the job directory name.
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

    args = argparse.Namespace(
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
            "1.1",
            "--max-retries",
            "2",
            "--retry-include",
            "AgentSetupTimeoutError",
            "--retry-include",
            "NonZeroAgentExitCodeError",
        ],
        artifact=[],
        no_default_artifacts=False,
        save_generated_qe_save=False,
        jobs_root=harbor_generator.ROOT / "jobsDeepseekProTerminus2Candidates",

        target_success_runs=2 if cli.target_runs is None else None,
        target_runs=cli.target_runs,

        validate_new_success=False,
        max_attempts_per_needed_success=0,
        delete_failed_attempt_folders=False,
        success_wave_timeout_sec=4500,
        success_wave_kill_after_sec=30,
        success_roots=[harbor_generator.ROOT / "jobsDeepseekProTerminus2Candidates"],
        include_result_dir_name=[],
        least_success_first=False,
        no_gemini_cached_defaults=True,
        gemini_ipv4_first=False,
        no_gemini_run_timeout=True,
        no_gemini_host_network=True,
    )

    # print(': "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY before running}"')
    # print('export OPENAI_API_KEY="$DASHSCOPE_API_KEY"')
    print(': "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running}"')
    print(
        'export OPENAI_BASE_URL="${OPENAI_BASE_URL:-'
        + DEFAULT_DEEPSEEK_BASE_URL
        + '}"'
    )
    if cli.target_runs is not None:
        counts = existing_run_counts(
            args.jobs_root,
            valid_materials=requested,
        )

        repeated_tasks = []

        for task in tasks:
            _num_wann, material, _source = task
            existing = counts[material]
            needed = max(0, cli.target_runs - existing)

            print(
                f"# {material}: existing={existing}, "
                f"target={cli.target_runs}, new={needed}"
            )

            repeated_tasks.extend([task] * needed)

        if repeated_tasks:
            harbor_generator.print_ordered_commands(args, repeated_tasks)
        else:
            print("# Every material already has the requested number of runs.")
            print("true")
    else:
        harbor_generator.print_target_success_loop(args, tasks)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
