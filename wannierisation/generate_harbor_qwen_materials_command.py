#!/usr/bin/env python3
"""
export DASHSCOPE_API_KEY='sk-ws-H.IXHHLI.DEgO.MEUCIQD8CpneKErwSi9ZQfLGr3Zn-9CxX7ol68K39sLCJA4s7gIgeYIluCto8WtKrypGHTOmiGwk2b4FPtB8i1hz2PUkHDs'
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import generate_harbor_num_wann_order_command as harbor_generator


# Edit only this list. Names must exactly match directories in wannier_200.
MATERIALS = [
    'AsB',
    #'BrLi',
    'He',
    'FLi',
    #'Ne'
    # 'BrNa' 

]

# This endpoint is for an Alibaba Cloud Model Studio key created in Singapore.
# Override it without editing this file by exporting QWEN_BASE_URL first.
DEFAULT_QWEN_BASE_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
MODEL = "qwen3-coder-480b-a35b-instruct"
MAX_MATERIALS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate num_wann-ordered Harbor runs using Qwen Code."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=harbor_generator.DEFAULT_DATASET,
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


def main() -> None:
    cli = parse_args()
    materials = [name.strip() for name in MATERIALS if name.strip()]

    if not materials:
        raise SystemExit(
            "MATERIALS is empty. Edit this new script and add up to 10 directory names."
        )
    if len(materials) > MAX_MATERIALS:
        raise SystemExit(f"MATERIALS may contain at most {MAX_MATERIALS} entries.")
    if len(materials) != len(set(materials)):
        raise SystemExit("MATERIALS contains duplicate entries.")
    if cli.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

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

    # Match the Namespace consumed by the shared command builder. Its default
    # artifact list intentionally skips /app/artifacts to avoid a duplicate
    # artifacts/artifacts export.
    args = argparse.Namespace(
        dataset=cli.dataset,
        agent="qwen-coder",  # Harbor's name for its native Qwen Code adapter.
        model=MODEL,
        n_concurrent=1,
        batch_size=cli.batch_size,
        stop_on_error=cli.stop_on_error,
        docker_prune_after_batch=False,
        docker_prune_after_material=False,
        delete_after_run=True,
        extra_arg=[
            "--agent-kwarg",
            f"prompt_template_path={harbor_generator.ROOT / 'scripts' / 'qwen_execution_prompt.j2'}",
            "--jobs-dir",
            str(harbor_generator.ROOT / "jobsQwen"),
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
        jobs_root=harbor_generator.ROOT / "jobsQwen",
        no_gemini_cached_defaults=True,
        gemini_ipv4_first=False,
        no_gemini_run_timeout=True,
        no_gemini_host_network=True,
    )

    # These lines are evaluated by the caller before the generated Harbor runs.
    # Parameter expansion leaves the secret out of this script's stdout.
    print(': "${DASHSCOPE_API_KEY:?Export DASHSCOPE_API_KEY before running}"')
    print('export OPENAI_API_KEY="$DASHSCOPE_API_KEY"')
    print(
        'export OPENAI_BASE_URL="${QWEN_BASE_URL:-'
        + DEFAULT_QWEN_BASE_URL
        + '}"'
    )
    harbor_generator.print_ordered_commands(args, tasks)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
