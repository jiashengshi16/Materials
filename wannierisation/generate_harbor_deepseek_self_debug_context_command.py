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
    self_debug_generator.ROOT / "jobsDeepseekProTerminus2CodexSelfReview"
    if WORKFLOW == "codex_self_review"
    else self_debug_generator.ROOT / "jobsDeepseekProTerminus2SelfDebugContext"
)
DEFAULT_SELF_DEBUG_REVIEWS_ROOT = (
    DEEPSEEK_SELF_DEBUG_REVIEWS_ROOT
)

# Leave empty to use all materials that have DeepSeek self-debug reports.
MATERIALS: list[str] = [
    # "Al4Mn2O8",
    # "Al4O8Zn2",
    # "Al4Sc2",
    # "Al8Zr4",
    # "Ar2",
    # "C2Cd2O6",
    # "C2Cu2O6",
    # "Hg3O3",
]

NEXT_RUN_TRACE_ARTIFACTS = [
    "/app/workflow/next_run_file_trace.log",
    "/app/workflow/NEXT_RUN_CONTEXT_SUMMARY.json",
]
NEXT_RUN_TRACE_WRAPPER_NAME = "trace_next_run_file_access.sh"
NEXT_RUN_TRACE_VERIFIER_NAME = "verify_next_run_context_access.py"
NEXT_RUN_TRACE_WRAPPER_APP_PATH = "/app/trace_next_run_file_access.sh"


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
    return """# Auto-start Terminus login shells under the configured file-access tracer.
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
        f"RUN chmod +x /app/{NEXT_RUN_TRACE_WRAPPER_NAME} "
        f"/app/{NEXT_RUN_TRACE_VERIFIER_NAME} && "
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
    if "harbor-agent-trace.sh" not in dockerfile_text:
        additions += profile_hook
    if not additions:
        return dockerfile_text

    marker = "COPY material /app/material\n"
    if marker in dockerfile_text:
        return dockerfile_text.replace(marker, additions + marker, 1)
    return dockerfile_text + "\n" + additions


def install_next_run_trace_tools(tasks: list[tuple[int, str, Path]]) -> None:
    for _num_wann, _material, task_dir in tasks:
        environment_dir = task_dir / "environment"
        wrapper_path = environment_dir / NEXT_RUN_TRACE_WRAPPER_NAME
        wrapper_path.write_text(next_run_trace_wrapper_script(), encoding="utf-8")
        wrapper_path.chmod(0o755)

        verifier_path = environment_dir / NEXT_RUN_TRACE_VERIFIER_NAME
        verifier_path.write_text(next_run_trace_verifier_script(), encoding="utf-8")
        verifier_path.chmod(0o755)

        dockerfile_path = environment_dir / "Dockerfile"
        dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
        dockerfile_path.write_text(
            inject_next_run_trace_tools_into_dockerfile(dockerfile_text),
            encoding="utf-8",
        )


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
