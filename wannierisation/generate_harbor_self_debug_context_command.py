#!/usr/bin/env python3
"""Print Harbor commands that run tasks in increasing instruction num_wann."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from datetime import datetime
import json
import errno
import os
import re
import shlex
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
SELF_DEBUG_REVIEWS_ROOT = ROOT / "jobs" / "gemini_self_debug_reviews_chemical_similarity"
DEFAULT_CANDIDATE_RUN_ERROR_TABLE = ROOT / "candidate_run_error_table.csv"
DEFAULT_CANDIDATE_SELF_DEBUG_REVIEWS_ROOT = ROOT / "jobs" / "gemini_self_debug_reviews_chemical_similarity"
DEFAULT_AUGMENTED_DATASET_PARENT = ROOT / "harbor_datasets"
NUM_WANN_RE = re.compile(r"\bnum_wann\s*=\s*(\d+)\b")
# The task test wrapper already copies /app/artifacts/. into /logs/artifacts,
# which Harbor stores as the trial's top-level artifacts directory. Exporting
# /app/artifacts again creates a duplicate artifacts/artifacts tree.
DEFAULT_ARTIFACTS = ["/app/report.json", "/app/REPORT.md"]
SELF_DEBUG_TRACE_ARTIFACTS = [
    "/app/workflow/gemini_file_trace.log",
    "/app/workflow/SELF_DEBUG_CONTEXT_SUMMARY.json",
]
TRACE_WRAPPER_NAME = "trace_agent_file_access.sh"
TRACE_VERIFIER_NAME = "verify_self_debug_context_access.py"
TRACE_WRAPPER_APP_PATH = "/app/trace_agent_file_access.sh"
DEFAULT_TRACE_AGENT_WRAPPER_ENV = "HARBOR_AGENT_COMMAND_WRAPPER"
QE_SAVE_EXPORT_ARTIFACTS = [
    "/app/workflow/run_dir/out",
    "/app/workflow/run_dir/scf.out",
    "/app/workflow/run_dir/nscf.out",
]
DEFAULT_N_CONCURRENT = 8
QE_SAVE_RELATIVE_PATH = Path("environment") / "material" / "qe_save"
QE_SAVE_INSTALLER = ROOT / "scripts" / "install_harbor_qe_save.py"
HOST_NETWORK_COMPOSE = ROOT / "docker" / "harbor-host-network.compose.yml"
GEMINI_IPV4_NODE_OPTIONS = "NODE_OPTIONS=--dns-result-order=ipv4first"
GEMINI_RUN_TIMEOUT_ENV = "HARBOR_GEMINI_RUN_TIMEOUT_SEC=4500"
CACHED_GEMINI_AGENT_IMPORT = "harbor_agents.cached_gemini_cli:CachedGeminiCli"
DEFAULT_GEMINI_AGENT_TIMEOUT_MULTIPLIER = "1.1"
DEFAULT_MAX_RETRIES = "2"
DEFAULT_RETRY_INCLUDES = ["AgentSetupTimeoutError", "NonZeroAgentExitCodeError"]
DOCKER_SYSTEM_PRUNE_COMMAND = ["docker", "system", "prune", "--force", '--all']
DEFAULT_SUCCESS_ROOTS = [SELF_DEBUG_REVIEWS_ROOT]
DEFAULT_EXCLUDED_RESULT_DIR_NAMES = {"randprojections", "case_files"}


def relpath_for_command(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def read_num_wann(task_dir: Path) -> int:
    instruction = task_dir / "instruction.md"
    if not instruction.exists():
        raise SystemExit(f"missing instruction.md: {instruction}")

    match = NUM_WANN_RE.search(instruction.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"could not find 'num_wann = <int>' in {instruction}")
    return int(match.group(1))


def material_from_record(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    material = record.get("material") or record.get("material_from_folder")
    if isinstance(material, str) and material:
        return material
    return None


def material_from_diagnostics_path(path: Path, valid_materials: set[str]) -> str | None:
    for parent in path.parents:
        if parent.name in valid_materials:
            return parent.name
        prefix = parent.name.split("__", 1)[0]
        if prefix in valid_materials:
            return prefix
        match = re.search(r"__num_wann_\d+__(?P<material>[^/]+)$", parent.name)
        if match and match.group("material") in valid_materials:
            return match.group("material")
    return None


def read_json(path: Path) -> object:
    if not path.is_file():
        raise SystemExit(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def successful_run_counts(
    roots: list[Path],
    *,
    valid_materials: set[str],
    excluded_dir_names: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for root in roots:
        if not root.exists():
            continue
        if not root.is_dir():
            raise SystemExit(f"success-count root is not a directory: {root}")
        for diagnostics_path in root.rglob("diagnostics.json"):
            if diagnostics_path.parent.name != "verifier":
                continue
            if excluded_dir_names.intersection(diagnostics_path.parts):
                continue
            try:
                data = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or data.get("status") != "success":
                continue
            material = material_from_record(data) or material_from_diagnostics_path(
                diagnostics_path,
                valid_materials,
            )
            if material in valid_materials:
                counts[material] += 1
    return counts

def failed_or_unknown_materials_from_diagnostics(path: Path) -> set[str]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"expected a JSON object in {path}")

    failed_or_unknown = data.get("failed_or_unknown")
    if not isinstance(failed_or_unknown, list):
        raise SystemExit(f"expected {path} to contain a failed_or_unknown list")

    return {
        material
        for record in failed_or_unknown
        if (material := material_from_record(record)) is not None
    }

def material_names_from_results(path: Path) -> set[str]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"expected {path} to contain a results list")
    return {
        material
        for record in results
        if (material := material_from_record(record)) is not None
    }


def has_qe_save(task_dir: Path) -> bool:
    return (task_dir / QE_SAVE_RELATIVE_PATH).is_dir()


def has_agent_node_options(extra_args: list[str]) -> bool:
    for index, arg in enumerate(extra_args):
        if arg.startswith("--agent-env=NODE_OPTIONS=") or arg.startswith("--ae=NODE_OPTIONS="):
            return True
        if arg in {"--agent-env", "--ae"} and index + 1 < len(extra_args):
            if extra_args[index + 1].startswith("NODE_OPTIONS="):
                return True
    return False


def has_agent_env(extra_args: list[str], key: str) -> bool:
    prefix = f"{key}="
    for index, arg in enumerate(extra_args):
        if arg.startswith(f"--agent-env={prefix}") or arg.startswith(f"--ae={prefix}"):
            return True
        if arg in {"--agent-env", "--ae"} and index + 1 < len(extra_args):
            if extra_args[index + 1].startswith(prefix):
                return True
    return False


def has_option(extra_args: list[str], names: set[str]) -> bool:
    for arg in extra_args:
        if arg in names:
            return True
        if any(arg.startswith(f"{name}=") for name in names):
            return True
    return False


def has_extra_docker_compose(extra_args: list[str]) -> bool:
    return has_option(extra_args, {"--extra-docker-compose"})


def retry_includes(extra_args: list[str]) -> set[str]:
    values: set[str] = set()
    for index, arg in enumerate(extra_args):
        if arg.startswith("--retry-include="):
            values.add(arg.split("=", 1)[1])
        elif arg == "--retry-include" and index + 1 < len(extra_args):
            values.add(extra_args[index + 1])
    return values

def is_gemini_agent(agent: str) -> bool:
    return agent in {"gemini-cli", CACHED_GEMINI_AGENT_IMPORT}

def gemini_default_extra_args(args: argparse.Namespace) -> list[str]:
    if not is_gemini_agent(args.agent) or args.no_gemini_cached_defaults:
        return []

    extra: list[str] = []

    if not has_option(args.extra_arg, {"--agent-timeout-multiplier"}):
        extra.extend(["--agent-timeout-multiplier", DEFAULT_GEMINI_AGENT_TIMEOUT_MULTIPLIER])

    if args.target_success_runs is None and not has_option(args.extra_arg, {"--max-retries", "-r"}):
        extra.extend(["--max-retries", DEFAULT_MAX_RETRIES])

    existing_retries = retry_includes(args.extra_arg)
    if args.target_success_runs is None:
        for exception in DEFAULT_RETRY_INCLUDES:
            if exception not in existing_retries:
                extra.extend(["--retry-include", exception])

    return extra


def dataset_tasks(
    dataset: Path,
    *,
    include_materials: set[str] | None = None,
    exclude_materials: set[str] | None = None,
    require_qe_save: bool = False,
) -> list[tuple[int, str, Path]]:
    if not dataset.is_dir():
        raise SystemExit(f"dataset directory does not exist: {dataset}")

    tasks: list[tuple[int, str, Path]] = []
    for task_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        material = task_dir.name
        if include_materials is not None and material not in include_materials:
            continue
        if exclude_materials is not None and material in exclude_materials:
            continue
        if require_qe_save and not has_qe_save(task_dir):
            continue
        tasks.append((read_num_wann(task_dir), material, task_dir))
    return sorted(tasks)

def self_debug_reports_for_material(material: str, root: Path | None = None) -> list[tuple[str, Path, Path]]:
    root = SELF_DEBUG_REVIEWS_ROOT if root is None else root
    material_root = root / material
    if not material_root.is_dir():
        return []

    reports: list[tuple[str, Path, Path]] = []
    for case_dir in sorted(path for path in material_root.iterdir() if path.is_dir()):
        md_path = case_dir / "self_debug_report.md"
        json_path = case_dir / "self_debug_report.json"
        if md_path.is_file() and json_path.is_file():
            reports.append((case_dir.name, md_path, json_path))
    return reports


def materials_with_self_debug_reports() -> set[str]:
    if not SELF_DEBUG_REVIEWS_ROOT.is_dir():
        return set()

    materials: set[str] = set()
    for material_dir in sorted(path for path in SELF_DEBUG_REVIEWS_ROOT.iterdir() if path.is_dir()):
        if self_debug_reports_for_material(material_dir.name):
            materials.add(material_dir.name)
    return materials


def candidate_materials_from_table(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"candidate run-error table does not exist: {path}")

    candidates_by_material: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"material", "candidate_material"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise SystemExit(
                f"{path} must contain CSV columns: {', '.join(sorted(required_columns))}"
            )
        for row in reader:
            material = (row.get("material") or "").strip()
            candidate_material = (row.get("candidate_material") or "").strip()
            if not material or not candidate_material:
                continue
            candidates = candidates_by_material.setdefault(material, [])
            if candidate_material not in candidates:
                candidates.append(candidate_material)
    return candidates_by_material


def safe_context_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "case"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def self_debug_source_files(
    reports: list[tuple[str, Path, Path]],
    candidate_reports: dict[str, list[tuple[str, Path, Path]]] | None = None,
    *,
    include_same_material_reports: bool = True,
) -> list[dict[str, object]]:
    """Return every self-debug file the agent must read, in deterministic order."""
    candidate_reports = candidate_reports or {}
    records: list[dict[str, object]] = []

    if include_same_material_reports:
        for case_name, md_path, json_path in reports:
            safe_case = safe_context_name(case_name)
            for file_kind, src_path in (
                ("self_debug_report_md", md_path),
                ("self_debug_report_json", json_path),
            ):
                rel_path = Path("raw") / "same_material" / safe_case / src_path.name
                records.append(
                    {
                        "scope": "same_material",
                        "material": None,
                        "case": case_name,
                        "file_kind": file_kind,
                        "source_path": str(src_path),
                        "bundle_path": str(Path("self_debug_context") / rel_path),
                        "app_path": str(Path("/app/self_debug_context") / rel_path),
                        "sha256": sha256_file(src_path),
                    }
                )

    for candidate_material, reports_for_candidate in sorted(candidate_reports.items()):
        safe_material = safe_context_name(candidate_material)
        for case_name, md_path, json_path in reports_for_candidate:
            safe_case = safe_context_name(case_name)
            for file_kind, src_path in (
                ("self_debug_report_md", md_path),
                ("self_debug_report_json", json_path),
            ):
                rel_path = Path("raw") / "candidate" / safe_material / safe_case / src_path.name
                records.append(
                    {
                        "scope": "candidate_material",
                        "material": candidate_material,
                        "case": case_name,
                        "file_kind": file_kind,
                        "source_path": str(src_path),
                        "bundle_path": str(Path("self_debug_context") / rel_path),
                        "app_path": str(Path("/app/self_debug_context") / rel_path),
                        "sha256": sha256_file(src_path),
                    }
                )

    return records


def write_self_debug_context_bundle(
    target_root: Path,
    *,
    material: str,
    records: list[dict[str, object]],
) -> None:
    """Create one mandatory read bundle, an index, and raw copied files."""
    bundle_root = target_root / "self_debug_context"
    bundle_root.mkdir(parents=True, exist_ok=True)

    rendered: list[str] = [
        "# REQUIRED SELF-DEBUG CONTEXT BUNDLE",
        "",
        f"Target material: `{material}`",
        f"Expected self-debug file count: {len(records)}",
        "",
        "The agent must read every file section below before choosing projections, windows, target-band handling, or final status.",
        "This bundle is generated from `self_debug_context/index.json`; do not treat a subset as representative.",
        "",
    ]

    for index, record in enumerate(records, start=1):
        source_path = Path(str(record["source_path"]))
        bundle_path = Path(str(record["bundle_path"]))
        copy_target = target_root / bundle_path
        copy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copy_target)

        body = source_path.read_text(encoding="utf-8", errors="replace")
        fence = "```json" if source_path.suffix == ".json" else "```markdown"
        rendered.extend(
            [
                f"## Self-debug file {index} of {len(records)}",
                "",
                f"- scope: `{record['scope']}`",
                f"- candidate_material: `{record['material']}`",
                f"- case: `{record['case']}`",
                f"- file_kind: `{record['file_kind']}`",
                f"- app_path: `{record['app_path']}`",
                f"- sha256: `{record['sha256']}`",
                "",
                fence,
                body.rstrip(),
                "```",
                "",
            ]
        )

    index_payload = {
        "target_material": material,
        "expected_file_count": len(records),
        "expected_report_pair_count": len(records) // 2,
        "required_summary_path": "workflow/SELF_DEBUG_CONTEXT_SUMMARY.json",
        "required_bundle_path": "/app/self_debug_context/ALL_SELF_DEBUG_REPORTS.md",
        "records": records,
    }
    (bundle_root / "index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (bundle_root / "ALL_SELF_DEBUG_REPORTS.md").write_text(
        "\n".join(rendered),
        encoding="utf-8",
    )


def context_instruction_appendix(
    material: str,
    reports: list[tuple[str, Path, Path]],
    candidate_reports: dict[str, list[tuple[str, Path, Path]]] | None = None,
    *,
    include_same_material_reports: bool = True,
    expected_self_debug_file_count: int = 0,
) -> str:
    candidate_reports = candidate_reports or {}
    if expected_self_debug_file_count == 0:
        return ""

    return f"""

# Mandatory Self-Debug Context Preflight

Before choosing the Wannierisation strategy, before writing the first `<seed>.win`,
and before creating `workflow/run_dir`, read the complete self-debug bundle:

`/app/self_debug_context/ALL_SELF_DEBUG_REPORTS.md`

This bundle contains **{expected_self_debug_file_count} required self-debug files**.
They are also enumerated in:

`/app/self_debug_context/index.json`

You must not sample these files. You must not read only the first few. You must not
infer that the remaining files are similar. Read every section in the bundle.
Use the reports only as forensic context about projection, window, convergence,
validation, and final-status failure modes; this task instruction remains
authoritative for material, num_wann, num_bands, target-band, artifact, and
status constraints.

After reading the bundle, and before any Wannier90/QE command, create:

`workflow/SELF_DEBUG_CONTEXT_SUMMARY.json`

It must be valid JSON with this shape:

```json
{{
  "target_material": "{material}",
  "expected_file_count": {expected_self_debug_file_count},
  "read_file_count": {expected_self_debug_file_count},
  "all_files_read": true,
  "files": [
    {{
      "app_path": "/app/self_debug_context/raw/.../self_debug_report.md",
      "sha256": "...",
      "key_failure_or_lesson": "...",
      "projection_or_window_implication": "...",
      "used_in_current_strategy": true
    }}
  ],
  "cross_report_lessons": [],
  "current_strategy_implications": []
}}
```

Hard gate: if `workflow/SELF_DEBUG_CONTEXT_SUMMARY.json` is missing, invalid,
claims `all_files_read != true`, or has `read_file_count != expected_file_count`,
do not proceed. Return `status: "failed"` and explain that the self-debug
preflight was incomplete.
"""



def trace_wrapper_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/workflow
: > /app/workflow/gemini_file_trace.log
if ! command -v strace >/dev/null 2>&1; then
  echo "ERROR: strace is not installed in the task image; cannot enforce self-debug context reads" >&2
  echo "ERROR: strace_missing" > /app/workflow/gemini_file_trace.log
  exit 127
fi
exec strace -f \
  -e trace=openat,open,read,stat,newfstatat,access \
  -s 300 \
  -o /app/workflow/gemini_file_trace.log \
  "$@"
"""


def trace_verifier_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_TRACE_PATHS = [
    \"/app/self_debug_context/ALL_SELF_DEBUG_REPORTS.md\",
    \"/app/self_debug_context/index.json\",
]


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding=\"utf-8\"))
    except Exception as exc:
        raise SystemExit(f\"failed to read JSON {path}: {exc}\")
    if not isinstance(data, dict):
        raise SystemExit(f\"expected JSON object in {path}\")
    return data


def trace_has_path_access(trace_text: str, app_path: str) -> bool:
    # strace read(2) lines do not include the pathname after a successful open, so
    # the reliable pathname evidence is open/openat/stat/access/newfstatat.
    escaped = re.escape(app_path)
    return re.search(r\"\\b(openat|open|stat|newfstatat|access)\\([^\\n]*\" + escaped, trace_text) is not None


def verify(index_path: Path, summary_path: Path, trace_path: Path) -> list[str]:
    errors: list[str] = []
    if not index_path.is_file():
        return [f\"missing index.json: {index_path}\"]
    if not summary_path.is_file():
        return [f\"missing SELF_DEBUG_CONTEXT_SUMMARY.json: {summary_path}\"]
    if not trace_path.is_file():
        return [f\"missing gemini_file_trace.log: {trace_path}\"]

    index = load_json(index_path)
    summary = load_json(summary_path)
    trace_text = trace_path.read_text(encoding=\"utf-8\", errors=\"replace\")
    if \"trace_wrapper_not_invoked\" in trace_text:
        errors.append(
            \"trace wrapper was not invoked; HARBOR_AGENT_COMMAND_WRAPPER was ignored or not supported\"
        )

    for required_path in REQUIRED_TRACE_PATHS:
        if not trace_has_path_access(trace_text, required_path):
            errors.append(f\"no OS trace evidence of opening/stat/access for {required_path}\")
    if \"read(\" not in trace_text:
        errors.append(\"trace contains no read(2) syscalls\")

    expected_file_count = index.get(\"expected_file_count\")
    if summary.get(\"expected_file_count\") != expected_file_count:
        errors.append(
            f\"summary expected_file_count={summary.get('expected_file_count')!r} \"
            f\"does not match index expected_file_count={expected_file_count!r}\"
        )
    if summary.get(\"read_file_count\") != expected_file_count:
        errors.append(
            f\"summary read_file_count={summary.get('read_file_count')!r} \"
            f\"does not match expected_file_count={expected_file_count!r}\"
        )
    if summary.get(\"all_files_read\") is not True:
        errors.append(\"summary all_files_read is not true\")

    index_records = index.get(\"records\")
    if not isinstance(index_records, list):
        errors.append(\"index records is missing or not a list\")
        index_records = []
    summary_files = summary.get(\"files\")
    if not isinstance(summary_files, list):
        errors.append(\"summary files is missing or not a list\")
        summary_files = []

    seen = {
        (item.get(\"app_path\"), item.get(\"sha256\"))
        for item in summary_files
        if isinstance(item, dict)
    }
    for record in index_records:
        if not isinstance(record, dict):
            errors.append(\"index contains a non-object record\")
            continue
        app_path = record.get(\"app_path\")
        sha256 = record.get(\"sha256\")
        if not app_path or not sha256:
            errors.append(f\"index record missing app_path or sha256: {record!r}\")
            continue
        if (app_path, sha256) not in seen:
            errors.append(f\"summary missing indexed file app_path/sha256: {app_path} {sha256}\")

    if isinstance(expected_file_count, int) and len(summary_files) < expected_file_count:
        errors.append(
            f\"summary files has {len(summary_files)} entries; expected at least {expected_file_count}\"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--index\", default=\"/app/self_debug_context/index.json\")
    parser.add_argument(\"--summary\", default=\"/app/workflow/SELF_DEBUG_CONTEXT_SUMMARY.json\")
    parser.add_argument(\"--trace\", default=\"/app/workflow/gemini_file_trace.log\")
    args = parser.parse_args()
    errors = verify(Path(args.index), Path(args.summary), Path(args.trace))
    if errors:
        print(\"SELF_DEBUG_CONTEXT_ACCESS_VERIFICATION_FAILED\", file=sys.stderr)
        for error in errors:
            print(f\"- {error}\", file=sys.stderr)
        return 1
    print(\"SELF_DEBUG_CONTEXT_ACCESS_VERIFICATION_OK\")
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
"""


def write_trace_tools(environment_dir: Path) -> None:
    wrapper_path = environment_dir / TRACE_WRAPPER_NAME
    wrapper_path.write_text(trace_wrapper_script(), encoding="utf-8")
    wrapper_path.chmod(0o755)
    verifier_path = environment_dir / TRACE_VERIFIER_NAME
    verifier_path.write_text(trace_verifier_script(), encoding="utf-8")
    verifier_path.chmod(0o755)


def inject_trace_tools_into_dockerfile(dockerfile_text: str) -> str:
    install_snippet = (
        "RUN if command -v apt-get >/dev/null 2>&1; then "
        "apt-get update && apt-get install -y --no-install-recommends strace && "
        "rm -rf /var/lib/apt/lists/*; "
        "elif command -v apk >/dev/null 2>&1; then apk add --no-cache strace; "
        "elif command -v dnf >/dev/null 2>&1; then dnf install -y strace && dnf clean all; "
        "else echo 'WARNING: no known package manager for installing strace' >&2; fi\n"
    )
    copy_snippet = (
        f"COPY {TRACE_WRAPPER_NAME} /app/{TRACE_WRAPPER_NAME}\n"
        f"COPY {TRACE_VERIFIER_NAME} /app/{TRACE_VERIFIER_NAME}\n"
        f"RUN chmod +x /app/{TRACE_WRAPPER_NAME} /app/{TRACE_VERIFIER_NAME} && "
        "mkdir -p /app/workflow && "
        "printf 'ERROR: trace_wrapper_not_invoked\\n' > /app/workflow/gemini_file_trace.log\n"
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
            # Malformed but keep previous behavior for nonstandard inputs.
            dockerfile_text = install_snippet + dockerfile_text

    if f"COPY {TRACE_WRAPPER_NAME} /app/{TRACE_WRAPPER_NAME}" not in dockerfile_text:
        marker = "COPY material /app/material\n"
        if marker in dockerfile_text:
            dockerfile_text = dockerfile_text.replace(marker, copy_snippet + marker, 1)
        else:
            dockerfile_text += "\n" + copy_snippet
    return dockerfile_text

def materialize_self_debug_context_dataset(
    source_dataset: Path,
    tasks: list[tuple[int, str, Path]],
    *,
    include_same_material_reports: bool = True,
    candidate_materials_by_material: dict[str, list[str]] | None = None,
    candidate_self_debug_reviews_root: Path | None = None,
) -> tuple[Path, list[tuple[int, str, Path]]]:
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    target_dataset = (
        DEFAULT_AUGMENTED_DATASET_PARENT
        / f"{source_dataset.name}__self_debug_context__{timestamp}__pid{os.getpid()}"
    )
    target_dataset.mkdir(parents=True, exist_ok=False)

    def link_or_copy(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(src, dst)

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
                        "self_debug_reviews",
                        "candidate_self_debug_reviews",
                        "self_debug_context",
                    ),
                )
                continue
            link = target_task / child.name
            link.symlink_to(child.resolve(), target_is_directory=child.is_dir())

        reports = (
            self_debug_reports_for_material(material)
            if include_same_material_reports
            else []
        )
        candidate_reports: dict[str, list[tuple[str, Path, Path]]] = {}
        if candidate_materials_by_material and candidate_self_debug_reviews_root is not None:
            for candidate_material in candidate_materials_by_material.get(material, []):
                reports_for_candidate = self_debug_reports_for_material(
                    candidate_material,
                    candidate_self_debug_reviews_root,
                )
                if reports_for_candidate:
                    candidate_reports[candidate_material] = reports_for_candidate

        self_debug_records = self_debug_source_files(
            reports,
            candidate_reports,
            include_same_material_reports=include_same_material_reports,
        )
        write_self_debug_context_bundle(
            target_task,
            material=material,
            records=self_debug_records,
        )
        write_self_debug_context_bundle(
            target_task / "environment",
            material=material,
            records=self_debug_records,
        )
        write_trace_tools(target_task / "environment")

        context_root = target_task / "self_debug_reviews"
        environment_context_root = target_task / "environment" / "self_debug_reviews"
        for case_name, md_path, json_path in reports:
            case_target = context_root / safe_context_name(case_name)
            case_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md_path, case_target / "self_debug_report.md")
            shutil.copy2(json_path, case_target / "self_debug_report.json")
            environment_case_target = environment_context_root / safe_context_name(case_name)
            environment_case_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md_path, environment_case_target / "self_debug_report.md")
            shutil.copy2(json_path, environment_case_target / "self_debug_report.json")

        candidate_context_root = target_task / "candidate_self_debug_reviews"
        environment_candidate_context_root = (
            target_task / "environment" / "candidate_self_debug_reviews"
        )
        for candidate_material, reports_for_candidate in candidate_reports.items():
            safe_material = safe_context_name(candidate_material)
            for case_name, md_path, json_path in reports_for_candidate:
                safe_case = safe_context_name(case_name)
                case_target = candidate_context_root / safe_material / safe_case
                case_target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md_path, case_target / "self_debug_report.md")
                shutil.copy2(json_path, case_target / "self_debug_report.json")
                environment_case_target = (
                    environment_candidate_context_root / safe_material / safe_case
                )
                environment_case_target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md_path, environment_case_target / "self_debug_report.md")
                shutil.copy2(json_path, environment_case_target / "self_debug_report.json")

        source_dockerfile = source_task / "environment" / "Dockerfile"
        target_dockerfile = target_task / "environment" / "Dockerfile"
        dockerfile_text = source_dockerfile.read_text(encoding="utf-8")
        dockerfile_text = inject_trace_tools_into_dockerfile(dockerfile_text)
        dockerfile_text = dockerfile_text.replace(
            "COPY self_debug_reviews /app/self_debug_reviews\n",
            "",
        )
        dockerfile_text = dockerfile_text.replace(
            "COPY candidate_self_debug_reviews /app/candidate_self_debug_reviews\n",
            "",
        )
        if reports and "COPY self_debug_reviews /app/self_debug_reviews" not in dockerfile_text:
            marker = "COPY material /app/material\n"
            if marker in dockerfile_text:
                dockerfile_text = dockerfile_text.replace(
                    marker,
                    marker + "COPY self_debug_reviews /app/self_debug_reviews\n",
                    1,
                )
            else:
                dockerfile_text += "\nCOPY self_debug_reviews /app/self_debug_reviews\n"
        if (
            candidate_reports
            and "COPY candidate_self_debug_reviews /app/candidate_self_debug_reviews"
            not in dockerfile_text
        ):
            marker = "COPY self_debug_reviews /app/self_debug_reviews\n"
            if marker in dockerfile_text:
                dockerfile_text = dockerfile_text.replace(
                    marker,
                    marker + "COPY candidate_self_debug_reviews /app/candidate_self_debug_reviews\n",
                    1,
                )
            else:
                marker = "COPY material /app/material\n"
                if marker in dockerfile_text:
                    dockerfile_text = dockerfile_text.replace(
                        marker,
                        marker
                        + "COPY candidate_self_debug_reviews /app/candidate_self_debug_reviews\n",
                        1,
                    )
                else:
                    dockerfile_text += (
                        "\nCOPY candidate_self_debug_reviews "
                        "/app/candidate_self_debug_reviews\n"
                    )
        if (
            self_debug_records
            and "COPY self_debug_context /app/self_debug_context" not in dockerfile_text
        ):
            marker = "COPY material /app/material\n"
            if marker in dockerfile_text:
                dockerfile_text = dockerfile_text.replace(
                    marker,
                    marker + "COPY self_debug_context /app/self_debug_context\n",
                    1,
                )
            else:
                dockerfile_text += "\nCOPY self_debug_context /app/self_debug_context\n"
        target_dockerfile.write_text(dockerfile_text, encoding="utf-8")

        instruction_text = (source_task / "instruction.md").read_text(encoding="utf-8")
        instruction_text += context_instruction_appendix(
            material,
            reports,
            candidate_reports,
            include_same_material_reports=include_same_material_reports,
            expected_self_debug_file_count=len(self_debug_records),
        )
        (target_task / "instruction.md").write_text(instruction_text, encoding="utf-8")

        augmented_tasks.append((num_wann, material, target_task))

    lines = [
        "# Self-Debug Context Augmented Dataset",
        "",
        f"Source dataset: `{source_dataset}`",
        f"Self-debug reviews root: `{SELF_DEBUG_REVIEWS_ROOT}`",
        (
            "Candidate self-debug reviews root: "
            f"`{candidate_self_debug_reviews_root}`"
            if candidate_self_debug_reviews_root is not None
            else "Candidate self-debug reviews root: not enabled"
        ),
        f"Task count: {len(augmented_tasks)}",
        "",
        "Each task directory links to the original task inputs and includes a mandatory",
        (
            "`self_debug_context/ALL_SELF_DEBUG_REPORTS.md` bundle plus raw copied "
            "self-debug report files for the same material."
            if include_same_material_reports
            else "`self_debug_context/ALL_SELF_DEBUG_REPORTS.md` bundle with candidate report files only; same-material reports are not copied."
        ),
        "",
    ]
    (target_dataset / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return target_dataset, augmented_tasks


def build_command(args: argparse.Namespace, dataset: Path, *, n_concurrent: int | None = None) -> list[str]:
    concurrency = args.n_concurrent if n_concurrent is None else n_concurrent
    command = [
        "harbor",
        "run",
        "-p",
        relpath_for_command(dataset),
        "-a",
        args.agent,
        "-m",
        args.model,
        "--n-concurrent",
        str(concurrency),
    ]

    if args.jobs_root is not None and not has_option(args.extra_arg, {"--jobs-dir", "-o"}):
        command.extend(["--jobs-dir", relpath_for_command(args.jobs_root)])

    if args.delete_after_run and not any(arg in {"--delete", "--no-delete"} for arg in args.extra_arg):
        command.append("--delete")

    default_extra_args = gemini_default_extra_args(args)
    all_extra_args = [*default_extra_args, *args.extra_arg]

    if args.gemini_ipv4_first and is_gemini_agent(args.agent) and not has_agent_node_options(all_extra_args):
        command.extend(["--agent-env", GEMINI_IPV4_NODE_OPTIONS])
    if (
        is_gemini_agent(args.agent)
        and not args.no_gemini_run_timeout
        and not has_agent_env(all_extra_args, "HARBOR_GEMINI_RUN_TIMEOUT_SEC")
    ):
        command.extend(["--agent-env", GEMINI_RUN_TIMEOUT_ENV])
    if (
        is_gemini_agent(args.agent)
        and not args.no_gemini_file_trace
        and args.trace_agent_wrapper_env_name
        and not has_agent_env(all_extra_args, args.trace_agent_wrapper_env_name)
    ):
        command.extend([
            "--agent-env",
            f"{args.trace_agent_wrapper_env_name}={TRACE_WRAPPER_APP_PATH}",
        ])
    if (
        is_gemini_agent(args.agent)
        and not args.no_gemini_host_network
        and not has_extra_docker_compose(all_extra_args)
    ):
        command.extend(["--extra-docker-compose", relpath_for_command(HOST_NETWORK_COMPOSE)])

    if all_extra_args:
        command.extend(all_extra_args)

    artifacts = [] if args.no_default_artifacts else list(DEFAULT_ARTIFACTS)
    if args.save_generated_qe_save:
        artifacts.extend(QE_SAVE_EXPORT_ARTIFACTS)
    if not args.no_gemini_file_trace:
        for artifact in SELF_DEBUG_TRACE_ARTIFACTS:
            if artifact not in artifacts:
                artifacts.append(artifact)
    artifacts.extend(args.artifact)
    for artifact in artifacts:
        command.extend(["--artifact", artifact])

    return command


def print_ordered_commands(args: argparse.Namespace, tasks: list[tuple[int, str, Path]]) -> None:
    batch_size = args.batch_size
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    mode = f"run sorted batches of {batch_size} task(s) at a time"
    print(f"# Mode: {mode}")
    print(f"# Running {len(tasks)} tasks")
    if not tasks:
        print("# No matching tasks")
        print("true")
        return
    print(f"# Smallest num_wann: {tasks[0][1]} ({tasks[0][0]})")
    print(f"# Largest num_wann: {tasks[-1][1]} ({tasks[-1][0]})")
    print(f"# Dataset: {relpath_for_command(args.dataset)}")
    run_group = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_group = f"num_wann_ordered__{run_group}__pid{os.getpid()}"
    print("run_harbor_num_wann_ordered() {")
    print("  overall_status=0")
    if args.validate_new_success:
        print("  harbor_job_has_success() {")
        print("    python3 - \"$1\" <<'PY'")
        print("import json")
        print("import sys")
        print("from pathlib import Path")
        print("job_dir = Path(sys.argv[1])")
        print("for path in job_dir.rglob('diagnostics.json'):")
        print("    if path.parent.name != 'verifier' or 'randprojections' in path.parts:")
        print("        continue")
        print("    try:")
        print("        data = json.loads(path.read_text(encoding='utf-8'))")
        print("    except Exception:")
        print("        continue")
        print("    if isinstance(data, dict) and data.get('status') == 'success':")
        print("        raise SystemExit(0)")
        print("raise SystemExit(1)")
        print("PY")
        print("  }")
    prune_after_batch = args.docker_prune_after_batch or args.docker_prune_after_material
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        print("  pids=()")
        first_num_wann = batch[0][0]
        last_num_wann = batch[-1][0]
        print(
            "  "
            + shlex.join(
                [
                    "printf",
                    "Starting sorted batch %s-%s (num_wann %s-%s)\\n",
                    str(batch_start + 1),
                    str(batch_start + len(batch)),
                    str(first_num_wann),
                    str(last_num_wann),
                ]
            )
        )
        for offset, (num_wann, material, source) in enumerate(batch, start=batch_start + 1):
            command = build_command(args, source, n_concurrent=1)
            job_name = f"{run_group}__{offset:04d}__num_wann_{num_wann:03d}__{material}"
            print(
                "  "
                + shlex.join(
                    [
                        "printf",
                        "  [%s/%s] num_wann=%s %s\\n",
                        str(offset),
                        str(len(tasks)),
                        str(num_wann),
                        material,
                    ]
                )
            )
            if args.validate_new_success:
                print("  (")
                print("    task_status=0")
                print("    attempt=1")
                print("    while true; do")
                print("      attempt_label=$(printf '%03d' \"$attempt\")")
                print(f"      attempt_job_name={shlex.quote(job_name)}__try_${{attempt_label}}")
                print(f"      attempt_job_dir={shlex.quote(str(args.jobs_root))}/$attempt_job_name")
                print(f"      printf '    attempt %s for {material}\\n' \"$attempt_label\"")
                print(f"      {shlex.join(command)} --job-name \"$attempt_job_name\" || task_status=$?")
                if args.save_generated_qe_save:
                    installer_prefix = shlex.join([str(QE_SAVE_INSTALLER), "--job-dir"])
                    installer_suffix = shlex.join(["--task-dir", str(source)])
                    print(f"      {installer_prefix} \"$attempt_job_dir\" {installer_suffix} || task_status=$?")
                print("      if harbor_job_has_success \"$attempt_job_dir\"; then")
                print("        exit 0")
                print("      fi")
                print("      task_status=1")
                if args.delete_failed_attempt_folders:
                    print("      rm -rf -- \"$attempt_job_dir\" || task_status=$?")
                if args.max_attempts_per_needed_success > 0:
                    print(f"      if [ \"$attempt\" -ge {args.max_attempts_per_needed_success} ]; then")
                    print("        exit \"$task_status\"")
                    print("      fi")
                print("      attempt=$((attempt + 1))")
                print("    done")
                print("  ) &")
            elif args.save_generated_qe_save:
                command.extend(["--job-name", job_name])
                installer = [
                    str(QE_SAVE_INSTALLER),
                    "--job-dir",
                    str(args.jobs_root / job_name),
                    "--task-dir",
                    str(source),
                ]
                print("  (")
                print("    task_status=0")
                print(f"    {shlex.join(command)} || task_status=$?")
                print(f"    {shlex.join(installer)} || task_status=$?")
                print('    exit "$task_status"')
                print("  ) &")
            else:
                command.extend(["--job-name", job_name])
                print(f"  {shlex.join(command)} &")
            print("  pids+=(\"$!\")")
        print("  batch_status=0")
        print('  for pid in "${pids[@]}"; do')
        print('    wait "$pid" || batch_status=$?')
        print("  done")
        print('  if [ "$batch_status" -ne 0 ]; then')
        print('    overall_status="$batch_status"')
        if args.stop_on_error:
            print('    return "$overall_status"')
        print("  fi")
        if prune_after_batch:
            print(
                "  "
                + shlex.join(
                    [
                        "printf",
                        "Pruning unused Docker resources after sorted batch %s-%s\\n",
                        str(batch_start + 1),
                        str(batch_start + len(batch)),
                    ]
                )
            )
            print(f"  {shlex.join(DOCKER_SYSTEM_PRUNE_COMMAND)} || overall_status=$?")
    print('  return "$overall_status"')
    print("}")
    print("run_harbor_num_wann_ordered")


def print_target_success_loop(args: argparse.Namespace, tasks: list[tuple[int, str, Path]]) -> None:
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    run_group = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_group = f"num_wann_ordered__{run_group}__pid{os.getpid()}"
    command_records = [
        {
            "num_wann": num_wann,
            "material": material,
            "task_dir": str(source),
            "command": build_command(args, source, n_concurrent=1),
        }
        for num_wann, material, source in tasks
    ]
    success_roots = [str(path) for path in (args.success_roots or DEFAULT_SUCCESS_ROOTS)]
    excluded_dir_names = sorted(DEFAULT_EXCLUDED_RESULT_DIR_NAMES - set(args.include_result_dir_name))

    print(f"# Mode: top up to {args.target_success_runs} successful run(s) per material")
    print(f"# Wave size: {args.batch_size}")
    print(f"# Wave wall timeout: {args.success_wave_timeout_sec} seconds")
    print(f"# Candidate materials: {len(tasks)}")
    print(f"# Success roots: {', '.join(relpath_for_command(Path(path)) for path in success_roots)}")
    print(f"# New jobs root: {relpath_for_command(args.jobs_root)}")
    if args.delete_failed_attempt_folders:
        print("# Failed attempt folders: delete")
    else:
        print("# Failed attempt folders: keep")
    print("python3 - <<'PY'")
    print("from __future__ import annotations")
    print("from collections import Counter")
    print("import json")
    print("import os")
    print("import re")
    print("import shutil")
    print("import signal")
    print("import subprocess")
    print("import sys")
    print("import time")
    print("from pathlib import Path")
    print("")
    print(f"TASKS = {json.dumps(command_records, indent=2)}")
    print(f"SUCCESS_ROOTS = {[str(path) for path in success_roots]!r}")
    print(f"EXCLUDED_DIR_NAMES = {excluded_dir_names!r}")
    print(f"TARGET_SUCCESS_RUNS = {args.target_success_runs!r}")
    print(f"BATCH_SIZE = {args.batch_size!r}")
    print(f"WAVE_TIMEOUT_SEC = {args.success_wave_timeout_sec!r}")
    print(f"KILL_AFTER_SEC = {args.success_wave_kill_after_sec!r}")
    print(f"JOBS_ROOT = {str(args.jobs_root)!r}")
    print(f"RUN_GROUP = {run_group!r}")
    print(f"DELETE_FAILED_ATTEMPT_FOLDERS = {args.delete_failed_attempt_folders!r}")
    print(f"PRUNE_AFTER_WAVE = {(args.docker_prune_after_batch or args.docker_prune_after_material)!r}")
    print(f"DOCKER_PRUNE_COMMAND = {DOCKER_SYSTEM_PRUNE_COMMAND!r}")
    print("")
    print("VALID_MATERIALS = {task['material'] for task in TASKS}")
    print("TASK_BY_MATERIAL = {task['material']: task for task in TASKS}")
    print("")
    print("def material_from_record(record):")
    print("    if not isinstance(record, dict):")
    print("        return None")
    print("    material = record.get('material') or record.get('material_from_folder')")
    print("    if isinstance(material, str) and material:")
    print("        return material")
    print("    return None")
    print("")
    print("def material_from_diagnostics_path(path):")
    print("    for parent in path.parents:")
    print("        if parent.name in VALID_MATERIALS:")
    print("            return parent.name")
    print("        prefix = parent.name.split('__', 1)[0]")
    print("        if prefix in VALID_MATERIALS:")
    print("            return prefix")
    print("        match = re.search(r'__num_wann_\\d+__(?P<material>[^/]+)$', parent.name)")
    print("        if match and match.group('material') in VALID_MATERIALS:")
    print("            return match.group('material')")
    print("    return None")
    print("")
    print("def successful_run_counts():")
    print("    counts = Counter()")
    print("    for root_text in SUCCESS_ROOTS:")
    print("        root = Path(root_text)")
    print("        if not root.exists():")
    print("            continue")
    print("        for diagnostics_path in root.rglob('diagnostics.json'):")
    print("            if diagnostics_path.parent.name != 'verifier':")
    print("                continue")
    print("            if set(diagnostics_path.parts).intersection(EXCLUDED_DIR_NAMES):")
    print("                continue")
    print("            try:")
    print("                data = json.loads(diagnostics_path.read_text(encoding='utf-8'))")
    print("            except Exception:")
    print("                continue")
    print("            if not isinstance(data, dict) or data.get('status') != 'success':")
    print("                continue")
    print("            material = material_from_record(data) or material_from_diagnostics_path(diagnostics_path)")
    print("            if material in VALID_MATERIALS:")
    print("                counts[material] += 1")
    print("    return counts")
    print("")
    print("def _latest_existing(paths):")
    print("    paths = [path for path in paths if path.is_file()]")
    print("    if not paths:")
    print("        return None")
    print("    return max(paths, key=lambda path: path.stat().st_mtime)")
    print("")
    print("def self_debug_gate_passes(job_dir, task_dir):")
    print("    job_path = Path(job_dir)")
    print("    task_path = Path(task_dir)")
    print("    index_path = task_path / 'environment' / 'self_debug_context' / 'index.json'")
    print("    if not index_path.is_file():")
    print("        index_path = task_path / 'self_debug_context' / 'index.json'")
    print("    trace_path = _latest_existing(job_path.rglob('gemini_file_trace.log'))")
    print("    summary_path = _latest_existing(job_path.rglob('SELF_DEBUG_CONTEXT_SUMMARY.json'))")
    print("    if not index_path.is_file() or trace_path is None or summary_path is None:")
    print("        print(f'    self-debug gate failed: missing index/trace/summary for {job_path}', flush=True)")
    print("        return False")
    print("    verifier_path = task_path / 'environment' / 'verify_self_debug_context_access.py'")
    print("    if not verifier_path.is_file():")
    print("        print(f'    self-debug gate failed: missing verifier script {verifier_path}', flush=True)")
    print("        return False")
    print("    result = subprocess.run([sys.executable, str(verifier_path), '--index', str(index_path), '--summary', str(summary_path), '--trace', str(trace_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)")
    print("    if result.returncode != 0:")
    print("        print(result.stdout, flush=True)")
    print("        return False")
    print("    return True")
    print("")
    print("def job_success_status(job_dir, task_dir):")
    print("    job_path = Path(job_dir)")
    print("    diagnostics_success = False")
    print("    for diagnostics_path in job_path.rglob('diagnostics.json'):")
    print("        if diagnostics_path.parent.name != 'verifier':")
    print("            continue")
    print("        if set(diagnostics_path.parts).intersection(EXCLUDED_DIR_NAMES):")
    print("            continue")
    print("        try:")
    print("            data = json.loads(diagnostics_path.read_text(encoding='utf-8'))")
    print("        except Exception:")
    print("            continue")
    print("        if isinstance(data, dict) and data.get('status') == 'success':")
    print("            diagnostics_success = True")
    print("            break")
    print("    if not diagnostics_success:")
    print("        return False, 'diagnostics_not_success'")
    print("    if not self_debug_gate_passes(job_dir, task_dir):")
    print("        return False, 'self_debug_gate_failed'")
    print("    return True, 'success'")
    print("")
    print("def select_wave(counts, wave_index):")
    print("    pending = [")
    print("        task")
    print("        for task in sorted(TASKS, key=lambda item: (item['num_wann'], item['material']))")
    print("        if counts[task['material']] < TARGET_SUCCESS_RUNS")
    print("    ]")
    print("    if not pending:")
    print("        return []")
    print("    start = ((wave_index - 1) * BATCH_SIZE) % len(pending)")
    print("    return [pending[(start + offset) % len(pending)] for offset in range(min(BATCH_SIZE, len(pending)))]")
    print("")
    print("def terminate_process_group(process, label):")
    print("    if process.poll() is not None:")
    print("        return")
    print("    print(f'  timeout: terminating {label}', flush=True)")
    print("    try:")
    print("        os.killpg(process.pid, signal.SIGTERM)")
    print("    except ProcessLookupError:")
    print("        return")
    print("    deadline = time.monotonic() + KILL_AFTER_SEC")
    print("    while process.poll() is None and time.monotonic() < deadline:")
    print("        time.sleep(1)")
    print("    if process.poll() is None:")
    print("        print(f'  timeout: force killing {label}', flush=True)")
    print("        try:")
    print("            os.killpg(process.pid, signal.SIGKILL)")
    print("        except ProcessLookupError:")
    print("            pass")
    print("")
    print("def run_prune():")
    print("    if not PRUNE_AFTER_WAVE:")
    print("        return 0")
    print("    print('Pruning unused Docker resources after wave', flush=True)")
    print("    return subprocess.run(DOCKER_PRUNE_COMMAND).returncode")
    print("")
    print("def main():")
    print("    wave_index = 1")
    print("    total_started = 0")
    print("    while True:")
    print("        counts = successful_run_counts()")
    print("        pending_count = sum(1 for task in TASKS if counts[task['material']] < TARGET_SUCCESS_RUNS)")
    print("        if pending_count == 0:")
    print("            print(f'All {len(TASKS)} material(s) have at least {TARGET_SUCCESS_RUNS} successful run(s).', flush=True)")
    print("            return 0")
    print("        wave = select_wave(counts, wave_index)")
    print("        if not wave:")
    print("            print('No eligible pending materials remain.', flush=True)")
    print("            return 0")
    print("        print(f'Starting wave {wave_index}: {len(wave)} attempt(s), {pending_count} material(s) still below target', flush=True)")
    print("        processes = []")
    print("        for task in wave:")
    print("            total_started += 1")
    print("            material = task['material']")
    print("            num_wann = task['num_wann']")
    print("            job_name = f'{RUN_GROUP}__wave_{wave_index:04d}__slot_{len(processes) + 1:02d}__attempt_{total_started:05d}__num_wann_{num_wann:03d}__{material}'")
    print("            job_dir = str(Path(JOBS_ROOT) / job_name)")
    print("            command = [*task['command'], '--job-name', job_name]")
    print("            print(f'  [{len(processes) + 1}/{len(wave)}] num_wann={num_wann} {material}', flush=True)")
    print("            process = subprocess.Popen(command, preexec_fn=os.setsid)")
    print("            processes.append({'process': process, 'material': material, 'job_name': job_name, 'job_dir': job_dir, 'task_dir': task['task_dir']})")
    print("        deadline = time.monotonic() + WAVE_TIMEOUT_SEC")
    print("        while any(item['process'].poll() is None for item in processes) and time.monotonic() < deadline:")
    print("            time.sleep(5)")
    print("        for item in processes:")
    print("            terminate_process_group(item['process'], item['job_name'])")
    print("        wave_status = 0")
    print("        for item in processes:")
    print("            status = item['process'].wait()")
    print("            success, reason = job_success_status(item['job_dir'], item['task_dir'])")
    print("            if success:")
    print("                print(f\"  success: {item['material']} ({item['job_name']})\", flush=True)")
    print("                continue")
    print("            wave_status = 1")
    print("            print(f\"  non-success: {item['material']} ({item['job_name']}), exit={status}, reason={reason}\", flush=True)")
    print("            if DELETE_FAILED_ATTEMPT_FOLDERS:")
    print("                if reason == 'self_debug_gate_failed':")
    print("                    print(f\"  preserving job folder because self-debug gate failed: {item['job_dir']}\", flush=True)")
    print("                else:")
    print("                    shutil.rmtree(item['job_dir'], ignore_errors=True)")
    print("        prune_status = run_prune()")
    print("        if prune_status != 0:")
    print("            wave_status = prune_status")
    print("        wave_index += 1")
    print("")
    print("raise SystemExit(main())")
    print("PY")


def main() -> None:
    global SELF_DEBUG_REVIEWS_ROOT, DEFAULT_SUCCESS_ROOTS

    parser = argparse.ArgumentParser(
        description=(
            "Read every task's instruction.md, sort by num_wann ascending, "
            "attach Gemini self-debug report context, "
            "and print a Harbor run command for all tasks in that order."
        )
    )
    parser.add_argument("--dataset", "-p", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent", "-a", default=CACHED_GEMINI_AGENT_IMPORT)
    parser.add_argument("--model", "-m", default="google/gemini-3.1-pro-preview")
    parser.add_argument(
        "--n-concurrent",
        "-n",
        type=int,
        default=DEFAULT_N_CONCURRENT,
        help=(
            "Number of concurrent Harbor trials for the generated single-job command. "
            "Default: 8, so Harbor keeps eight trials active while reading tasks from "
            "the num_wann-ordered dataset."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one extra argument to the Harbor command. Repeat for multiple args.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help=(
            "Pass one additional Harbor --artifact path to every generated harbor run. "
            "The plotting-summary paths /app/report.json and /app/REPORT.md are "
            "already included by default. The attempt artifacts are preserved by "
            "the task wrapper without exporting /app/artifacts again."
        ),
    )
    parser.add_argument(
        "--save-generated-qe-save",
        action="store_true",
        help=(
            "Export a valid workflow/run_dir/out tree from each Harbor trial and "
            "install it into the source task's environment/material/qe_save after the run. "
            "Existing qe_save directories are never overwritten."
        ),
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        help=(
            "Harbor jobs directory for generated runs, passed to Harbor as --jobs-dir "
            "and used to locate exported QE-save artifacts. Defaults to jobs, or "
            "reruns when --target-success-runs is used."
        ),
    )
    parser.add_argument(
        "--no-default-artifacts",
        action="store_true",
        help="Do not include the default plotting artifact paths in generated Harbor commands.",
    )
    parser.add_argument(
        "--materials-only",
        action="store_true",
        help="Print just the material names in num_wann order.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Run this many sorted tasks at a time in the explicit batch launcher. Default: 4."
        ),
    )
    parser.add_argument(
        "--docker-prune-after-batch",
        action="store_true",
        help=(
            "After each sorted batch finishes, run 'docker system prune --force'. "
            "This is global Docker cleanup: running containers are preserved, but "
            "stopped containers, unused networks, dangling images, and build cache "
            "from other runs may be removed."
        ),
    )
    parser.add_argument(
        "--docker-prune-after-material",
        action="store_true",
        help=(
            "Run 'docker system prune --force' after each material by requiring "
            "--batch-size 1. This avoids pruning while another material from this "
            "launcher is still running."
        ),
    )
    parser.add_argument(
        "--diagnostics-summary",
        type=Path,
        help="Diagnostics summary JSON containing failed_or_unknown and results lists.",
    )
    parser.add_argument(
        "--self-debug-reviews-root",
        type=Path,
        default=SELF_DEBUG_REVIEWS_ROOT,
        help=(
            "Directory containing per-material self-debug report folders to copy into "
            f"the generated tasks. Defaults to {relpath_for_command(SELF_DEBUG_REVIEWS_ROOT)}."
        ),
    )
    parser.add_argument(
        "--include-candidate-self-debug-reports",
        action="store_true",
        help=(
            "Also copy self-debug reports for candidate_material entries from "
            "--candidate-run-error-table into each generated task."
        ),
    )
    parser.add_argument(
        "--candidate-self-debug-reports-only",
        action="store_true",
        help=(
            "Copy only candidate_material self-debug reports from "
            "--candidate-run-error-table, not previous reports for the same material. "
            "Implies --include-candidate-self-debug-reports. In --target-success-runs "
            "mode, the CSV material column defines eligible target materials."
        ),
    )
    parser.add_argument(
        "--candidate-run-error-table",
        type=Path,
        default=DEFAULT_CANDIDATE_RUN_ERROR_TABLE,
        help=(
            "CSV table containing material and candidate_material columns. Used only "
            "with --include-candidate-self-debug-reports."
        ),
    )
    parser.add_argument(
        "--candidate-self-debug-reviews-root",
        type=Path,
        default=DEFAULT_CANDIDATE_SELF_DEBUG_REVIEWS_ROOT,
        help=(
            "Directory containing per-candidate-material self-debug report folders. "
            "Used only with --include-candidate-self-debug-reports."
        ),
    )
    parser.add_argument(
        "--rerun-non-successful-and-unrun",
        action="store_true",
        help=(
            "Use --diagnostics-summary to run failed/unknown materials plus materials "
            "present in --dataset but absent from diagnostics results."
        ),
    )
    parser.add_argument(
        "--target-success-runs",
        type=int,
        help=(
            "Top each material that already has previous self-debug reports up to "
            "this many successful runs. Existing successes are counted only when "
            "verifier/diagnostics.json contains status=success. With "
            "--candidate-self-debug-reports-only, eligible materials come from the "
            "--candidate-run-error-table material column instead."
        ),
    )
    parser.add_argument(
        "--success-root",
        type=Path,
        action="append",
        dest="success_roots",
        help=(
            "Directory to scan for existing successful diagnostics. Repeat for multiple "
            f"roots. Defaults to {relpath_for_command(SELF_DEBUG_REVIEWS_ROOT)} in this "
            "self-debug context launcher."
        ),
    )
    parser.add_argument(
        "--include-result-dir-name",
        action="append",
        default=[],
        help=(
            "Directory name to include even if it is excluded by default from success "
            "counting. The default excluded names are randprojections and case_files."
        ),
    )
    parser.add_argument(
        "--validate-new-success",
        dest="validate_new_success",
        action="store_true",
        default=None,
        help=(
            "After each generated Harbor job, inspect that job's diagnostics.json and "
            "retry the task until status=success. Defaults to enabled when "
            "--target-success-runs is used."
        ),
    )
    parser.add_argument(
        "--no-validate-new-success",
        dest="validate_new_success",
        action="store_false",
        help="Do not wrap generated runs in a diagnostics status=success retry loop.",
    )
    parser.add_argument(
        "--max-attempts-per-needed-success",
        type=int,
        default=0,
        help=(
            "Maximum attempts for each missing successful run when validating new "
            "successes. Default 0 means keep retrying until success."
        ),
    )
    parser.add_argument(
        "--success-wave-timeout-sec",
        type=int,
        default=3500,
        help=(
            "For --target-success-runs, run each wave for at most this many wall-clock "
            "seconds before terminating unfinished Harbor attempts. Default: 3500."
        ),
    )
    parser.add_argument(
        "--success-wave-kill-after-sec",
        type=int,
        default=30,
        help=(
            "After a wave timeout, wait this many seconds after SIGTERM before SIGKILL. "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--delete-failed-attempt-folders",
        action="store_true",
        help=(
            "In --target-success-runs mode, remove failed/timed-out Harbor job "
            "folders after each wave."
        ),
    )
    parser.add_argument(
        "--require-qe-save",
        action="store_true",
        help="Only run tasks that contain environment/material/qe_save.",
    )
    parser.add_argument(
        "--delete-after-run",
        dest="delete_after_run",
        action="store_true",
        default=True,
        help="Pass Harbor --delete so each environment is removed after completion. Default: enabled.",
    )
    parser.add_argument(
        "--no-delete-after-run",
        dest="delete_after_run",
        action="store_false",
        help="Do not add Harbor --delete to generated commands.",
    )
    parser.add_argument(
        "--gemini-ipv4-first",
        dest="gemini_ipv4_first",
        action="store_true",
        default=True,
        help=(
            "For gemini-cli, pass NODE_OPTIONS=--dns-result-order=ipv4first "
            "to avoid broken Docker IPv6 routes. Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-gemini-cached-defaults",
        action="store_true",
        help=(
            "Do not auto-inject the cached Gemini agent, shorter Gemini agent "
            "timeout, retry count, or transient Gemini setup/exit retry includes."
        ),
    )
    parser.add_argument(
        "--no-gemini-ipv4-first",
        dest="gemini_ipv4_first",
        action="store_false",
        help="Do not add NODE_OPTIONS=--dns-result-order=ipv4first for gemini-cli.",
    )
    parser.add_argument(
        "--no-gemini-run-timeout",
        action="store_true",
        help=(
            "Do not add HARBOR_GEMINI_RUN_TIMEOUT_SEC=4000 for the cached Gemini "
            "agent wrapper."
        ),
    )
    parser.add_argument(
        "--no-gemini-file-trace",
        action="store_true",
        help=(
            "Do not inject the strace wrapper, trace artifacts, or post-run "
            "self-debug context access gate for Gemini runs."
        ),
    )
    parser.add_argument(
        "--trace-agent-wrapper-env-name",
        default=DEFAULT_TRACE_AGENT_WRAPPER_ENV,
        help=(
            "Agent-layer environment variable used to point Gemini execution at "
            f"{TRACE_WRAPPER_APP_PATH}. Harbor's Gemini agent wrapper must honor "
            "this variable for OS-level tracing to be collected. Default: "
            f"{DEFAULT_TRACE_AGENT_WRAPPER_ENV}."
        ),
    )
    parser.add_argument(
        "--no-gemini-host-network",
        action="store_true",
        help=(
            "Do not add the Docker Compose host-network overlay for gemini-cli. "
            "By default this avoids broken Docker bridge outbound HTTPS on this host."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first sorted batch that has a failed Harbor command.",
    )
    args = parser.parse_args()
    SELF_DEBUG_REVIEWS_ROOT = args.self_debug_reviews_root.expanduser().resolve()
    DEFAULT_SUCCESS_ROOTS = [SELF_DEBUG_REVIEWS_ROOT]
    if not args.model:
        raise SystemExit("--model/-m cannot be empty")
    if args.n_concurrent < 1:
        raise SystemExit("--n-concurrent/-n must be at least 1")
    if args.target_success_runs is not None and args.target_success_runs < 1:
        raise SystemExit("--target-success-runs must be at least 1")
    if args.max_attempts_per_needed_success < 0:
        raise SystemExit("--max-attempts-per-needed-success cannot be negative")
    if args.success_wave_timeout_sec < 1:
        raise SystemExit("--success-wave-timeout-sec must be at least 1")
    if args.success_wave_kill_after_sec < 0:
        raise SystemExit("--success-wave-kill-after-sec cannot be negative")
    if args.docker_prune_after_material and args.batch_size != 1:
        raise SystemExit("--docker-prune-after-material requires --batch-size 1")
    if args.target_success_runs is not None and args.save_generated_qe_save:
        raise SystemExit("--target-success-runs does not support --save-generated-qe-save")

    if args.jobs_root is None:
        args.jobs_root = SELF_DEBUG_REVIEWS_ROOT
    if args.validate_new_success is None:
        args.validate_new_success = args.target_success_runs is not None
    if args.candidate_self_debug_reports_only:
        args.include_candidate_self_debug_reports = True
    args.candidate_run_error_table = args.candidate_run_error_table.expanduser().resolve()
    args.candidate_self_debug_reviews_root = (
        args.candidate_self_debug_reviews_root.expanduser().resolve()
    )

    candidate_materials_by_material = None
    if args.include_candidate_self_debug_reports:
        candidate_materials_by_material = candidate_materials_from_table(
            args.candidate_run_error_table
        )

    include_materials = None
    exclude_materials = None
    success_counts: Counter[str] = Counter()
    if args.rerun_non_successful_and_unrun:
        if args.diagnostics_summary is None:
            raise SystemExit("--rerun-non-successful-and-unrun requires --diagnostics-summary")
        failed_or_unknown_materials = failed_or_unknown_materials_from_diagnostics(args.diagnostics_summary)
        run_materials = material_names_from_results(args.diagnostics_summary)
        dataset_materials = {path.name for path in args.dataset.iterdir() if path.is_dir()}
        include_materials = failed_or_unknown_materials | (dataset_materials - run_materials)

    if args.target_success_runs is not None:
        dataset_materials = {path.name for path in args.dataset.iterdir() if path.is_dir()}
        context_materials = (
            set(candidate_materials_by_material or {})
            if args.candidate_self_debug_reports_only
            else materials_with_self_debug_reports()
        )
        candidate_materials = dataset_materials & context_materials
        if not candidate_materials:
            context_description = (
                f"candidate rows in {args.candidate_run_error_table}"
                if args.candidate_self_debug_reports_only
                else f"previous self-debug report pairs under {SELF_DEBUG_REVIEWS_ROOT}"
            )
            raise SystemExit(
                f"No dataset materials have {context_description}"
            )
        excluded_dir_names = DEFAULT_EXCLUDED_RESULT_DIR_NAMES - set(args.include_result_dir_name)
        success_counts = successful_run_counts(
            args.success_roots or DEFAULT_SUCCESS_ROOTS,
            valid_materials=candidate_materials,
            excluded_dir_names=excluded_dir_names,
        )
        below_target_materials = {
            material
            for material in candidate_materials
            if success_counts[material] < args.target_success_runs
        }
        include_materials = (
            below_target_materials
            if include_materials is None
            else include_materials & below_target_materials
        )

    tasks = dataset_tasks(
        args.dataset,
        include_materials=include_materials,
        exclude_materials=exclude_materials,
        require_qe_save=args.require_qe_save,
    )
    if args.target_success_runs is not None and args.materials_only:
        print(" ".join(material for _num_wann, material, _source in tasks))
        return

    if tasks and not args.materials_only:
        augmented_dataset, tasks = materialize_self_debug_context_dataset(
            args.dataset,
            tasks,
            include_same_material_reports=not args.candidate_self_debug_reports_only,
            candidate_materials_by_material=candidate_materials_by_material,
            candidate_self_debug_reviews_root=(
                args.candidate_self_debug_reviews_root
                if args.include_candidate_self_debug_reports
                else None
            ),
        )
        args.dataset = augmented_dataset

    if args.target_success_runs is not None:
        if not tasks:
            print("# No matching tasks")
            print("true")
            return
        print_target_success_loop(args, tasks)
        return

    if args.materials_only:
        print(" ".join(material for _num_wann, material, _source in tasks))
        return

    if not tasks:
        print("# No matching tasks")
        print("true")
        return

    print_ordered_commands(args, tasks)
    return

if __name__ == "__main__":
    main()
