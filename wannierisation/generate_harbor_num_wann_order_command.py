#!/usr/bin/env python3
"""Print Harbor commands that run tasks in increasing instruction num_wann."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
NUM_WANN_RE = re.compile(r"\bnum_wann\s*=\s*(\d+)\b")
# The task test wrapper already copies /app/artifacts/. into /logs/artifacts,
# which Harbor stores as the trial's top-level artifacts directory. Exporting
# /app/artifacts again creates a duplicate artifacts/artifacts tree.
DEFAULT_ARTIFACTS = ["/app/report.json", "/app/REPORT.md"]
QE_SAVE_EXPORT_ARTIFACTS = [
    "/app/workflow/run_dir/out",
    "/app/workflow/run_dir/scf.out",
    "/app/workflow/run_dir/nscf.out",
]
DEFAULT_N_CONCURRENT = 4
QE_SAVE_RELATIVE_PATH = Path("environment") / "material" / "qe_save"
QE_SAVE_INSTALLER = ROOT / "scripts" / "install_harbor_qe_save.py"
HOST_NETWORK_COMPOSE = ROOT / "docker" / "harbor-host-network.compose.yml"
GEMINI_IPV4_NODE_OPTIONS = "NODE_OPTIONS=--dns-result-order=ipv4first"
GEMINI_RUN_TIMEOUT_ENV = "HARBOR_GEMINI_RUN_TIMEOUT_SEC=4000"
CACHED_GEMINI_AGENT_IMPORT = "harbor_agents.cached_gemini_cli:CachedGeminiCli"
DEFAULT_GEMINI_AGENT_TIMEOUT_MULTIPLIER = "1.1"
DEFAULT_MAX_RETRIES = "2"
DEFAULT_RETRY_INCLUDES = ["AgentSetupTimeoutError", "NonZeroAgentExitCodeError"]
DOCKER_SYSTEM_PRUNE_COMMAND = ["docker", "system", "prune", "--force", '--all']
DEFAULT_SUCCESS_ROOTS = [ROOT / "jobs", ROOT / "reruns"]
DEFAULT_EXCLUDED_RESULT_DIR_NAMES = {"randprojections"}


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


def gemini_default_extra_args(args: argparse.Namespace) -> list[str]:
    if args.agent != "gemini-cli":
        return []

    extra: list[str] = []
    if not has_option(args.extra_arg, {"--agent-import-path"}):
        extra.extend(["--agent-import-path", CACHED_GEMINI_AGENT_IMPORT])
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


def ordered_dataset_dir(dataset: Path) -> Path:
    return dataset.parent / f"{dataset.name}__num_wann_ordered"


def materialize_ordered_dataset(source_dataset: Path, target_dataset: Path, tasks: list[tuple[int, str, Path]]) -> None:
    target_dataset.mkdir(parents=True, exist_ok=True)
    wanted_names = {
        f"{index:04d}__num_wann_{num_wann:03d}__{material}"
        for index, (num_wann, material, _task_dir) in enumerate(tasks, start=1)
    }

    for path in list(target_dataset.iterdir()):
        if path.name == "README.md":
            continue
        if path.name not in wanted_names and path.is_symlink():
            path.unlink()
        elif path.name not in wanted_names:
            raise SystemExit(
                f"Refusing to remove non-symlink path in generated dataset: {path}. "
                "Choose a different --ordered-dataset-dir."
            )

    for index, (num_wann, material, source) in enumerate(tasks, start=1):
        target = target_dataset / f"{index:04d}__num_wann_{num_wann:03d}__{material}"
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            raise SystemExit(
                f"Refusing to overwrite existing generated dataset entry: {target}. "
                "Choose a different --ordered-dataset-dir."
            )
        target.symlink_to(source.resolve(), target_is_directory=True)

    lines = [
        "# Num-Wann Ordered Wannier 200 Harbor Dataset",
        "",
        f"Source dataset: `{source_dataset}`",
        f"Task count: {len(tasks)}",
        "",
        "Order:",
    ]
    lines.extend(
        f"{index}. `{material}`: `num_wann = {num_wann}`"
        for index, (num_wann, material, _source) in enumerate(tasks, start=1)
    )
    lines.extend(["", "Generated by `scripts/generate_harbor_num_wann_order_command.py`.", ""])
    (target_dataset / "README.md").write_text("\n".join(lines), encoding="utf-8")


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

    if args.gemini_ipv4_first and args.agent == "gemini-cli" and not has_agent_node_options(all_extra_args):
        command.extend(["--agent-env", GEMINI_IPV4_NODE_OPTIONS])
    if (
        args.agent == "gemini-cli"
        and not args.no_gemini_run_timeout
        and not has_agent_env(all_extra_args, "HARBOR_GEMINI_RUN_TIMEOUT_SEC")
    ):
        command.extend(["--agent-env", GEMINI_RUN_TIMEOUT_ENV])
    if args.agent == "gemini-cli" and not has_extra_docker_compose(all_extra_args):
        command.extend(["--extra-docker-compose", relpath_for_command(HOST_NETWORK_COMPOSE)])

    if all_extra_args:
        command.extend(all_extra_args)

    artifacts = [] if args.no_default_artifacts else list(DEFAULT_ARTIFACTS)
    if args.save_generated_qe_save:
        artifacts.extend(QE_SAVE_EXPORT_ARTIFACTS)
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
    print("def job_has_success(job_dir):")
    print("    job_path = Path(job_dir)")
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
    print("            return True")
    print("    return False")
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
    print("            processes.append({'process': process, 'material': material, 'job_name': job_name, 'job_dir': job_dir})")
    print("        deadline = time.monotonic() + WAVE_TIMEOUT_SEC")
    print("        while any(item['process'].poll() is None for item in processes) and time.monotonic() < deadline:")
    print("            time.sleep(5)")
    print("        for item in processes:")
    print("            terminate_process_group(item['process'], item['job_name'])")
    print("        wave_status = 0")
    print("        for item in processes:")
    print("            status = item['process'].wait()")
    print("            success = job_has_success(item['job_dir'])")
    print("            if success:")
    print("                print(f\"  success: {item['material']} ({item['job_name']})\", flush=True)")
    print("                continue")
    print("            wave_status = 1")
    print("            print(f\"  non-success: {item['material']} ({item['job_name']}), exit={status}\", flush=True)")
    print("            if DELETE_FAILED_ATTEMPT_FOLDERS:")
    print("                shutil.rmtree(item['job_dir'], ignore_errors=True)")
    print("        prune_status = run_prune()")
    print("        if prune_status != 0:")
    print("            wave_status = prune_status")
    print("        wave_index += 1")
    print("")
    print("raise SystemExit(main())")
    print("PY")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read every task's instruction.md, sort by num_wann ascending, "
            "and print a Harbor run command for all tasks in that order."
        )
    )
    parser.add_argument("--dataset", "-p", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent", "-a", default="gemini-cli")
    parser.add_argument("--model", "-m", default="google/gemini-3.1-pro-preview")
    parser.add_argument(
        "--n-concurrent",
        "-n",
        type=int,
        default=DEFAULT_N_CONCURRENT,
        help=(
            "Number of concurrent Harbor trials for the generated single-job command. "
            "Default: 4, so Harbor keeps four trials active while reading tasks from "
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
        "--no-ordered-dataset",
        action="store_true",
        help="Print a command with --include-task-name filters instead of creating an ordered dataset.",
    )
    parser.add_argument(
        "--single-job",
        action="store_true",
        help=(
            "Run one Harbor job over an ordered symlink dataset. This is not the default "
            "because Harbor may enumerate local dataset directories in filesystem order."
        ),
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
            "Top each material up to this many successful runs. Existing successes "
            "are counted only when verifier/diagnostics.json contains status=success."
        ),
    )
    parser.add_argument(
        "--success-root",
        type=Path,
        action="append",
        dest="success_roots",
        help=(
            "Directory to scan for existing successful diagnostics. Repeat for multiple "
            "roots. Defaults to jobs and reruns when --target-success-runs is used."
        ),
    )
    parser.add_argument(
        "--include-result-dir-name",
        action="append",
        default=[],
        help=(
            "Directory name to include even if it is excluded by default from success "
            "counting. The default excluded name is randprojections."
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
        "--no-gemini-run-timeout",
        action="store_true",
        help=(
            "Do not add HARBOR_GEMINI_RUN_TIMEOUT_SEC=600 for the cached Gemini "
            "agent wrapper."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first sorted batch that has a failed Harbor command.",
    )
    parser.add_argument(
        "--ordered-dataset-dir",
        type=Path,
        help="Directory to write the ordered symlink dataset. Defaults to a generated name next to --dataset.",
    )
    args = parser.parse_args()
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
    if args.save_generated_qe_save and args.single_job:
        raise SystemExit("--save-generated-qe-save requires the default explicit batch launcher")
    if args.target_success_runs is not None and args.single_job:
        raise SystemExit("--target-success-runs requires the default explicit batch launcher")
    if args.target_success_runs is not None and args.save_generated_qe_save:
        raise SystemExit("--target-success-runs does not support --save-generated-qe-save")

    if args.jobs_root is None:
        args.jobs_root = ROOT / "reruns" if args.target_success_runs is not None else ROOT / "jobs"
    if args.validate_new_success is None:
        args.validate_new_success = args.target_success_runs is not None

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
        excluded_dir_names = DEFAULT_EXCLUDED_RESULT_DIR_NAMES - set(args.include_result_dir_name)
        success_counts = successful_run_counts(
            args.success_roots or DEFAULT_SUCCESS_ROOTS,
            valid_materials=dataset_materials,
            excluded_dir_names=excluded_dir_names,
        )
        below_target_materials = {
            material
            for material in dataset_materials
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

    if not args.single_job:
        print_ordered_commands(args, tasks)
        return

    if args.no_ordered_dataset:
        command = build_command(args, args.dataset)
        for _num_wann, material, _source in tasks:
            command.extend(["--include-task-name", f"wannier_200/{material}"])
        run_dataset = args.dataset
    else:
        run_dataset = args.ordered_dataset_dir or ordered_dataset_dir(args.dataset)
        materialize_ordered_dataset(args.dataset, run_dataset, tasks)

    if not args.no_ordered_dataset:
        command = build_command(args, run_dataset)

    print(f"# Mode: include all tasks sorted by instruction num_wann ascending")
    print(f"# Running {len(tasks)} tasks")
    print(f"# Smallest num_wann: {tasks[0][1]} ({tasks[0][0]})")
    print(f"# Largest num_wann: {tasks[-1][1]} ({tasks[-1][0]})")
    print(f"# Dataset: {relpath_for_command(run_dataset)}")
    print(shlex.join(command))


if __name__ == "__main__":
    main()
