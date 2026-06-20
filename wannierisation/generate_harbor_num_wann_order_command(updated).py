#!/usr/bin/env python3
"""Print Harbor commands that run tasks in increasing instruction num_wann."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
NUM_WANN_RE = re.compile(r"\bnum_wann\s*=\s*(\d+)\b")
DEFAULT_ARTIFACTS = ["/app/artifacts", "/app/report.json", "/app/REPORT.md"]
DEFAULT_N_CONCURRENT = 4
QE_SAVE_RELATIVE_PATH = Path("environment") / "material" / "qe_save"
HOST_NETWORK_COMPOSE = ROOT / "docker" / "harbor-host-network.compose.yml"
GEMINI_IPV4_NODE_OPTIONS = "NODE_OPTIONS=--dns-result-order=ipv4first"
GEMINI_RUN_TIMEOUT_ENV = "HARBOR_GEMINI_RUN_TIMEOUT_SEC=5400"
CACHED_GEMINI_AGENT_IMPORT = "harbor_agents.cached_gemini_cli:CachedGeminiCli"
DEFAULT_GEMINI_AGENT_TIMEOUT_MULTIPLIER = "1.1"
DEFAULT_MAX_RETRIES = "2"
DEFAULT_RETRY_INCLUDES = ["AgentSetupTimeoutError", "NonZeroAgentExitCodeError"]


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


def read_json(path: Path) -> object:
    if not path.is_file():
        raise SystemExit(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    if args.agent != "gemini-cli" or args.no_gemini_cached_defaults:
        return []

    extra: list[str] = []
    if not has_option(args.extra_arg, {"--agent-import-path"}):
        extra.extend(["--agent-import-path", CACHED_GEMINI_AGENT_IMPORT])
    if not has_option(args.extra_arg, {"--agent-timeout-multiplier"}):
        extra.extend(["--agent-timeout-multiplier", DEFAULT_GEMINI_AGENT_TIMEOUT_MULTIPLIER])
    if not has_option(args.extra_arg, {"--max-retries", "-r"}):
        extra.extend(["--max-retries", DEFAULT_MAX_RETRIES])

    existing_retries = retry_includes(args.extra_arg)
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
    if (
        args.agent == "gemini-cli"
        and not args.no_gemini_host_network
        and not has_extra_docker_compose(all_extra_args)
    ):
        command.extend(["--extra-docker-compose", relpath_for_command(HOST_NETWORK_COMPOSE)])

    if all_extra_args:
        command.extend(all_extra_args)

    artifacts = [] if args.no_default_artifacts else list(DEFAULT_ARTIFACTS)
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
            command.extend(["--job-name", job_name])
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
    print('  return "$overall_status"')
    print("}")
    print("run_harbor_num_wann_ordered")


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
            "The plotting-critical paths /app/artifacts, /app/report.json, and "
            "/app/REPORT.md are already included by default."
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
        "--background-batches",
        action="store_true",
        help=(
            "Deprecated alias for the default explicit sorted batch launcher."
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
            "Do not add HARBOR_GEMINI_RUN_TIMEOUT_SEC=600 for the cached Gemini "
            "agent wrapper."
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

    include_materials = None
    exclude_materials = None
    if args.rerun_non_successful_and_unrun:
        if args.diagnostics_summary is None:
            raise SystemExit("--rerun-non-successful-and-unrun requires --diagnostics-summary")
        failed_or_unknown_materials = failed_or_unknown_materials_from_diagnostics(args.diagnostics_summary)
        run_materials = material_names_from_results(args.diagnostics_summary)
        dataset_materials = {path.name for path in args.dataset.iterdir() if path.is_dir()}
        include_materials = failed_or_unknown_materials | (dataset_materials - run_materials)

    tasks = dataset_tasks(
        args.dataset,
        include_materials=include_materials,
        exclude_materials=exclude_materials,
        require_qe_save=args.require_qe_save,
    )
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
