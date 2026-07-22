#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/jiasheng/WannierisationBenchmarking")

CANDIDATES_CSV = ROOT / "include_only_candidates.csv"
NO_KNOWLEDGE_RUNS = ROOT / "jobsDeepseekProTerminus2Controlled"
WITH_KNOWLEDGE_RUNS = ROOT / "jobsGeminiReviewsDeepseek" / "ChemSimReruns"
OUTPUT_CSV = ROOT / "deepseekcontrolledstats.csv"

# Deepseek V4 Pro prices, in dollars per 1 million tokens.
INPUT_TOKEN_COST_PER_1M = 0.44
CACHE_TOKEN_COST_PER_1M = 0.003625
OUTPUT_TOKEN_COST_PER_1M = 0.87

RUN_MATERIAL_RE = re.compile(r"__num_wann_\d+__(.+)$")


def read_candidates() -> dict[str, list[str]]:
    candidates_by_target: dict[str, list[str]] = defaultdict(list)

    with CANDIDATES_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"target_material", "candidate_material"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{CANDIDATES_CSV} is missing columns: {sorted(missing)}")

        for row in reader:
            target = (row.get("target_material") or "").strip()
            candidate = (row.get("candidate_material") or "").strip()
            if not target or not candidate:
                continue
            if candidate not in candidates_by_target[target]:
                candidates_by_target[target].append(candidate)

    return dict(candidates_by_target)


def material_from_result(data: dict[str, object], result_path: Path) -> str:
    task_name = data.get("task_name")
    if isinstance(task_name, str) and "/" in task_name:
        return task_name.rsplit("/", 1)[-1]

    task_id = data.get("task_id")
    if isinstance(task_id, dict):
        task_path = task_id.get("path")
        if isinstance(task_path, str) and "/" in task_path:
            return task_path.rsplit("/", 1)[-1]

    trial_name = data.get("trial_name")
    if isinstance(trial_name, str) and "__" in trial_name:
        return trial_name.split("__", 1)[0]

    match = RUN_MATERIAL_RE.search(result_path.parent.name)
    if match:
        return match.group(1)

    return result_path.parent.name.rsplit("__", 1)[-1]


def int_token(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def token_tuple(stats: dict[str, object]) -> tuple[int, int, int]:
    return (
        int_token(stats.get("n_input_tokens")),
        int_token(stats.get("n_cache_tokens")),
        int_token(stats.get("n_output_tokens")),
    )


def calculated_cost_usd(tokens: tuple[int, int, int]) -> float:
    input_tokens, cache_tokens, output_tokens = tokens
    uncached_input_tokens = max(input_tokens - cache_tokens, 0)
    return (
        (uncached_input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_1M
        + (cache_tokens / 1_000_000) * CACHE_TOKEN_COST_PER_1M
        + (output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_1M
    )


def cost_usd(stats: dict[str, object], tokens: tuple[int, int, int]) -> float:
    cost = stats.get("cost_usd")
    if cost is None:
        return calculated_cost_usd(tokens)
    return float(cost)


def load_run(result_path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable result: {result_path} ({exc})")
        return None

    stats = data.get("stats")
    if not isinstance(stats, dict):
        return None

    tokens = token_tuple(stats)
    return {
        "material": material_from_result(data, result_path),
        "result_path": result_path,
        "tokens": tokens,
        "cost_usd": cost_usd(stats, tokens),
    }


def collect_runs(run_root: Path) -> dict[str, list[dict[str, object]]]:
    runs_by_material: dict[str, list[dict[str, object]]] = defaultdict(list)

    # Only immediate run-folder result files are used. Nested trial result files
    # duplicate the same aggregate in this Harbor layout.
    for result_path in sorted(run_root.glob("*/result.json")):
        run = load_run(result_path)
        if run is None:
            continue
        runs_by_material[str(run["material"])].append(run)

    return dict(runs_by_material)


def format_avg_tokens(runs: list[dict[str, object]]) -> str:
    if not runs:
        return ""

    token_tuples = [run["tokens"] for run in runs]
    num_runs = len(token_tuples)

    avg_input = sum(tokens[0] for tokens in token_tuples) / num_runs
    avg_cache = sum(tokens[1] for tokens in token_tuples) / num_runs
    avg_output = sum(tokens[2] for tokens in token_tuples) / num_runs

    return f"({avg_input:.2f}, {avg_cache:.2f}, {avg_output:.2f})"


def format_avg_cost(runs: list[dict[str, object]]) -> str:
    if not runs:
        return ""

    avg_cost = sum(float(run["cost_usd"]) for run in runs) / len(runs)
    return f"{avg_cost:.9f}"


def write_stats_csv() -> None:
    candidates_by_target = read_candidates()
    no_knowledge_runs = collect_runs(NO_KNOWLEDGE_RUNS)
    with_knowledge_runs = collect_runs(WITH_KNOWLEDGE_RUNS)

    headers = [
        "Target Material",
        "# Similar Materials",
        "Similar Materials Names",
        "Avg Token Usage (No Knowledge)",
        "Avg Price (No Knowledge)",
        "Avg Token Usage (With Knowledge)",
        "Avg Price (With Knowledge)",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for target in sorted(candidates_by_target):
            candidates = candidates_by_target[target]
            no_knowledge = no_knowledge_runs.get(target, [])
            with_knowledge = with_knowledge_runs.get(target, [])

            writer.writerow(
                {
                    "Target Material": target,
                    "# Similar Materials": len(candidates),
                    "Similar Materials Names": ", ".join(candidates),
                    "Avg Token Usage (No Knowledge)": format_avg_tokens(no_knowledge),
                    "Avg Price (No Knowledge)": format_avg_cost(no_knowledge),
                    "Avg Token Usage (With Knowledge)": format_avg_tokens(with_knowledge),
                    "Avg Price (With Knowledge)": format_avg_cost(with_knowledge),
                }
            )


if __name__ == "__main__":
    write_stats_csv()
    print(f"Wrote {OUTPUT_CSV}")
