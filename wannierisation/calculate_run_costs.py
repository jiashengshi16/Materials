#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# Edit these defaults, or pass positional arguments:
#   python scripts/calculate_run_costs.py /path/to/jobs "Deepseek V4 Flash" 07-12 07-13
INPUT_FOLDER = "/home/jiasheng/WannierisationBenchmarking/jobsDeepseekProTerminus2"
MODEL_NAME = "Deepseek V4 Pro"
DAYS_TO_INCLUDE = ["07-13"]

COMPARE_INPUT_FOLDER = "/home/jiasheng/WannierisationBenchmarking/jobsGeminiFlash35"
COMPARE_MODEL_NAME = "Gemini 3.5 Flash"
# Empty means search all days in the comparison folder for the selected materials.
COMPARE_DAYS_TO_INCLUDE: list[str] = []

# Dollars per 1M tokens. Use a tuple when the provider gave a price range.
MODEL_PRICES = {
    "Gemini 3.1 Pro": {"input": 2.00, "output": 12.00},
    "Gemini 3.5 Flash": {"input": 1.50, "cache_input": 0.15, "output": 9.00},
    "Gemini 2.5 Flash": {"input": 0.30, "output": 2.50},
    "GPT 5.5": {"input": 5.00, "output": 30.00},
    "GPT 5.4": {"input": 2.50, "output": 15.00},
    "GPT 5.4 mini": {"input": 0.75, "output": 4.50},
    "Qwen3 Coder Plus": {"input": 0.65, "output": (3.25, 5.00)},
    "Deepseek V4 Pro": {"input": 0.44, "cache_input": 0.003625, "output": 0.87},
    "Deepseek V4 Flash": {"input": 0.14, "cache_input": 0.0028, "output": 0.28},
    "GLM 5.2": {"input": 0.98, "output": 3.08},
    "GLM 4.5 Air": {"input": None, "output": 1.10},
}

DATE_RE = re.compile(r"__(\d{4})-(\d{2})-(\d{2})__")


@dataclass(frozen=True)
class RunUsage:
    path: Path
    trial_name: str
    material: str
    day: str
    finished: bool
    has_diagnostics: bool
    input_tokens: int
    cache_tokens: int
    output_tokens: int
    observed_cost_usd: float | None


def normalize_day(day: str) -> str:
    day = day.strip()
    if not day:
        return day
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return day[5:]
    if re.fullmatch(r"\d{2}-\d{2}", day):
        return day
    raise SystemExit(f"Bad day {day!r}. Use MM-DD, like 07-13, or YYYY-MM-DD.")


def day_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = DATE_RE.search(part)
        if match:
            return f"{match.group(2)}-{match.group(3)}"
    return None


def as_int(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


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

    match = re.search(r"__num_wann_\d+__(.+)$", result_path.parent.name)
    if match:
        return match.group(1)

    return result_path.parent.name.split("__", 1)[0]


def load_run_usage(result_path: Path) -> RunUsage | None:
    day = day_from_path(result_path)
    if day is None:
        return None

    try:
        data = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Skipping unreadable JSON: {result_path} ({exc})")
        return None

    agent_result = data.get("agent_result")
    if isinstance(agent_result, dict):
        diagnostics_path = result_path.parent / "verifier" / "diagnostics.json"
        return RunUsage(
            path=result_path,
            trial_name=str(data.get("trial_name") or result_path.parent.name),
            material=material_from_result(data, result_path),
            day=day,
            finished=bool(data.get("finished_at")),
            has_diagnostics=diagnostics_path.is_file(),
            input_tokens=as_int(agent_result.get("n_input_tokens")),
            cache_tokens=as_int(agent_result.get("n_cache_tokens")),
            output_tokens=as_int(agent_result.get("n_output_tokens")),
            observed_cost_usd=as_optional_float(agent_result.get("cost_usd")),
        )

    # Fallback for aggregate-only result files. These are ignored when any
    # trial-level result exists under the same input folder, to avoid double
    # counting the usual Harbor layout.
    stats = data.get("stats")
    if isinstance(stats, dict) and (
        "n_input_tokens" in stats or "n_output_tokens" in stats
    ):
        return RunUsage(
            path=result_path,
            trial_name=result_path.parent.name,
            material=material_from_result(data, result_path),
            day=day,
            finished=bool(data.get("finished_at")),
            has_diagnostics=any(result_path.parent.rglob("verifier/diagnostics.json")),
            input_tokens=as_int(stats.get("n_input_tokens")),
            cache_tokens=as_int(stats.get("n_cache_tokens")),
            output_tokens=as_int(stats.get("n_output_tokens")),
            observed_cost_usd=as_optional_float(stats.get("cost_usd")),
        )

    return None


def collect_runs(
    input_folder: Path,
    included_days: set[str] | None,
    include_materials: set[str] | None = None,
) -> list[RunUsage]:
    trial_runs: list[RunUsage] = []
    aggregate_runs: list[RunUsage] = []

    for result_path in sorted(input_folder.rglob("result.json")):
        run = load_run_usage(result_path)
        if run is None:
            continue
        if included_days is not None and run.day not in included_days:
            continue
        if include_materials is not None and run.material not in include_materials:
            continue
        if not run.finished or not run.has_diagnostics:
            continue

        if result_path.parent.parent == input_folder:
            aggregate_runs.append(run)
        else:
            trial_runs.append(run)

    return trial_runs or aggregate_runs


def normalize_model_name(name: str) -> str:
    lowered = name.strip().lower()
    for model_name in MODEL_PRICES:
        if model_name.lower() == lowered:
            return model_name
    matches = [model_name for model_name in MODEL_PRICES if lowered in model_name.lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise SystemExit(
            f"Ambiguous model {name!r}. Matches: {', '.join(matches)}"
        )
    raise SystemExit(
        f"Unknown model {name!r}. Known models: {', '.join(MODEL_PRICES)}"
    )


def cost_for_tokens(
    input_tokens: int, output_tokens: int, input_price: float, output_price: float
) -> float:
    return (input_tokens / 1_000_000) * input_price + (
        output_tokens / 1_000_000
    ) * output_price


def cost_for_cached_tokens(
    uncached_input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    input_price: float,
    cache_input_price: float,
    output_price: float,
) -> float:
    return (
        (uncached_input_tokens / 1_000_000) * input_price
        + (cached_input_tokens / 1_000_000) * cache_input_price
        + (output_tokens / 1_000_000) * output_price
    )


def money(value: float) -> str:
    return f"${value:,.4f}"


def print_cost_summary(
    runs: list[RunUsage],
    model_name: str,
    days: set[str] | None,
    input_folder: Path,
    title: str,
    selected_materials: set[str] | None = None,
) -> None:
    price = MODEL_PRICES[model_name]
    input_price = price["input"]
    cache_input_price = price.get("cache_input")
    output_price = price["output"]

    if input_price is None:
        raise SystemExit(
            f"{model_name} is missing an input-token price in the table, so total cost "
            "cannot be calculated."
        )

    total_input = sum(run.input_tokens for run in runs)
    total_cache = sum(run.cache_tokens for run in runs)
    total_output = sum(run.output_tokens for run in runs)
    observed_costs = [run.observed_cost_usd for run in runs if run.observed_cost_usd is not None]
    uncached_input = max(total_input - total_cache, 0)

    output_prices = output_price if isinstance(output_price, tuple) else (output_price,)
    total_costs = [
        cost_for_tokens(total_input, total_output, input_price, out_price)
        for out_price in output_prices
    ]
    cached_costs = [
        cost_for_cached_tokens(
            uncached_input,
            total_cache,
            total_output,
            input_price,
            cache_input_price,
            out_price,
        )
        for out_price in output_prices
    ] if cache_input_price is not None else []
    uncached_costs = [
        cost_for_tokens(uncached_input, total_output, input_price, out_price)
        for out_price in output_prices
    ]

    materials = {run.material for run in runs}

    print(title)
    print(f"Input folder: {input_folder}")
    print(f"Model: {model_name}")
    print(
        "Included days: "
        + (", ".join(sorted(days)) if days is not None else "all days")
    )
    if selected_materials is not None:
        missing = sorted(selected_materials - materials)
        extra_runs = len(runs) - len(materials)
        print(f"Selected materials matched: {len(materials)}/{len(selected_materials)}")
        if extra_runs:
            print(f"Extra runs from repeated materials: {extra_runs}")
        if missing:
            print(f"Missing selected materials: {', '.join(missing)}")
    print(f"Runs counted: {len(runs)}")
    print(f"Materials counted: {len(materials)}")
    print("")
    print(f"Input tokens: {total_input:,}")
    print(f"Cached tokens included in input: {total_cache:,}")
    print(f"Non-cached input tokens: {uncached_input:,}")
    print(f"Output tokens: {total_output:,}")
    print("")

    if observed_costs:
        observed_total = sum(observed_costs)
        print(f"Observed cost_usd total from result.json: {money(observed_total)}")
        print(f"Observed cost_usd average per priced run: {money(observed_total / len(observed_costs))}")
        if len(observed_costs) != len(runs):
            print(f"Runs missing observed cost_usd: {len(runs) - len(observed_costs)}")
        print("")

    if len(total_costs) == 1:
        if cached_costs:
            print(f"Total cost with cache-hit pricing: {money(cached_costs[0])}")
            print(f"Average cost per run with cache-hit pricing: {money(cached_costs[0] / len(runs))}")
            print(f"Cache-hit input price: ${cache_input_price:g} per 1M tokens")
        else:
            print(f"Total cost, counting cached tokens at full input price: {money(total_costs[0])}")
            print(f"Average cost per run: {money(total_costs[0] / len(runs))}")
            print(f"Total cost, ignoring cached input tokens: {money(uncached_costs[0])}")
            print(f"Average cost per run ignoring cached input: {money(uncached_costs[0] / len(runs))}")
    else:
        if cached_costs:
            print(
                "Total cost with cache-hit pricing: "
                f"{money(cached_costs[0])} - {money(cached_costs[-1])}"
            )
            print(
                "Average cost per run with cache-hit pricing: "
                f"{money(cached_costs[0] / len(runs))} - {money(cached_costs[-1] / len(runs))}"
            )
            print(f"Cache-hit input price: ${cache_input_price:g} per 1M tokens")
        else:
            print(
                "Total cost, counting cached tokens at full input price: "
                f"{money(total_costs[0])} - {money(total_costs[-1])}"
            )
            print(
                "Average cost per run: "
                f"{money(total_costs[0] / len(runs))} - {money(total_costs[-1] / len(runs))}"
            )
            print(
                "Total cost, ignoring cached input tokens: "
                f"{money(uncached_costs[0])} - {money(uncached_costs[-1])}"
            )
            print(
                "Average cost per run ignoring cached input: "
                f"{money(uncached_costs[0] / len(runs))} - {money(uncached_costs[-1] / len(runs))}"
            )

    if cache_input_price is None:
        print("")
        print("Note: cached-token prices are unknown here, so both full-input and")
        print("ignore-cached-input estimates are shown.")


def parse_days(days: list[str]) -> set[str] | None:
    if not days:
        return None
    return {normalize_day(day) for day in days}


def parse_simple_positionals() -> tuple[Path, str, set[str]]:
    if len(sys.argv) == 1:
        return Path(INPUT_FOLDER), normalize_model_name(MODEL_NAME), {
            normalize_day(day) for day in DAYS_TO_INCLUDE
        }

    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python scripts/calculate_run_costs.py INPUT_FOLDER MODEL_NAME DAY [DAY ...]\n"
            'Example: python scripts/calculate_run_costs.py jobsDeepseekFlashTerminus2 "Deepseek V4 Flash" 07-12 07-13'
        )

    input_folder = Path(sys.argv[1]).expanduser()
    model_name = normalize_model_name(sys.argv[2])
    days = {normalize_day(day) for day in sys.argv[3:]}
    return input_folder, model_name, days


def main() -> None:
    input_folder, model_name, days = parse_simple_positionals()

    if not input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_folder}")
    if not days:
        raise SystemExit("No days selected.")

    runs = collect_runs(input_folder, days)
    if not runs:
        raise SystemExit(
            f"No result.json files with token usage found for days: {', '.join(sorted(days))}"
        )

    print_cost_summary(runs, model_name, days, input_folder, "Selected runs")

    compare_input_folder = Path(COMPARE_INPUT_FOLDER).expanduser()
    compare_model_name = normalize_model_name(COMPARE_MODEL_NAME)
    compare_days = parse_days(COMPARE_DAYS_TO_INCLUDE)
    selected_materials = {run.material for run in runs}

    if compare_input_folder.is_dir():
        compare_runs = collect_runs(
            compare_input_folder,
            compare_days,
            include_materials=selected_materials,
        )
        print("")
        print("-" * 72)
        print("")
        if compare_runs:
            print_cost_summary(
                compare_runs,
                compare_model_name,
                compare_days,
                compare_input_folder,
                "Comparison runs for the same selected materials",
                selected_materials=selected_materials,
            )
        else:
            print("Comparison runs for the same selected materials")
            print(f"Input folder: {compare_input_folder}")
            print("No finished runs with diagnostics.json matched the selected materials.")
    else:
        print("")
        print(f"Comparison folder does not exist: {compare_input_folder}")


if __name__ == "__main__":
    main()
