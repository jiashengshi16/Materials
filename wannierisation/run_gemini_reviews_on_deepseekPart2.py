#!/usr/bin/env python3
"""Contradiction-only consolidation of Gemini self-debug reviews for DeepSeek.

Pipeline
========

1. Per-candidate-material contradiction review
   For every unique candidate material referenced by the candidate CSV, collect
   all per-run reviews and the original LLM-readable evidence for those runs.

   Gemini's only synthesis task is to decide whether the reports make genuinely
   contradictory claims about the same or materially equivalent run situation.
   A detailed report is written only when such a contradiction exists. Different
   recipes, failure stages, outcomes, or chemistry-specific situations are not
   contradictions and must not be converted into general lessons.

2. Per-target cross-candidate contradiction review
   For each target material, stage every candidate-material contradiction report,
   every source run used to produce it, and compact target-side task inputs.

   The target report preserves candidate-material contradictions and adds detail
   only for genuine cross-candidate conceptual contradictions. It does not create
   cross-material consensus, transferable chemical rules, or a target recipe.
   When no contradiction is found, the report is intentionally minimal.

The final target outputs retain the existing compatibility layout:

    <target>/target_consensus/self_debug_report.md
    <target>/target_consensus/self_debug_report.json

A self-link candidate CSV is generated so the existing chemically-similar
DeepSeek context loader can load exactly one contradiction-focused target report.

Large/binary raw numerical files (``.amn``, ``.mmn``, ``.chk``, ``*_hr.dat`` and
QE save trees) are intentionally not copied directly. The original reviewer also
reduces ``.amn`` deterministically before LLM inspection. All other relevant
text/JSON evidence is staged from the original run directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_BIN = "gemini"
DEFAULT_MAX_CONCURRENT_MATERIALS = 12
DEFAULT_MAX_CONCURRENT_TARGETS = 12
DEFAULT_MAX_ATTEMPTS = 5
CONTRADICTION_SCHEMA_VERSION = 3

REVIEW_ROOT_RELATIVE = Path("jobsGeminiReviewsDeepseekIter2") / "gemini_self_debug_reviews"
MATERIAL_OUTPUT_ROOT_RELATIVE = (
    Path("jobsGeminiReviewsDeepseekIter2") / "gemini_material_consolidated_reviews"
)
TARGET_OUTPUT_ROOT_RELATIVE = (
    Path("jobsGeminiReviewsDeepseekIter2") / "gemini_target_consolidated_reviews"
)
DEFAULT_RUN_ROOT_RELATIVES = [Path("jobsDeepseekProTerminus2ControlledIter2")]
DATASET_ROOT_RELATIVE = Path("harbor_datasets") / "wannier_200"
CANDIDATE_TABLE_RELATIVE = Path("temp.csv")
COMPATIBILITY_CSV_RELATIVE = (
    Path("jobsGeminiReviewsDeepseekIter2")
    / "consolidated_target_context_candidates.csv"
)
INDEX_RELATIVE = Path("jobsGeminiReviewsDeepseekIter2") / "gemini_consolidation_index.json"
REVIEWER_SCRIPT_RELATIVE = Path("scripts") / "run_gemini_reviews_on_deepseekPart1.py"

MATERIAL_CASE_NAME = "material_consensus"
TARGET_CASE_NAME = "target_consensus"
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.+\-]+$")

# Same LLM-readable evidence surface as the original reviewer.
ATTEMPT_PATTERNS = (
    "{material}.win",
    "{material}.wout",
    "run_manifest.json",
    "{material}.eig",
    "{material}.nnkp",
    "{material}.pp.log",
    "*.pp.log",
    "*pw2wan*.log",
    "*.werr",
    "*.err",
)

WORKFLOW_RELATIVE_PATHS = (
    "artifacts/app/workflow/recipe_request.json",
    "artifacts/app/workflow/compile_recipe_report.json",
    "artifacts/app/workflow/locked_runner.log",
    "artifacts/app/workflow/LOCKED_RECIPE.json",
    "artifacts/app/workflow/DECISIONS.md",
    "artifacts/app/workflow/locked_runner_state.json",
    "config.json",
    "exception.txt",
    "result.json",
    "trial.log",
    "verifier/test-stdout.txt",
    "verifier/test-stderr.txt",
)

TARGET_TASK_RELATIVE_PATHS = (
    "instruction.md",
    "instructions.md",
    "prompt.md",
    "task.md",
    "metadata.json",
    "task.toml",
    "nscf/input/nscf.in",
    "scf/input/scf.in",
)


@dataclass(frozen=True)
class RunReview:
    material: str
    case_id: str
    json_path: Path
    md_path: Path | None
    data: dict[str, Any]


@dataclass(frozen=True)
class SourceCase:
    material: str
    case_id: str
    run_root: Path
    job_dir: Path
    trial_dir: Path
    attempt_dir: Path


@dataclass(frozen=True)
class MaterialConsolidationResult:
    material: str
    output_dir: Path
    json_path: Path
    md_path: Path
    source_manifest: dict[str, Any]
    skipped_as_fresh: bool


@dataclass(frozen=True)
class TargetConsolidationResult:
    target: str
    output_dir: Path
    json_path: Path
    md_path: Path
    candidates_requested: tuple[str, ...]
    candidates_used: tuple[str, ...]
    missing_candidates: tuple[str, ...]
    source_manifest: dict[str, Any]
    skipped_as_fresh: bool


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_file_if_present(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    try:
        copy_file(src, dst)
    except OSError:
        return False
    return True


def copy_tree_hardlink(src: Path, dst: Path) -> None:
    """Copy a tree, preferring hard links so target staging is inexpensive."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            copy_file(path, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_material_name(name: str, *, field: str) -> str:
    value = name.strip()
    if not value:
        raise ValueError(f"{field} cannot be blank")
    if value in {".", ".."} or not SAFE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"unsafe {field} {value!r}; expected letters, numbers, '.', '_', '+', or '-'"
        )
    return value


def safe_component(value: str, *, fallback: str = "case") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", value).strip("._")
    return cleaned or fallback


def relative_display(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def path_from_cli(value: Path | None, default: Path) -> Path:
    return (value if value is not None else default).expanduser().resolve()


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def candidate_materials_from_include_only_csv(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"candidate include-only CSV does not exist: {path}")

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

        for row_number, row in enumerate(reader, start=2):
            raw_target = (row.get(target_column) or "").strip()
            raw_candidate = (row.get("candidate_material") or "").strip()
            if not raw_target and not raw_candidate:
                continue
            if not raw_target or not raw_candidate:
                print(
                    f"Warning: ignoring incomplete row {row_number} in {path}",
                    file=sys.stderr,
                )
                continue
            try:
                target = validate_material_name(raw_target, field="target material")
                candidate = validate_material_name(raw_candidate, field="candidate material")
            except ValueError as exc:
                raise SystemExit(f"Invalid row {row_number} in {path}: {exc}") from exc
            values = candidates_by_target.setdefault(target, [])
            if candidate not in values:
                values.append(candidate)

    if not candidates_by_target:
        raise SystemExit(f"No usable target/candidate rows found in {path}")
    return candidates_by_target


def collect_run_reviews(material: str, reviews_root: Path) -> list[RunReview]:
    material_dir = reviews_root / material
    if not material_dir.is_dir():
        return []

    reviews: list[RunReview] = []
    for json_path in sorted(material_dir.rglob("self_debug_report.json")):
        data = read_json_object(json_path)
        if not data:
            print(f"Warning: skipping unreadable review {json_path}", file=sys.stderr)
            continue
        reported_material = data.get("material")
        if isinstance(reported_material, str) and reported_material.strip() != material:
            print(
                f"Warning: skipping {json_path}; material mismatch {reported_material!r}",
                file=sys.stderr,
            )
            continue
        case_id = str(data.get("case_id") or json_path.parent.name)
        md_path = json_path.with_name("self_debug_report.md")
        reviews.append(
            RunReview(
                material=material,
                case_id=case_id,
                json_path=json_path,
                md_path=md_path if md_path.is_file() else None,
                data=data,
            )
        )
    return reviews


# ---------------------------------------------------------------------------
# Original source-case resolution and evidence staging
# ---------------------------------------------------------------------------


def candidate_trial_dirs(job_dir: Path) -> list[Path]:
    if any((job_dir / name).exists() for name in ("artifacts", "agent", "verifier")):
        return [job_dir]
    trials = [
        path
        for path in sorted(job_dir.iterdir())
        if path.is_dir()
        and any((path / name).exists() for name in ("artifacts", "agent", "verifier"))
    ]
    return trials or [job_dir]


def trial_attempt_dir(trial_dir: Path) -> Path:
    candidates = (
        trial_dir / "artifacts" / "attempt_1",
        trial_dir / "artifacts" / "logs" / "artifacts" / "attempt_1",
        trial_dir / "attempt_1",
        trial_dir / "artifacts",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    matches = sorted(path for path in trial_dir.rglob("attempt_1") if path.is_dir())
    return matches[0] if matches else trial_dir


def locate_source_case(review: RunReview, run_roots: list[Path]) -> SourceCase | None:
    job_folder = review.data.get("job_folder")
    trial_folder = review.data.get("trial_folder")
    reported_root = review.data.get("run_root")
    if not isinstance(job_folder, str) or not job_folder:
        return None

    preferred = [root for root in run_roots if root.name == reported_root]
    ordered_roots = preferred + [root for root in run_roots if root not in preferred]

    for run_root in ordered_roots:
        job_dir = run_root / job_folder
        if not job_dir.is_dir():
            continue

        trial_dir: Path | None = None
        if isinstance(trial_folder, str) and trial_folder:
            if trial_folder == job_dir.name and any(
                (job_dir / name).exists() for name in ("artifacts", "agent", "verifier")
            ):
                trial_dir = job_dir
            elif (job_dir / trial_folder).is_dir():
                trial_dir = job_dir / trial_folder
            else:
                for candidate in candidate_trial_dirs(job_dir):
                    if candidate.name == trial_folder:
                        trial_dir = candidate
                        break
        if trial_dir is None:
            trials = candidate_trial_dirs(job_dir)
            if len(trials) == 1:
                trial_dir = trials[0]
            else:
                # Try the case id suffix as a last deterministic discriminator.
                matching = [trial for trial in trials if trial.name in review.case_id]
                if len(matching) == 1:
                    trial_dir = matching[0]
        if trial_dir is None:
            continue

        return SourceCase(
            material=review.material,
            case_id=review.case_id,
            run_root=run_root,
            job_dir=job_dir,
            trial_dir=trial_dir,
            attempt_dir=trial_attempt_dir(trial_dir),
        )
    return None


def find_attempt_file(attempt_dir: Path, material: str, suffix: str) -> Path | None:
    exact = attempt_dir / f"{material}{suffix}"
    if exact.is_file():
        return exact
    matches = sorted(path for path in attempt_dir.glob(f"*{suffix}") if path.is_file())
    return matches[0] if len(matches) == 1 else None


def first_user_message_from_trajectory(trial_dir: Path) -> str | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    data = read_json_object(trajectory_path)
    if not data:
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
        for record in messages:
            if not isinstance(record, dict) or record.get("role") not in {"user", "human"}:
                continue
            content = record.get("content")
            if isinstance(content, str) and content.strip():
                return content.rstrip() + "\n"
    return None


def dataset_instruction_path(dataset_root: Path, material: str) -> Path | None:
    material_dir = dataset_root / material
    for name in ("instruction.md", "instructions.md", "prompt.md", "task.md"):
        candidate = material_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(material_dir.glob("*instruction*.md"))
    return matches[0] if len(matches) == 1 else None


def load_reviewer_module(path: Path | None) -> ModuleType | None:
    if path is None or not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("gemini_raw_reviewer_helper", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(
            f"Warning: could not import reviewer helper {path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.modules.pop(spec.name, None)
        return None
    return module


def generate_amn_summary(
    source: SourceCase,
    review: RunReview,
    reviewer_module: ModuleType | None,
) -> dict[str, Any] | None:
    if reviewer_module is None:
        return None
    required = ("TrialCase", "summarize_amn_case")
    if not all(hasattr(reviewer_module, name) for name in required):
        return None
    manifest = read_json_object(source.attempt_dir / "run_manifest.json")
    try:
        if hasattr(reviewer_module, "parse_job_name"):
            job_metadata = dict(reviewer_module.parse_job_name(source.job_dir))
        else:
            job_metadata = {}
        job_metadata.update(
            {
                "run_root": source.run_root.name,
                "run_root_path": str(source.run_root),
            }
        )
        case = reviewer_module.TrialCase(
            material=source.material,
            job_dir=source.job_dir,
            trial_dir=source.trial_dir,
            attempt_dir=source.attempt_dir,
            case_id=source.case_id,
            job_metadata=job_metadata,
            manifest=manifest,
        )
        summary = reviewer_module.summarize_amn_case(case)
    except Exception as exc:
        print(
            f"Warning: AMN summary generation failed for {review.material} {review.case_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    return summary if isinstance(summary, dict) else None


def source_evidence_files(source: SourceCase) -> list[tuple[Path, Path]]:
    files: dict[tuple[Path, Path], None] = {}
    material = source.material

    for raw_pattern in ATTEMPT_PATTERNS:
        pattern = raw_pattern.format(material=material)
        for src in sorted(source.attempt_dir.glob(pattern)):
            if src.is_file():
                rel = Path("artifacts") / "attempt_1" / src.name
                files[(src, rel)] = None

    for relative in WORKFLOW_RELATIVE_PATHS:
        src = source.trial_dir / relative
        if src.is_file():
            files[(src, Path("workflow_contract") / relative)] = None

    compile_dir = source.trial_dir / "artifacts" / "app" / "workflow" / "compile_recipe_reports"
    for src in sorted(compile_dir.glob("compile_attempt_*.json")):
        if src.is_file():
            rel = (
                Path("workflow_contract")
                / "artifacts"
                / "app"
                / "workflow"
                / "compile_recipe_reports"
                / src.name
            )
            files[(src, rel)] = None

    for name in ("trajectory.json",):
        src = source.trial_dir / "agent" / name
        if src.is_file():
            files[(src, Path("agent") / name)] = None

    diagnostics = source.trial_dir / "verifier" / "diagnostics.json"
    if diagnostics.is_file():
        files[(diagnostics, Path("verifier") / "diagnostics.json")] = None

    return list(files)


def retained_case_files_dir(review: RunReview) -> Path | None:
    candidate = review.json_path.parent / "case_files"
    return candidate if candidate.is_dir() else None


def evidence_manifest_for_review(
    review: RunReview,
    source: SourceCase | None,
    dataset_root: Path,
) -> dict[str, Any]:
    retained = retained_case_files_dir(review)
    if retained is not None:
        files = [path for path in sorted(retained.rglob("*")) if path.is_file()]
        return {
            "evidence_mode": "retained_case_files",
            "source_case_files": str(retained),
            "files": [
                {
                    "path": str(path.relative_to(retained)),
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        }

    if source is None:
        return {
            "evidence_mode": "missing",
            "error": "could not locate original job/trial directory",
            "files": [],
        }

    files = source_evidence_files(source)
    manifest_files = [
        {
            "path": str(rel),
            "source": str(src),
            "sha256": sha256_file(src),
        }
        for src, rel in files
    ]
    trajectory_prompt = first_user_message_from_trajectory(source.trial_dir)
    instruction_path = dataset_instruction_path(dataset_root, review.material)
    instruction_digest = None
    instruction_source = None
    if trajectory_prompt is not None:
        instruction_digest = hashlib.sha256(trajectory_prompt.encode("utf-8")).hexdigest()
        instruction_source = "agent/trajectory.json:first user message"
    elif instruction_path is not None:
        instruction_digest = sha256_file(instruction_path)
        instruction_source = str(instruction_path)

    return {
        "evidence_mode": "reconstructed_from_original_run",
        "run_root": str(source.run_root),
        "job_dir": str(source.job_dir),
        "trial_dir": str(source.trial_dir),
        "attempt_dir": str(source.attempt_dir),
        "files": manifest_files,
        "original_task_instruction_source": instruction_source,
        "original_task_instruction_sha256": instruction_digest,
        "amn_source_present": find_attempt_file(source.attempt_dir, review.material, ".amn") is not None,
    }


def stage_review_evidence(
    case_dir: Path,
    review: RunReview,
    source: SourceCase | None,
    dataset_root: Path,
    reviewer_module: ModuleType | None,
) -> dict[str, Any]:
    evidence_dir = case_dir / "case_files"
    retained = retained_case_files_dir(review)
    if retained is not None:
        copy_tree_hardlink(retained, evidence_dir)
        return {
            "status": "available",
            "mode": "retained_case_files",
            "path": str(evidence_dir.relative_to(case_dir)),
        }

    if source is None:
        return {
            "status": "missing",
            "mode": "unresolved_source_case",
            "reason": "Could not locate the original job/trial directory from the per-run report metadata.",
        }

    staged: list[dict[str, str]] = []
    for src, rel in source_evidence_files(source):
        dst = evidence_dir / rel
        if copy_file_if_present(src, dst):
            staged.append({"source": str(src), "staged": str(rel)})

    task_text = first_user_message_from_trajectory(source.trial_dir)
    task_source = "agent/trajectory.json:first user message"
    if task_text is None:
        instruction_path = dataset_instruction_path(dataset_root, review.material)
        if instruction_path is not None:
            task_text = instruction_path.read_text(encoding="utf-8", errors="replace")
            task_source = str(instruction_path)
        else:
            task_text = "Original task instructions were not available for this case.\n"
            task_source = "not available"
    write_text(evidence_dir / "original_task_instructions.md", task_text)

    amn_summary = generate_amn_summary(source, review, reviewer_module)
    amn_status = "unavailable"
    if amn_summary is not None:
        write_json(
            evidence_dir / "artifacts" / "attempt_1" / f"{review.material}.amn_summary.json",
            amn_summary,
        )
        amn_status = str(amn_summary.get("status") or "generated")

    metadata = {
        "material": review.material,
        "case_id": review.case_id,
        "source_run_root": str(source.run_root),
        "source_job_path": str(source.job_dir),
        "source_trial_path": str(source.trial_dir),
        "source_attempt_path": str(source.attempt_dir),
        "original_task_instructions_source": task_source,
        "staged_files": staged,
        "amn_summary_status": amn_status,
        "large_numeric_files_intentionally_not_staged": [
            ".amn (reduced into .amn_summary.json when helper is available)",
            ".mmn",
            ".chk",
            "*_hr.dat",
            "QE save trees",
        ],
    }
    write_json(evidence_dir / "case_metadata.json", metadata)
    return {
        "status": "available",
        "mode": "reconstructed_from_original_run",
        "path": str(evidence_dir.relative_to(case_dir)),
        "amn_summary_status": amn_status,
        "staged_file_count": len(staged),
    }


# ---------------------------------------------------------------------------
# Source manifests and validation
# ---------------------------------------------------------------------------


def material_source_manifest(
    material: str,
    reviews: list[RunReview],
    source_cases: dict[str, SourceCase | None],
    *,
    reviews_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for review in reviews:
        item: dict[str, Any] = {
            "case_id": review.case_id,
            "review_json_path": relative_display(review.json_path, reviews_root),
            "review_json_sha256": sha256_file(review.json_path),
            "source_evidence": evidence_manifest_for_review(
                review,
                source_cases.get(review.case_id),
                dataset_root,
            ),
        }
        if review.md_path is not None:
            item["review_md_path"] = relative_display(review.md_path, reviews_root)
            item["review_md_sha256"] = sha256_file(review.md_path)
        sources.append(item)
    return {
        "schema_version": CONTRADICTION_SCHEMA_VERSION,
        "stage": "per_candidate_material_contradiction_review",
        "material": material,
        "source_run_count": len(reviews),
        "sources": sources,
    }


def target_source_manifest(
    target: str,
    candidates_requested: list[str],
    material_results: dict[str, MaterialConsolidationResult],
    *,
    material_output_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate in candidates_requested:
        result = material_results.get(candidate)
        if result is None or not result.json_path.is_file() or not result.md_path.is_file():
            missing.append(candidate)
            continue
        sources.append(
            {
                "candidate_material": candidate,
                "json_path": relative_display(result.json_path, material_output_root),
                "json_sha256": sha256_file(result.json_path),
                "md_path": relative_display(result.md_path, material_output_root),
                "md_sha256": sha256_file(result.md_path),
                "evidence_manifest_sha256": sha256_file(result.output_dir / "source_manifest.json"),
            }
        )

    target_files: list[dict[str, str]] = []
    target_dir = dataset_root / target
    for relative in TARGET_TASK_RELATIVE_PATHS:
        src = target_dir / relative
        if src.is_file():
            target_files.append({"path": relative, "sha256": sha256_file(src)})

    return {
        "schema_version": CONTRADICTION_SCHEMA_VERSION,
        "stage": "per_target_cross_candidate_contradiction_review",
        "target_material": target,
        "candidate_materials_requested": candidates_requested,
        "candidate_materials_used": [item["candidate_material"] for item in sources],
        "missing_candidate_materials": missing,
        "candidate_sources": sources,
        "target_task_files": target_files,
    }


def source_manifest_matches(output_dir: Path, expected: dict[str, Any]) -> bool:
    path = output_dir / "source_manifest.json"
    return path.is_file() and read_json_object(path) == expected


def _list_field(data: dict[str, Any], key: str) -> list[Any] | None:
    value = data.get(key)
    return value if isinstance(value, list) else None


def _material_input_case_ids(output_dir: Path) -> list[str]:
    index = read_json_object(output_dir / "inputs" / "index.json")
    reviews = index.get("run_reviews")
    if not isinstance(reviews, list):
        return []
    return [
        str(item.get("case_id"))
        for item in reviews
        if isinstance(item, dict) and item.get("case_id")
    ]


def _target_input_index(output_dir: Path) -> dict[str, Any]:
    return read_json_object(output_dir / "inputs" / "index.json")


def _target_input_candidate_contradictions(output_dir: Path) -> set[str]:
    """Return candidates whose material-level report found a contradiction."""
    index = _target_input_index(output_dir)
    entries = index.get("candidate_consolidations")
    if not isinstance(entries, list):
        return set()

    result: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("candidate_material")
        relative = entry.get("material_consolidated_json")
        if not isinstance(candidate, str) or not isinstance(relative, str):
            continue
        data = read_json_object(output_dir / relative)
        if data.get("contradiction_found") is True:
            result.add(candidate)
    return result


def material_output_is_valid(output_dir: Path) -> bool:
    md_path = output_dir / "material_consolidated.md"
    json_path = output_dir / "material_consolidated.json"
    if not md_path.is_file() or not md_path.read_text(encoding="utf-8").strip():
        return False

    data = read_json_object(json_path)
    if data.get("schema_version") != CONTRADICTION_SCHEMA_VERSION:
        return False
    if not isinstance(data.get("material"), str) or not data["material"].strip():
        return False
    if data.get("case_id") != "material_contradiction_review":
        return False

    verdict = data.get("verdict")
    if verdict not in {"contradiction_found", "no_contradiction", "insufficient_evidence"}:
        return False
    contradiction_found = data.get("contradiction_found")
    if not isinstance(contradiction_found, bool):
        return False
    if not isinstance(data.get("summary"), str):
        return False

    required_lists = (
        "source_case_ids",
        "source_evidence_checks_performed",
        "review_claims_corrected_by_source_evidence",
        "contradictions",
        "non_contradictory_differences",
        "evidence_gaps",
    )
    if any(_list_field(data, key) is None for key in required_lists):
        return False

    contradictions = data["contradictions"]
    if contradiction_found != bool(contradictions):
        return False
    if verdict == "contradiction_found" and not contradiction_found:
        return False
    if verdict in {"no_contradiction", "insufficient_evidence"} and contradiction_found:
        return False

    expected_case_ids = set(_material_input_case_ids(output_dir))
    reported_case_ids = {str(value) for value in data["source_case_ids"]}
    if expected_case_ids and reported_case_ids != expected_case_ids:
        return False
    if data.get("source_run_count") != len(expected_case_ids):
        return False

    forbidden_generalization_fields = (
        "stable_findings",
        "run_dependent_findings",
        "safe_transferable_knowledge",
        "high_confidence_cross_candidate_knowledge",
        "conditional_knowledge",
        "candidate_specific_knowledge",
        "deepseek_decision_checks",
    )
    if any(data.get(key) not in (None, [], "") for key in forbidden_generalization_fields):
        return False

    # A no-contradiction result must remain intentionally sparse.
    if not contradiction_found:
        if len(data["non_contradictory_differences"]) > 5:
            return False
        md_text = md_path.read_text(encoding="utf-8")
        if len(md_text) > 1500 or len(md_text.splitlines()) > 20:
            return False
    else:
        for item in contradictions:
            if not isinstance(item, dict):
                return False
            required = {
                "topic",
                "contradiction_type",
                "status",
                "claims",
                "why_incompatible",
                "source_evidence",
                "resolution_or_conditions",
                "safe_deepseek_instruction",
            }
            if not required.issubset(item):
                return False
    return True


def target_output_validation_errors(output_dir: Path) -> list[str]:
    errors: list[str] = []
    md_path = output_dir / "ALL_SELF_DEBUG.md"
    json_path = output_dir / "ALL_SELF_DEBUG.json"
    if not md_path.is_file() or not md_path.read_text(encoding="utf-8").strip():
        errors.append("missing_or_empty_ALL_SELF_DEBUG.md")

    data = read_json_object(json_path)
    if data.get("schema_version") != CONTRADICTION_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if not isinstance(data.get("target_material"), str) or not data["target_material"].strip():
        errors.append("missing_target_material")
    if data.get("case_id") != "target_cross_candidate_contradiction_review":
        errors.append("case_id_mismatch")

    verdict = data.get("verdict")
    if verdict not in {"contradictions_found", "no_contradiction", "insufficient_evidence"}:
        errors.append("invalid_verdict")
    contradiction_found = data.get("contradiction_found")
    cross_found = data.get("cross_candidate_contradiction_found")
    if not isinstance(contradiction_found, bool) or not isinstance(cross_found, bool):
        errors.append("contradiction_booleans_must_be_bool")
    if not isinstance(data.get("summary"), str):
        errors.append("missing_summary")

    required_lists = (
        "candidate_materials_requested",
        "candidate_materials_used",
        "missing_candidate_materials",
        "source_evidence_checks_performed",
        "candidate_material_contradictions",
        "cross_candidate_conceptual_contradictions",
        "non_contradictory_differences",
        "deepseek_usage_constraints",
        "evidence_gaps",
    )
    missing_list_fields = [key for key in required_lists if _list_field(data, key) is None]
    if missing_list_fields:
        errors.append("missing_list_fields:" + ",".join(missing_list_fields))

    candidate_items = data.get("candidate_material_contradictions")
    cross_items = data.get("cross_candidate_conceptual_contradictions")
    candidate_items = candidate_items if isinstance(candidate_items, list) else []
    cross_items = cross_items if isinstance(cross_items, list) else []
    any_items = bool(candidate_items or cross_items)
    if contradiction_found != any_items:
        errors.append(
            f"contradiction_found_mismatch:got={contradiction_found!r},expected={any_items!r}"
        )
    if cross_found != bool(cross_items):
        errors.append(
            f"cross_candidate_contradiction_found_mismatch:got={cross_found!r},"
            f"expected={bool(cross_items)!r}"
        )
    if verdict == "contradictions_found" and not contradiction_found:
        errors.append("verdict_contradictions_found_but_no_contradictions")
    if verdict in {"no_contradiction", "insufficient_evidence"} and contradiction_found:
        errors.append(f"verdict_{verdict}_but_contradictions_present")

    input_index = _target_input_index(output_dir)
    for key in (
        "candidate_materials_requested",
        "candidate_materials_used",
        "missing_candidate_materials",
    ):
        expected = input_index.get(key)
        if isinstance(expected, list) and data.get(key) != expected:
            errors.append(f"{key}_mismatch")

    forbidden_generalization_fields = (
        "high_confidence_cross_candidate_knowledge",
        "conditional_knowledge",
        "candidate_specific_knowledge",
        "reconciled_cross_candidate_conflicts",
        "unresolved_cross_candidate_conflicts",
        "deepseek_decision_checks",
        "safe_transferable_knowledge",
        "stable_findings",
    )
    forbidden_present = [
        key for key in forbidden_generalization_fields if data.get(key) not in (None, [], "")
    ]
    if forbidden_present:
        errors.append("forbidden_generalization_fields:" + ",".join(forbidden_present))

    # Do not allow the target stage to silently discard a material-level contradiction.
    expected_candidates = _target_input_candidate_contradictions(output_dir)
    reported_candidates = {
        str(item.get("candidate_material"))
        for item in candidate_items
        if isinstance(item, dict) and item.get("candidate_material")
    }
    if expected_candidates != reported_candidates:
        errors.append(
            "candidate_material_contradictions_mismatch:"
            f"expected={sorted(expected_candidates)},reported={sorted(reported_candidates)}"
        )

    if not contradiction_found:
        non_contradictory = data.get("non_contradictory_differences")
        if isinstance(non_contradictory, list) and len(non_contradictory) > 5:
            errors.append("too_many_non_contradictory_differences_for_no_contradiction")
        if md_path.is_file():
            md_text = md_path.read_text(encoding="utf-8")
            if len(md_text) > 1500 or len(md_text.splitlines()) > 20:
                errors.append("markdown_too_long_for_no_contradiction")
    else:
        for item in candidate_items:
            if not isinstance(item, dict):
                errors.append("candidate_material_contradiction_item_not_object")
                continue
            if not {"candidate_material", "summary", "contradictions", "source_report"}.issubset(item):
                errors.append(
                    "candidate_material_contradiction_missing_fields:"
                    + str(item.get("candidate_material") or "<unknown>")
                )
            if not isinstance(item.get("contradictions"), list):
                errors.append(
                    "candidate_material_contradiction_contradictions_not_list:"
                    + str(item.get("candidate_material") or "<unknown>")
                )
        for item in cross_items:
            if not isinstance(item, dict):
                errors.append("cross_candidate_contradiction_item_not_object")
                continue
            required = {
                "topic",
                "status",
                "candidate_claims",
                "why_conceptually_incompatible",
                "chemistry_or_context_caveat",
                "source_evidence",
                "target_side_discriminator_checks",
                "safe_deepseek_instruction",
            }
            if not required.issubset(item):
                errors.append(
                    "cross_candidate_contradiction_missing_fields:"
                    + str(item.get("topic") or "<unknown>")
                )
    return errors


def target_output_is_valid(output_dir: Path) -> bool:
    return not target_output_validation_errors(output_dir)


def _write_skipped_run_status(output_dir: Path, *, reason: str) -> None:
    write_json(
        output_dir / "run_status.json",
        {
            "model": "not_invoked",
            "success": True,
            "skipped": True,
            "reason": reason,
            "attempts": [],
        },
    )


def write_single_run_material_output(
    output_dir: Path,
    material: str,
    reviews: list[RunReview],
) -> None:
    case_ids = [review.case_id for review in reviews]
    summary = "A contradiction review requires at least two runs; only one run was available."
    write_json(
        output_dir / "material_consolidated.json",
        {
            "schema_version": CONTRADICTION_SCHEMA_VERSION,
            "material": material,
            "case_id": "material_contradiction_review",
            "verdict": "insufficient_evidence",
            "source_run_count": len(reviews),
            "source_case_ids": case_ids,
            "contradiction_found": False,
            "summary": summary,
            "source_evidence_checks_performed": [],
            "review_claims_corrected_by_source_evidence": [],
            "contradictions": [],
            "non_contradictory_differences": [],
            "evidence_gaps": ["Fewer than two runs were available for same-material comparison."],
        },
    )
    write_text(
        output_dir / "material_consolidated.md",
        f"# Material contradiction review: {material}\n\n"
        f"**Result:** {summary}\n",
    )
    _write_skipped_run_status(output_dir, reason="fewer_than_two_runs")


def _candidate_material_contradiction_payloads(
    candidates_used: list[str],
    material_results: dict[str, MaterialConsolidationResult],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for candidate in candidates_used:
        result = material_results.get(candidate)
        if result is None:
            continue
        data = read_json_object(result.json_path)
        if data.get("contradiction_found") is not True:
            continue
        payloads.append(
            {
                "candidate_material": candidate,
                "source_case_id": data.get("case_id"),
                "summary": data.get("summary"),
                "contradictions": data.get("contradictions", []),
                "source_report": str(result.json_path),
            }
        )
    return payloads


def write_single_candidate_target_output(
    output_dir: Path,
    target: str,
    candidates_requested: list[str],
    candidates_used: list[str],
    missing_candidates: list[str],
    material_results: dict[str, MaterialConsolidationResult],
) -> None:
    material_contradictions = _candidate_material_contradiction_payloads(
        candidates_used,
        material_results,
    )
    contradiction_found = bool(material_contradictions)
    if contradiction_found:
        verdict = "contradictions_found"
        summary = (
            "Candidate-material contradictions were preserved, but fewer than two "
            "candidate materials were available for cross-candidate comparison."
        )
    else:
        verdict = "insufficient_evidence"
        summary = (
            "No candidate-material contradiction was present, and fewer than two "
            "candidate materials were available for cross-candidate comparison."
        )

    write_json(
        output_dir / "ALL_SELF_DEBUG.json",
        {
            "schema_version": CONTRADICTION_SCHEMA_VERSION,
            "material": target,
            "target_material": target,
            "case_id": "target_cross_candidate_contradiction_review",
            "verdict": verdict,
            "candidate_materials_requested": candidates_requested,
            "candidate_materials_used": candidates_used,
            "missing_candidate_materials": missing_candidates,
            "contradiction_found": contradiction_found,
            "cross_candidate_contradiction_found": False,
            "summary": summary,
            "source_evidence_checks_performed": [],
            "candidate_material_contradictions": material_contradictions,
            "cross_candidate_conceptual_contradictions": [],
            "non_contradictory_differences": [],
            "deepseek_usage_constraints": [
                "Use only the preserved contradiction warnings; no cross-candidate rule was inferred."
            ] if contradiction_found else [],
            "evidence_gaps": [
                "Fewer than two candidate materials were available for cross-candidate comparison."
            ],
        },
    )

    lines = [
        f"# Target contradiction review: {target}",
        "",
        f"**Result:** {summary}",
    ]
    if material_contradictions:
        lines.extend(["", "## Candidate-material contradictions"])
        for item in material_contradictions:
            lines.append(f"- **{item['candidate_material']}**: {item.get('summary') or 'Contradiction found.'}")
    write_text(output_dir / "ALL_SELF_DEBUG.md", "\n".join(lines) + "\n")
    _write_skipped_run_status(output_dir, reason="fewer_than_two_candidate_materials")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def material_prompt(material: str, run_count: int) -> str:
    return f"""# Contradiction-only same-material review: {material}

You are comparing {run_count} independently reviewed Wannierisation runs for the
same candidate material, `{material}`.

Your task is NOT to consolidate lessons, create best practices, recommend a future
recipe, or summarize every run. Your only synthesis task is to determine whether
the run reviews contain genuine contradictions.

The original LLM-readable source evidence for every run is staged under each input
case directory. Start with `inputs/index.json`. For every case, use
`self_debug_report.json` only as a claim index, then verify any potentially
contradictory claim against the relevant files under `case_files/`.

Do not browse the internet or inspect files outside this consolidation directory.

## What counts as a contradiction

Report a contradiction only when two or more reviews make incompatible claims
about the same or materially equivalent event, recipe feature, numerical signal,
or causal diagnosis, for example:

- the same projection recipe and outcome are called both defective and good;
- the same coordinate projector is claimed to be both atom-coincident and empty;
- one review says a specific observed warning occurred and another says it did not;
- the same run evidence is used to support opposite causal conclusions without a
  condition that makes both conclusions compatible.

## What does NOT count

Do not report a contradiction merely because:

- the recipes, windows, projections, failure stages, or outcomes differ;
- different runs expose different valid problems;
- one statement is a direct observation and another is only a hypothesis;
- both claims can coexist under explicit, evidence-backed conditions;
- the reports use different wording but reach compatible conclusions.

First compare exact recipes, coordinates, windows, failure stages, convergence,
localization, RMSE/reward, and the confidence level of each claim. Do not
majority-vote. Preserve threshold-dependent AMN language and never turn AMN
sensitivity into an unqualified rank-deficiency claim.

## Output policy

Write exactly:

- `material_consolidated.md`
- `material_consolidated.json`

Use schema version {CONTRADICTION_SCHEMA_VERSION}:

```json
{{
  "schema_version": {CONTRADICTION_SCHEMA_VERSION},
  "material": "{material}",
  "case_id": "material_contradiction_review",
  "verdict": "contradiction_found | no_contradiction | insufficient_evidence",
  "source_run_count": {run_count},
  "source_case_ids": [],
  "contradiction_found": false,
  "summary": "one concise sentence",
  "source_evidence_checks_performed": [],
  "review_claims_corrected_by_source_evidence": [],
  "contradictions": [],
  "non_contradictory_differences": [],
  "evidence_gaps": []
}}
```

If no genuine contradiction exists:

- set `verdict` to `no_contradiction`;
- set `contradiction_found` to `false`;
- leave `contradictions` empty;
- keep `non_contradictory_differences` empty unless one short entry is needed to
  explain why an apparent conflict was actually recipe- or situation-specific;
- make the Markdown minimal: a heading, one result sentence, and optionally a
  short evidence-gap sentence. Do not emit stable findings, lessons learned,
  transferable knowledge, candidate recommendations, or a general review.

If a genuine contradiction exists, each item in `contradictions` must include:

- `topic`;
- `contradiction_type`;
- `status`: `resolved | conditional | unresolved`;
- `claims`: the incompatible claims with case IDs and confidence levels;
- `why_incompatible`;
- `source_evidence` with staged paths;
- `resolution_or_conditions`;
- `safe_deepseek_instruction` explaining what a future run must not assume.

The detailed Markdown should contain only:

1. Contradiction summary
2. Source evidence checked
3. Detailed contradictions
4. Resolution or unresolved conditions
5. Safe DeepSeek usage constraints
6. Evidence gaps

Do not add sections for stable knowledge, good practices, or recommendations.
"""


def target_prompt(
    target: str,
    candidates_requested: list[str],
    candidates_used: list[str],
    missing_candidates: list[str],
    candidate_materials_with_contradictions: list[str],
) -> str:
    has_preserved_contradictions = bool(candidate_materials_with_contradictions)

    example_verdict = (
        "contradictions_found"
        if has_preserved_contradictions
        else "contradictions_found | no_contradiction | insufficient_evidence"
    )

    example_candidate_contradictions = [
        {
            "candidate_material": candidate,
            "summary": (
                "Copy the summary from the staged "
                "material_consolidated.json."
            ),
            "contradictions": [
                {
                    "instruction": (
                        "Copy the contradiction objects from the staged "
                        "material_consolidated.json."
                    )
                }
            ],
            "source_report": (
                "Use the staged material_consolidated.json path "
                "from inputs/index.json."
            ),
        }
        for candidate in candidate_materials_with_contradictions
    ]

    return f"""# Contradiction-only cross-candidate review for target: {target}

Candidate materials requested:
{', '.join(candidates_requested) or 'none'}

Candidate materials available:
{', '.join(candidates_used) or 'none'}

Missing candidate materials:
{', '.join(missing_candidates) or 'none'}

Candidate materials whose material-level reports already have
`contradiction_found: true` and therefore MUST appear in
`candidate_material_contradictions`:
{', '.join(candidate_materials_with_contradictions) or 'none'}

Your task is NOT to consolidate cross-material lessons or create a target recipe.
Produce a report containing only contradiction information.

This stage has access to:

- each candidate's `material_consolidated.json/.md` contradiction review;
- every underlying per-run review and staged source-evidence tree under that
  candidate's `source_runs/` directory;
- compact target task inputs under `target_material_files/` when available.

Start with `inputs/index.json`.

## Required two-part check

1. Preserve every candidate-material contradiction already found by the material
   stage. Do not silently drop, weaken, or rewrite it into a general rule.
2. Independently check for genuine conceptual contradictions across different
   candidate materials by comparing the underlying per-run claims and evidence.

A cross-candidate conceptual contradiction exists only when reports make opposite
claims about the same diagnostic mechanism or proposed remedy, such as treating
same-site same-l padding as categorically fatal in one candidate but recommending
it as the remedy in another, without clearly stated chemical or numerical
conditions that reconcile the claims.

Different materials are expected to have different chemistry and lessons. Do not
flag a contradiction merely because candidates used different recipes, orbitals,
windows, failure stages, or material-specific explanations. Chemistry can explain
an apparent conflict; when it does, classify it as a non-contradictory difference,
not as shared knowledge.

Do not invent a winning rule. Do not write statements such as "hit-the-cap is
standard", "spread outliers are physical", "symmetry is more important than
rank", or equivalent broad conclusions unless the contradiction itself is a claim
that such unsupported generalization appeared. Your output is a conflict map, not
a consensus document.

## Output policy

Write exactly:

- `ALL_SELF_DEBUG.md`
- `ALL_SELF_DEBUG.json`

Use schema version {CONTRADICTION_SCHEMA_VERSION}:

```json
{{
  "schema_version": {CONTRADICTION_SCHEMA_VERSION},
  "material": "{target}",
  "target_material": "{target}",
  "case_id": "target_cross_candidate_contradiction_review",
  "verdict": "{example_verdict}",
  "candidate_materials_requested": {json.dumps(candidates_requested)},
  "candidate_materials_used": {json.dumps(candidates_used)},
  "missing_candidate_materials": {json.dumps(missing_candidates)},
  "contradiction_found": {json.dumps(has_preserved_contradictions)},
  "cross_candidate_contradiction_found": false,
  "summary": "one concise sentence",
  "source_evidence_checks_performed": [],
  "candidate_material_contradictions":
    {json.dumps(example_candidate_contradictions, indent=2)},
  "cross_candidate_conceptual_contradictions": [],
  "non_contradictory_differences": [],
  "deepseek_usage_constraints": [],
  "evidence_gaps": []
}}
```

`candidate_material_contradictions` must contain one entry for every candidate
whose staged `material_consolidated.json` has `contradiction_found: true`. Each
entry must preserve the candidate material name, summary, contradiction items, and
source report path.

Set `contradiction_found` to `true` and `verdict` to `contradictions_found`
whenever either `candidate_material_contradictions` OR
`cross_candidate_conceptual_contradictions` is non-empty. A target with preserved
candidate-material contradictions is not a `no_contradiction` result, even when
there are no new cross-candidate conceptual contradictions.

For each item in `cross_candidate_conceptual_contradictions`, include:

- `topic`;
- `status`: `resolved_by_context | conditional | unresolved`;
- `candidate_claims` with candidate material, case ID, and confidence level;
- `why_conceptually_incompatible`;
- `chemistry_or_context_caveat`;
- `source_evidence` with staged paths;
- `target_side_discriminator_checks`;
- `safe_deepseek_instruction`.

If neither candidate-material contradictions nor cross-candidate conceptual
contradictions exist:

- set `verdict` to `no_contradiction`;
- set both contradiction booleans to `false`;
- leave both contradiction arrays empty;
- keep the Markdown minimal: a warning heading and one result sentence;
- do not output high-confidence knowledge, conditional knowledge, candidate-specific
  lessons, decision checklists, generalized advice, or a cross-material summary.

If contradictions exist, the Markdown should contain only:

1. How to use this contradiction report
2. Candidate-material contradictions
3. Cross-candidate conceptual contradictions
4. Context that resolves apparent conflicts
5. Safe DeepSeek usage constraints
6. Evidence gaps

Candidate evidence remains namespaced to its source material. Never convert a
contradiction into a universal chemical rule.
"""


# ---------------------------------------------------------------------------
# Gemini runner
# ---------------------------------------------------------------------------


def run_gemini_job(
    *,
    work_dir: Path,
    prompt: str,
    gemini_bin: str,
    model: str,
    max_attempts: int,
    output_names: tuple[str, ...],
    validator: Callable[[Path], bool],
    validation_errors: Callable[[Path], list[str]] | None = None,
) -> None:
    prompt_path = work_dir / "prompt.md"
    write_text(prompt_path, prompt)
    attempts: list[dict[str, Any]] = []
    combined_log = work_dir / "gemini_stdout_stderr.txt"
    status_path = work_dir / "run_status.json"
    command = [
        gemini_bin,
        "--yolo",
        "--skip-trust",
        f"--model={model}",
        f"--prompt={prompt}",
    ]

    for attempt_index in range(1, max_attempts + 1):
        for output_name in output_names:
            path = work_dir / output_name
            if path.exists():
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        attempt_log = work_dir / f"gemini_attempt_{attempt_index:02d}_stdout_stderr.txt"
        completed = subprocess.run(
            command,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout = completed.stdout or ""
        write_text(attempt_log, stdout)
        valid = validator(work_dir)
        errors = [] if valid or validation_errors is None else validation_errors(work_dir)
        attempts.append(
            {
                "attempt": attempt_index,
                "returncode": completed.returncode,
                "attempt_log_path": attempt_log.name,
                "valid_output": valid,
                **({"validation_errors": errors} if errors else {}),
            }
        )
        write_json(
            status_path,
            {
                "model": model,
                "gemini_bin": gemini_bin,
                "prompt_path": prompt_path.name,
                "max_attempts": max_attempts,
                "success": valid,
                "attempts": attempts,
            },
        )
        if valid:
            write_text(combined_log, stdout)
            return
        print(
            f"Gemini attempt {attempt_index}/{max_attempts} failed validation in {work_dir}"
            + (f": {'; '.join(errors)}" if errors else ""),
            file=sys.stderr,
        )

    write_text(
        combined_log,
        "\n\n".join(
            f"===== ATTEMPT {item['attempt']} returncode={item['returncode']} =====\n"
            + (work_dir / item["attempt_log_path"]).read_text(
                encoding="utf-8", errors="replace"
            )
            for item in attempts
        ),
    )
    raise RuntimeError(f"Gemini did not produce valid outputs in {work_dir}")


# ---------------------------------------------------------------------------
# Per-material stage
# ---------------------------------------------------------------------------


def stage_material_inputs(
    output_dir: Path,
    material: str,
    reviews: list[RunReview],
    source_cases: dict[str, SourceCase | None],
    *,
    reviews_root: Path,
    dataset_root: Path,
    reviewer_module: ModuleType | None,
) -> list[str]:
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    missing_evidence_cases: list[str] = []
    entries: list[dict[str, Any]] = []

    for index, review in enumerate(reviews, start=1):
        case_dir = inputs_dir / f"{index:03d}__{safe_component(review.case_id)}"
        json_dst = case_dir / "self_debug_report.json"
        copy_file(review.json_path, json_dst)
        entry: dict[str, Any] = {
            "case_id": review.case_id,
            "review_json_path": str(json_dst.relative_to(output_dir)),
            "source_review_json_path": relative_display(review.json_path, reviews_root),
        }
        if review.md_path is not None:
            md_dst = case_dir / "self_debug_report.md"
            copy_file(review.md_path, md_dst)
            entry["review_md_path"] = str(md_dst.relative_to(output_dir))

        evidence = stage_review_evidence(
            case_dir,
            review,
            source_cases.get(review.case_id),
            dataset_root,
            reviewer_module,
        )
        entry["source_evidence"] = evidence
        if evidence.get("status") != "available":
            missing_evidence_cases.append(review.case_id)
        entries.append(entry)

    write_json(
        inputs_dir / "index.json",
        {
            "schema_version": CONTRADICTION_SCHEMA_VERSION,
            "stage": "per_candidate_material_contradiction_review",
            "material": material,
            "run_reviews": entries,
            "missing_source_evidence_cases": missing_evidence_cases,
        },
    )
    return missing_evidence_cases


def publish_material_aliases(output_dir: Path, material_root: Path) -> None:
    primary_md = output_dir / "material_consolidated.md"
    primary_json = output_dir / "material_consolidated.json"
    copy_file(primary_md, output_dir / "self_debug_report.md")
    copy_file(primary_json, output_dir / "self_debug_report.json")
    copy_file(primary_md, material_root / "MATERIAL_CONSOLIDATED.md")
    copy_file(primary_json, material_root / "MATERIAL_CONSOLIDATED.json")


def consolidate_one_material(
    material: str,
    *,
    reviews_root: Path,
    run_roots: list[Path],
    dataset_root: Path,
    material_output_root: Path,
    reviewer_module: ModuleType | None,
    gemini_bin: str,
    model: str,
    max_attempts: int,
    force: bool,
    allow_missing_source_evidence: bool,
) -> MaterialConsolidationResult:
    reviews = collect_run_reviews(material, reviews_root)
    if not reviews:
        raise RuntimeError(f"No usable per-run reviews for {material}")

    source_cases = {
        review.case_id: locate_source_case(review, run_roots)
        for review in reviews
    }
    manifest = material_source_manifest(
        material,
        reviews,
        source_cases,
        reviews_root=reviews_root,
        dataset_root=dataset_root,
    )
    missing_cases = [
        review.case_id
        for review in reviews
        if manifest["sources"][reviews.index(review)]["source_evidence"]["evidence_mode"] == "missing"
    ]
    if missing_cases and not allow_missing_source_evidence:
        raise RuntimeError(
            f"Original source evidence could not be located for {material}: "
            + ", ".join(missing_cases)
            + ". Pass the correct --run-root values or use --allow-missing-source-evidence."
        )

    material_root = material_output_root / material
    output_dir = material_root / MATERIAL_CASE_NAME
    fresh = (
        not force
        and output_dir.is_dir()
        and source_manifest_matches(output_dir, manifest)
        and material_output_is_valid(output_dir)
    )
    if fresh:
        publish_material_aliases(output_dir, material_root)
        return MaterialConsolidationResult(
            material=material,
            output_dir=output_dir,
            json_path=output_dir / "material_consolidated.json",
            md_path=output_dir / "material_consolidated.md",
            source_manifest=manifest,
            skipped_as_fresh=True,
        )

    clean_dir(output_dir)
    staged_missing = stage_material_inputs(
        output_dir,
        material,
        reviews,
        source_cases,
        reviews_root=reviews_root,
        dataset_root=dataset_root,
        reviewer_module=reviewer_module,
    )
    if staged_missing and not allow_missing_source_evidence:
        raise RuntimeError(
            f"Staging original evidence failed for {material}: " + ", ".join(staged_missing)
        )
    write_json(output_dir / "source_manifest.json", manifest)
    if len(reviews) < 2:
        write_single_run_material_output(output_dir, material, reviews)
    else:
        run_gemini_job(
            work_dir=output_dir,
            prompt=material_prompt(material, len(reviews)),
            gemini_bin=gemini_bin,
            model=model,
            max_attempts=max_attempts,
            output_names=("material_consolidated.md", "material_consolidated.json"),
            validator=material_output_is_valid,
        )
    publish_material_aliases(output_dir, material_root)
    return MaterialConsolidationResult(
        material=material,
        output_dir=output_dir,
        json_path=output_dir / "material_consolidated.json",
        md_path=output_dir / "material_consolidated.md",
        source_manifest=manifest,
        skipped_as_fresh=False,
    )


# ---------------------------------------------------------------------------
# Per-target stage
# ---------------------------------------------------------------------------


def stage_target_task_files(output_dir: Path, target: str, dataset_root: Path) -> list[str]:
    source_dir = dataset_root / target
    staged: list[str] = []
    for relative in TARGET_TASK_RELATIVE_PATHS:
        src = source_dir / relative
        if src.is_file():
            dst = output_dir / "target_material_files" / relative
            copy_file(src, dst)
            staged.append(relative)
    return staged


def stage_target_inputs(
    output_dir: Path,
    target: str,
    candidates_requested: list[str],
    material_results: dict[str, MaterialConsolidationResult],
    missing_candidates: list[str],
    dataset_root: Path,
) -> list[str]:
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    candidates_used: list[str] = []

    for index, candidate in enumerate(candidates_requested, start=1):
        result = material_results.get(candidate)
        if result is None:
            continue
        candidate_dir = inputs_dir / f"{index:03d}__{safe_component(candidate)}"
        copy_file(result.json_path, candidate_dir / "material_consolidated.json")
        copy_file(result.md_path, candidate_dir / "material_consolidated.md")
        copy_file(
            result.output_dir / "source_manifest.json",
            candidate_dir / "source_manifest.json",
        )
        # This gives the target-level Gemini call the original source evidence,
        # not only the candidate-level summary.
        copy_tree_hardlink(result.output_dir / "inputs", candidate_dir / "source_runs")
        candidates_used.append(candidate)
        entries.append(
            {
                "candidate_material": candidate,
                "material_consolidated_json": str(
                    (candidate_dir / "material_consolidated.json").relative_to(output_dir)
                ),
                "material_consolidated_md": str(
                    (candidate_dir / "material_consolidated.md").relative_to(output_dir)
                ),
                "source_manifest": str(
                    (candidate_dir / "source_manifest.json").relative_to(output_dir)
                ),
                "source_runs_dir": str((candidate_dir / "source_runs").relative_to(output_dir)),
            }
        )

    target_files = stage_target_task_files(output_dir, target, dataset_root)
    write_json(
        inputs_dir / "index.json",
        {
            "schema_version": CONTRADICTION_SCHEMA_VERSION,
            "stage": "per_target_cross_candidate_contradiction_review",
            "target_material": target,
            "candidate_materials_requested": candidates_requested,
            "candidate_materials_used": candidates_used,
            "missing_candidate_materials": missing_candidates,
            "candidate_consolidations": entries,
            "target_material_files": target_files,
        },
    )
    return candidates_used


def publish_target_aliases(output_dir: Path, target_root: Path) -> None:
    primary_md = output_dir / "ALL_SELF_DEBUG.md"
    primary_json = output_dir / "ALL_SELF_DEBUG.json"
    copy_file(primary_md, output_dir / "self_debug_report.md")
    copy_file(primary_json, output_dir / "self_debug_report.json")
    copy_file(primary_md, target_root / "ALL_SELF_DEBUG.md")
    copy_file(primary_json, target_root / "ALL_SELF_DEBUG.json")


def consolidate_one_target(
    target: str,
    candidates_requested: list[str],
    *,
    material_results: dict[str, MaterialConsolidationResult],
    material_output_root: Path,
    target_output_root: Path,
    dataset_root: Path,
    gemini_bin: str,
    model: str,
    max_attempts: int,
    force: bool,
    require_all_candidates: bool,
) -> TargetConsolidationResult:
    manifest = target_source_manifest(
        target,
        candidates_requested,
        material_results,
        material_output_root=material_output_root,
        dataset_root=dataset_root,
    )
    candidates_used = list(manifest["candidate_materials_used"])
    missing_candidates = list(manifest["missing_candidate_materials"])
    if not candidates_used:
        raise RuntimeError(f"Target {target} has no usable candidate material consolidations")
    if require_all_candidates and missing_candidates:
        raise RuntimeError(
            f"Target {target} is missing candidate consolidations: "
            + ", ".join(missing_candidates)
        )

    target_root = target_output_root / target
    output_dir = target_root / TARGET_CASE_NAME
    fresh = (
        not force
        and output_dir.is_dir()
        and source_manifest_matches(output_dir, manifest)
        and target_output_is_valid(output_dir)
    )
    if fresh:
        publish_target_aliases(output_dir, target_root)
        return TargetConsolidationResult(
            target=target,
            output_dir=output_dir,
            json_path=output_dir / "ALL_SELF_DEBUG.json",
            md_path=output_dir / "ALL_SELF_DEBUG.md",
            candidates_requested=tuple(candidates_requested),
            candidates_used=tuple(candidates_used),
            missing_candidates=tuple(missing_candidates),
            source_manifest=manifest,
            skipped_as_fresh=True,
        )

    clean_dir(output_dir)
    staged_used = stage_target_inputs(
        output_dir,
        target,
        candidates_requested,
        material_results,
        missing_candidates,
        dataset_root,
    )
    if staged_used != candidates_used:
        raise RuntimeError(
            f"Internal candidate staging mismatch for {target}: {staged_used} != {candidates_used}"
        )
    write_json(output_dir / "source_manifest.json", manifest)
    if len(candidates_used) < 2:
        write_single_candidate_target_output(
            output_dir,
            target,
            candidates_requested,
            candidates_used,
            missing_candidates,
            material_results,
        )
    else:
        run_gemini_job(
            work_dir=output_dir,
            prompt=target_prompt(
                target,
                candidates_requested,
                candidates_used,
                missing_candidates,
                [
                    candidate
                    for candidate in candidates_used
                    if candidate in _target_input_candidate_contradictions(output_dir)
                ],
            ),
            gemini_bin=gemini_bin,
            model=model,
            max_attempts=max_attempts,
            output_names=("ALL_SELF_DEBUG.md", "ALL_SELF_DEBUG.json"),
            validator=target_output_is_valid,
            validation_errors=target_output_validation_errors,
        )
    publish_target_aliases(output_dir, target_root)
    return TargetConsolidationResult(
        target=target,
        output_dir=output_dir,
        json_path=output_dir / "ALL_SELF_DEBUG.json",
        md_path=output_dir / "ALL_SELF_DEBUG.md",
        candidates_requested=tuple(candidates_requested),
        candidates_used=tuple(candidates_used),
        missing_candidates=tuple(missing_candidates),
        source_manifest=manifest,
        skipped_as_fresh=False,
    )


# ---------------------------------------------------------------------------
# Downstream compatibility and index
# ---------------------------------------------------------------------------


def write_compatibility_csv(path: Path, targets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_material", "candidate_material"],
        )
        writer.writeheader()
        for target in sorted(targets):
            writer.writerow({"target_material": target, "candidate_material": target})


def write_overall_index(
    path: Path,
    *,
    reviews_root: Path,
    run_roots: list[Path],
    dataset_root: Path,
    candidate_table: Path,
    material_output_root: Path,
    target_output_root: Path,
    compatibility_csv: Path,
    material_results: dict[str, MaterialConsolidationResult],
    target_results: dict[str, TargetConsolidationResult],
    material_failures: dict[str, str],
    target_failures: dict[str, str],
    model: str,
) -> None:
    write_json(
        path,
        {
            "schema_version": CONTRADICTION_SCHEMA_VERSION,
            "model": model,
            "source_reviews_root": str(reviews_root),
            "source_run_roots": [str(path) for path in run_roots],
            "dataset_root": str(dataset_root),
            "original_candidate_table": str(candidate_table),
            "material_consolidations_root": str(material_output_root),
            "target_consolidations_root": str(target_output_root),
            "deepseek_compatibility_candidate_table": str(compatibility_csv),
            "deepseek_usage": {
                "candidate_run_error_table": str(compatibility_csv),
                "candidate_self_debug_reviews_root": str(target_output_root),
            },
            "material_consolidations": {
                material: {
                    "json": str(result.json_path),
                    "markdown": str(result.md_path),
                    "skipped_as_fresh": result.skipped_as_fresh,
                    "verdict": read_json_object(result.json_path).get("verdict"),
                    "contradiction_found": read_json_object(result.json_path).get("contradiction_found"),
                }
                for material, result in sorted(material_results.items())
            },
            "target_consolidations": {
                target: {
                    "json": str(result.json_path),
                    "markdown": str(result.md_path),
                    "candidate_materials_requested": list(result.candidates_requested),
                    "candidate_materials_used": list(result.candidates_used),
                    "missing_candidate_materials": list(result.missing_candidates),
                    "skipped_as_fresh": result.skipped_as_fresh,
                    "verdict": read_json_object(result.json_path).get("verdict"),
                    "contradiction_found": read_json_object(result.json_path).get("contradiction_found"),
                    "cross_candidate_contradiction_found": read_json_object(result.json_path).get("cross_candidate_contradiction_found"),
                }
                for target, result in sorted(target_results.items())
            },
            "material_failures": material_failures,
            "target_failures": target_failures,
        },
    )


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Contradiction-only per-material and cross-candidate Gemini review for "
            "DeepSeek self-debug context."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--reviews-root", type=Path, default=None)
    parser.add_argument("--candidate-table", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Original DeepSeek jobs root. Repeat for multiple roots. If omitted, "
            "defaults to jobsDeepseekProTerminus2ControlledIter2 under --repo-root."
        ),
    )
    parser.add_argument(
        "--reviewer-script",
        type=Path,
        default=None,
        help=(
            "Path to run_gemini_reviews_on_deepseek.py. Used only to regenerate the "
            "same deterministic AMN summary for existing cases."
        ),
    )
    parser.add_argument("--material-output-root", type=Path, default=None)
    parser.add_argument("--target-output-root", type=Path, default=None)
    parser.add_argument("--compatibility-csv", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--gemini-bin", default=DEFAULT_GEMINI_BIN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-concurrent-materials",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_MATERIALS,
    )
    parser.add_argument(
        "--max-concurrent-targets",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_TARGETS,
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Only consolidate this target material. Repeat to select multiple targets.",
    )
    parser.add_argument("--require-all-candidates", action="store_true")
    parser.add_argument(
        "--allow-missing-source-evidence",
        action="store_true",
        help=(
            "Allow a per-material consolidation to proceed when an original run "
            "directory cannot be located. Default behavior is to fail, because a "
            "reviews-only consolidation is not independently evidence-backed."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.max_concurrent_materials < 1 or cli.max_concurrent_targets < 1:
        raise SystemExit("concurrency values must be at least 1")
    if cli.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")

    repo_root = cli.repo_root.expanduser().resolve()
    reviews_root = path_from_cli(cli.reviews_root, repo_root / REVIEW_ROOT_RELATIVE)
    candidate_table = path_from_cli(cli.candidate_table, repo_root / CANDIDATE_TABLE_RELATIVE)
    dataset_root = path_from_cli(cli.dataset_root, repo_root / DATASET_ROOT_RELATIVE)
    run_roots = unique_paths(
        cli.run_root
        if cli.run_root
        else [repo_root / relative for relative in DEFAULT_RUN_ROOT_RELATIVES]
    )
    reviewer_script = path_from_cli(
        cli.reviewer_script,
        repo_root / REVIEWER_SCRIPT_RELATIVE,
    )
    material_output_root = path_from_cli(
        cli.material_output_root,
        repo_root / MATERIAL_OUTPUT_ROOT_RELATIVE,
    )
    target_output_root = path_from_cli(
        cli.target_output_root,
        repo_root / TARGET_OUTPUT_ROOT_RELATIVE,
    )
    compatibility_csv = path_from_cli(
        cli.compatibility_csv,
        repo_root / COMPATIBILITY_CSV_RELATIVE,
    )
    index_path = path_from_cli(cli.index_path, repo_root / INDEX_RELATIVE)

    if shutil.which(cli.gemini_bin) is None:
        raise SystemExit(f"Could not find Gemini CLI executable {cli.gemini_bin!r}")
    if not reviews_root.is_dir():
        raise SystemExit(f"Reviews root does not exist: {reviews_root}")
    missing_roots = [path for path in run_roots if not path.is_dir()]
    if missing_roots and not cli.allow_missing_source_evidence:
        raise SystemExit(
            "Original run root(s) do not exist: "
            + ", ".join(str(path) for path in missing_roots)
        )

    reviewer_module = load_reviewer_module(reviewer_script)
    if reviewer_module is None:
        print(
            "Warning: reviewer helper could not be loaded; original text/log evidence "
            "will still be staged, but AMN summaries can only be used when retained "
            "case_files already exist.",
            file=sys.stderr,
        )

    candidates_by_target = candidate_materials_from_include_only_csv(candidate_table)
    selected_targets = {
        validate_material_name(value, field="--target") for value in cli.target
    }
    if selected_targets:
        unknown = selected_targets - set(candidates_by_target)
        if unknown:
            raise SystemExit("Unknown target(s): " + ", ".join(sorted(unknown)))
        candidates_by_target = {
            target: candidates
            for target, candidates in candidates_by_target.items()
            if target in selected_targets
        }

    unique_candidates = sorted(
        {
            candidate
            for candidates in candidates_by_target.values()
            for candidate in candidates
        }
    )
    material_output_root.mkdir(parents=True, exist_ok=True)
    target_output_root.mkdir(parents=True, exist_ok=True)

    material_results: dict[str, MaterialConsolidationResult] = {}
    material_failures: dict[str, str] = {}
    material_workers = min(cli.max_concurrent_materials, max(1, len(unique_candidates)))
    print(
        f"Same-material contradiction review: {len(unique_candidates)} unique candidate(s), "
        f"concurrency={material_workers}."
    )
    with ThreadPoolExecutor(max_workers=material_workers) as pool:
        futures = {
            pool.submit(
                consolidate_one_material,
                material,
                reviews_root=reviews_root,
                run_roots=run_roots,
                dataset_root=dataset_root,
                material_output_root=material_output_root,
                reviewer_module=reviewer_module,
                gemini_bin=cli.gemini_bin,
                model=cli.model,
                max_attempts=cli.max_attempts,
                force=cli.force,
                allow_missing_source_evidence=cli.allow_missing_source_evidence,
            ): material
            for material in unique_candidates
        }
        for future in as_completed(futures):
            material = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                material_failures[material] = message
                print(f"FAILED material consolidation {material}: {message}", file=sys.stderr)
            else:
                material_results[material] = result
                state = "fresh" if result.skipped_as_fresh else "generated"
                print(f"Material contradiction review {state}: {material}")

    target_results: dict[str, TargetConsolidationResult] = {}
    target_failures: dict[str, str] = {}
    target_items = sorted(candidates_by_target.items())
    target_workers = min(cli.max_concurrent_targets, max(1, len(target_items)))
    print(
        f"Cross-candidate contradiction review: {len(target_items)} target(s), "
        f"concurrency={target_workers}."
    )
    with ThreadPoolExecutor(max_workers=target_workers) as pool:
        futures = {
            pool.submit(
                consolidate_one_target,
                target,
                candidates,
                material_results=material_results,
                material_output_root=material_output_root,
                target_output_root=target_output_root,
                dataset_root=dataset_root,
                gemini_bin=cli.gemini_bin,
                model=cli.model,
                max_attempts=cli.max_attempts,
                force=cli.force,
                require_all_candidates=cli.require_all_candidates,
            ): target
            for target, candidates in target_items
        }
        for future in as_completed(futures):
            target = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                target_failures[target] = message
                print(f"FAILED target consolidation {target}: {message}", file=sys.stderr)
            else:
                target_results[target] = result
                state = "fresh" if result.skipped_as_fresh else "generated"
                print(f"Target contradiction review {state}: {target}")

    successful_targets = sorted(target_results)
    write_compatibility_csv(compatibility_csv, successful_targets)
    write_overall_index(
        index_path,
        reviews_root=reviews_root,
        run_roots=run_roots,
        dataset_root=dataset_root,
        candidate_table=candidate_table,
        material_output_root=material_output_root,
        target_output_root=target_output_root,
        compatibility_csv=compatibility_csv,
        material_results=material_results,
        target_results=target_results,
        material_failures=material_failures,
        target_failures=target_failures,
        model=cli.model,
    )

    print(f"Wrote compatibility CSV: {compatibility_csv}")
    print(f"Wrote consolidation index: {index_path}")
    if material_failures or target_failures:
        raise SystemExit(
            f"Consolidation completed with {len(material_failures)} material failure(s) "
            f"and {len(target_failures)} target failure(s)."
        )


if __name__ == "__main__":
    main()
