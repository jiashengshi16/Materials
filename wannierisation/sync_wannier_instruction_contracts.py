#!/usr/bin/env python3
"""Sync shared benchmark contract text across Wannier instructions.

Dry-run by default. Use --apply to modify files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIRS = (
    Path("harbor_datasets/wannier_200"),
    Path("harbor_datasets/wannier_200__needs_eval_with_qe_save"),
)
DEFAULT_SOURCE = Path(
    "harbor_datasets/wannier_200/Al24W6/instruction.md"
)
SYNC_HEADING = "## Harbor grading note"
EARLY_HEADING = "## Workflow provenance rules"
DECISION_HEADING = "## Decision rationale requirement"
RECOMMENDED_HEADING = "## Recommended QE-save workflow"
STARTING_POINT_MARKER = "scientific starting point."
BENCHMARK_MARKER = "\n\nFor this benchmark,"
LEGACY_PSEUDO_NOTE = """If `material/pseudo/` contains UPF files, those are the pseudopotentials that
match the supplied QE inputs. Do not rerun SCF/NSCF when a valid QE NSCF save tree is available. Rerun DFT
only if the save-tree workflow is technically impossible, and record the exact
error that made it impossible."""

EARLY_CONTRACT = """## Workflow provenance rules

This is a Wannierisation-strategy benchmark. Do not assume any pre-existing
Wannier90 recipe is supplied. In particular, do not use `/search` or any other
external lookup for reference `.win`, `.amn`, `.mmn`, `.eig`, `.nnkp`, `.wout`,
`.chk`, or `_hr.dat` files outside the copied `material/` package. If such
files are present inside `material/`, treat them as accidental leakage, do not
use them, and record this in `REPORT.md`.

If `material/` or a copied `materials/` package contains a QE NSCF save tree,
use that save tree as the DFT-side input for Wannierisation. Do not rerun SCF
or NSCF unless the provided save tree is missing or unusable. If you rerun any
DFT-side step, record the reason explicitly in `run_manifest.json` and `REPORT.md`."""

DECISION_RATIONALE_CONTRACT = """## Decision rationale requirement

For every Wannierisation decision, explain clearly **why** the choice
was made. This includes, at minimum: projections, `num_wann`, `num_bands`,
target-band handling, disentanglement windows, frozen windows, band exclusions
or non-exclusions, whether to rerun any DFT-side step, how to respond to failed
commands, whether to make another attempt, and how to set final artifact status.

Maintain `workflow/DECISIONS.md` throughout the run. Each entry must include:
- the decision made;
- the evidence used, such as electron count, orbital character, QE logs/XML,
  band index range, energy range, command output, or missing files;
- why that evidence supports the decision;
- the expected effect on the Wannierisation.

Do not merely list parameter values. Explain the rationale for choosing them.
If a choice is uncertain or heuristic, state the uncertainty and why it is still
a reasonable choice.

PROJECTION CHOICES MUST NOT BE RANDOM. Do not use `random`, randomized trial projections, or arbitrary placeholder projections
as a fallback.

At the end, copy the decision rationales into `REPORT.md` and summarize the key
rationales in `artifacts/attempt_<N>/run_manifest.json.notes`."""

PSEUDO_NOTE = LEGACY_PSEUDO_NOTE

NO_MANUAL_WANNIER_REPAIR_CONTRACT = """## Generated-file provenance rules

Never hand-edit, pad, truncate, reorder, synthesize, or fabricate `.amn`,
`.mmn`, `.eig`, `.nnkp`, `.chk`, `_hr.dat`, or files inside a QE `.save` tree.

If Wannier90 reports a mismatch among `.win`, `.eig`, `.amn`, `.mmn`, or
`.nnkp`, regenerate the derived files from a consistent `.win` and the QE save
tree. The only workflow files you may author manually are scripts, `.win`,
`.pw2wan`, reports, and manifests."""

MATERIAL_READ_ONLY_CONTRACT = """## Material package is read-only

Treat `material/` as immutable input. Never delete, overwrite, patch, download
into, chmod, chown, or regenerate files under `material/`. If a file is needed,
copy it into `workflow/run_dir/` first and modify only files under `workflow/`
or `artifacts/`.

Do not write pseudopotentials, QE XML, save-tree files, Wannier files, or helper
outputs into `material/`. A run that modifies `material/` is invalid."""

RECOMMENDED_QE_SAVE_WORKFLOW = """## Recommended QE-save workflow

First normalize the QE save tree. Check these paths, in order:

- `material/qe_save/out/aiida.save`
- `material/qe_save/aiida.save`
- `material/out/aiida.save`
- `material/aiida.save`

Create `workflow/run_dir/out/aiida.save` from the first valid one.

When `material/qe_save/out/aiida.save` exists, use it directly as the DFT-side
input to `pw2wannier90.x`; do not inspect or print the wavefunction files.
Do not spend the run doing open-ended analysis before creating files. Start by
creating `workflow/run_dir`, writing a first `<seed>.win`, and running the
QE-save workflow below. Keep any pre-work to a few compact metadata/log
queries.

The usual workflow is:

```bash
seed="<task seedname>"
mkdir -p workflow/run_dir
cp -a material/qe_save/out workflow/run_dir/out
cp material/pseudo/*.UPF workflow/run_dir/ 2>/dev/null || true
cd workflow/run_dir

# Write ${seed}.win with the task's required num_wann/num_bands, projections,
# kpoint_path/bands_plot settings if desired, and disentanglement windows.
wannier90.x -pp "${seed}"

cat > "${seed}.pw2wan" <<EOF
&inputpp
  outdir = './out'
  prefix = 'aiida'
  seedname = '${seed}'
  write_mmn = .true.
  write_amn = .true.
  write_eig = .true.
/
EOF
pw2wannier90.x -in "${seed}.pw2wan"
wannier90.x "${seed}"
```

Only rerun SCF/NSCF if this save-tree workflow fails for a recorded technical
reason. Use compact scripts to summarize QE logs/XML when choosing projections
or windows; keep the full save tree as file input rather than chat output.

Keep terminal output compact. Do not print full QE inputs, XML files, logs, or
K-point lists into the chat transcript with commands like `cat` on
`material/nscf/input/nscf.in` or `material/qe_save/logs/nscf.out`. Extract only
the needed lines with `grep`, `head`, `tail`, `sed -n`, or small scripts, and
redirect generated full K-point lists or workflow files directly to files.
Also avoid full listings of `material/qe_save/out/aiida.save` and avoid
`grep -A`/`grep -B` on UPF wavefunction or `PP_PSWFC` blocks, because those
include large binary/checkpoint listings or long radial numeric tables."""

SHARED_EARLY_BLOCK = "\n\n".join(
    [
        EARLY_CONTRACT,
        MATERIAL_READ_ONLY_CONTRACT,
        NO_MANUAL_WANNIER_REPAIR_CONTRACT,
        DECISION_RATIONALE_CONTRACT,
        PSEUDO_NOTE,
        RECOMMENDED_QE_SAVE_WORKFLOW,
    ]
)


def extract_synced_tail(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    start = text.find(SYNC_HEADING)
    if start == -1:
        raise SystemExit(f"missing {SYNC_HEADING!r} in {source}")
    return text[start:].strip() + "\n"


def instruction_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("*/instruction.md"))


def unique_instruction_files(dataset_dirs: list[Path]) -> list[Path]:
    """Return instruction files across dataset views, de-duping symlink targets."""
    paths_by_real_path: dict[Path, Path] = {}
    for dataset_dir in dataset_dirs:
        for path in instruction_files(dataset_dir):
            paths_by_real_path.setdefault(path.resolve(), path)
    return sorted(paths_by_real_path.values())


def resolve_repo_relative(path: Path) -> Path:
    """Resolve relative paths from cwd first, then from the repository root."""
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def remove_block_from_heading(text: str, heading: str, end_markers: tuple[str, ...]) -> str:
    """Remove a markdown block beginning at heading and ending at first marker."""
    start = text.find(heading)
    if start == -1:
        return text

    marker_positions = [pos for marker in end_markers if (pos := text.find(marker, start)) != -1]
    if not marker_positions:
        return text[:start].rstrip() + "\n"

    end = min(marker_positions)
    return (text[:start].rstrip() + text[end:]).rstrip() + "\n"


def remove_known_shared_blocks(text: str) -> str:
    """Strip old shared blocks so the canonical versions can be reinserted.

    This handles both layouts seen in the dataset:
    - the generator layout, where pseudo/recommended text appears before Harbor;
    - the synced layout, where provenance/recommended text appears before the
      material-specific "For this benchmark" block.
    """
    text = remove_block_from_heading(
        text,
        EARLY_HEADING,
        (DECISION_HEADING, RECOMMENDED_HEADING, BENCHMARK_MARKER, SYNC_HEADING),
    )
    text = remove_block_from_heading(
        text,
        DECISION_HEADING,
        (RECOMMENDED_HEADING, BENCHMARK_MARKER, SYNC_HEADING),
    )
    while LEGACY_PSEUDO_NOTE in text:
        text = text.replace(LEGACY_PSEUDO_NOTE, "").replace("\n\n\n", "\n\n")
    text = remove_block_from_heading(
        text,
        RECOMMENDED_HEADING,
        (BENCHMARK_MARKER, SYNC_HEADING),
    )
    return text.replace("\n\n\n", "\n\n")

def normalize_material_preamble(text: str) -> str:
    return text.replace(
        "Build and, if feasible within 2 hours, run a Wannierisation workflow.",
        "Build and run a Wannierisation workflow.",
    )

def sync_shared_contract_text(text: str) -> str:
    text = normalize_material_preamble(text)
    text = remove_known_shared_blocks(text)

    insert_at = text.find(STARTING_POINT_MARKER)
    if insert_at == -1:
        raise ValueError(f"missing {STARTING_POINT_MARKER!r}")

    insert_at += len(STARTING_POINT_MARKER)
    return (
        text[:insert_at].rstrip()
        + "\n\n"
        + SHARED_EARLY_BLOCK
        + "\n\n"
        + text[insert_at:].lstrip()
    )


# Backward-compatible name for older callers/tests.
sync_early_contract_text = sync_shared_contract_text


def sync_instruction(path: Path, synced_tail: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text = sync_shared_contract_text(text)
    start = new_text.find(SYNC_HEADING)
    if start == -1:
        suffix = "\n" if new_text.endswith("\n") else "\n\n"
        new_text = new_text + suffix + synced_tail
    else:
        new_text = new_text[:start].rstrip() + "\n\n" + synced_tail

    if new_text == text:
        return False

    path.write_text(new_text, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync shared provenance, decision-rationale, QE-save workflow, "
            "Harbor grading note, and Required JSON contract blocks to "
            "wannier_200 material instructions and linked dataset views."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=None,
        type=Path,
        help=(
            "Dataset directory containing per-material instruction.md files. "
            "Can be passed more than once. Defaults to syncing both "
            "harbor_datasets/wannier_200 and "
            "harbor_datasets/wannier_200__needs_eval_with_qe_save."
        ),
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        type=Path,
        help=(
            "Instruction file to copy the synced Harbor/JSON tail from. "
            "Defaults to the Al24W6 instruction in the base wannier_200 "
            "dataset."
        ),
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        help="Limit to one material name. Can be passed more than once.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this, only prints what would change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_repo_relative(args.source)
    synced_tail = extract_synced_tail(source)
    wanted = set(args.material)
    dataset_dirs = [
        resolve_repo_relative(dataset_dir)
        for dataset_dir in (args.dataset_dir or list(DEFAULT_DATASET_DIRS))
    ]
    paths = unique_instruction_files(dataset_dirs)
    if wanted:
        paths = [path for path in paths if path.parent.name in wanted]

    changed: list[Path] = []
    unchanged: list[Path] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            new_text = sync_shared_contract_text(text)
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        start = new_text.find(SYNC_HEADING)
        if start == -1:
            candidate = new_text.rstrip() + "\n\n" + synced_tail
        else:
            candidate = new_text[:start].rstrip() + "\n\n" + synced_tail

        if candidate == text:
            unchanged.append(path)
        else:
            changed.append(path)

    action = "updated" if args.apply else "would update"
    for path in changed:
        if args.apply:
            sync_instruction(path, synced_tail)
        print(f"{action}: {path}")

    print(
        f"summary: {len(unchanged)} already synced, "
        f"{len(changed)} {'updated' if args.apply else 'would change'}"
    )


if __name__ == "__main__":
    main()
