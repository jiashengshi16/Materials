#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import generate_harbor_num_wann_order_command as harbor_generator
# MATERIALS = [
#     'Se2Sn',
#     'O2Sr',
#     'Te2Zr',
#     'AgY',
#     'Al3Ta',
#     'Al2Os',
#     'Si6Ta3'
# ]
# MATERIALS = [
#     'Zn2',
#     'Se4Tl4',
#     'BrLi',
#     'C2Cd2O6',
#     'Ar2',
#     'He',
#     'Kr2',
#     'Mo4S6',
#     'O2Pd2',
#     'Al8Zr4'
# ]
MATERIALS = [
    'Al18Co4',
    'Al4Mn2O8',
    'Al4O8Zn2',
    'Al4Sc2',
    'Al8Zr4',
    'Ar2',
    'Au2Sc',
    'B2Cr',
    'B2Hf',
    'Bi4Cl12',
    'C2Cd2O6',
    'C2Cu2O6',
    'C4O12Sr4',
    'Cl2V',
    'Cl4Li4O16',
    'Co4Sc8',
    'Cr2F4',
    'Cr6Si2',
    'H8O16W4',
    'He',
    'Hf10Si6',
    'Hg3O3',
    'Mg2O10Ti4',
    'N2Na2O6',
    'NNb',
    'Ne',
    'Ni4Zr4',
    'O2Pd2',
    'Pd4S8',
    'RuTi',
    'Ag2Hf2',
    'Ag2Y',
    'AgMg',
    'AgSc',
    'Al2Mo6',
    'As2Ni2',
    'Au2Hf2',
    'Au2Nb',
    'Au2Pb4',
    'Au4V2',
    'B2Ta',
    'B2Y',
    'BiNa',
    'Br2V',
    'Br4Ca2',
    'BrNa',
    'C3Ag3N3',
    'CBe2',
    'CTi',
    'Ca4Cr4O12',
    'Cl6Rh2',
    'CoSi2',
    'Cr6Ga2',
    'Cu3Pt',
    'Cu4Sb2',
    'F4H2K2',
    'F4Sn',
    'F8V2',
    'FNa',
    'GaRu',
    'HK',
    'Hf2I6',
    'HfIr',
    'HfOs',
    'HfRh',
    'INa',
    'InPd',
    'K2O2',
    'K4O6S2',
    'Na2O',
    'Na2Se',
    'NiPt',
    'O2Rb',
    'OsTi',
    'Pd2Y2',
    'PdSc',
    'Pt3Y',
    'Pt4Y4',
    'S2Ta',
    'SeSr',
    'W',
    'Zn2',
]

MODEL = "google/gemini-3.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate num_wann-ordered Harbor runs for a small material list "
            "using Gemini 3.5 Flash."
        )
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
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=harbor_generator.ROOT / "jobsGeminiFlash35",
        help="Directory in which Harbor stores these test jobs.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    materials = [name.strip() for name in MATERIALS if name.strip()]

    if not materials:
        raise SystemExit("MATERIALS is empty. Add at least one directory name.")
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

    args = argparse.Namespace(
        dataset=cli.dataset,
        agent="gemini-cli",
        model=MODEL,
        n_concurrent=1,
        batch_size=cli.batch_size,
        stop_on_error=cli.stop_on_error,
        docker_prune_after_batch=False,
        docker_prune_after_material=False,
        delete_after_run=True,
        extra_arg=[],
        artifact=[],
        no_default_artifacts=False,
        save_generated_qe_save=False,
        jobs_root=cli.jobs_root,

        # Require two successful runs for every selected material.
        target_success_runs=2,
        validate_new_success=True,
        max_attempts_per_needed_success=0,
        delete_failed_attempt_folders=False,
        success_wave_timeout_sec=5400,
        success_wave_kill_after_sec=30,

        # Count successes stored in this test's jobs directory.
        success_roots=[cli.jobs_root],
        include_result_dir_name=[],
        least_success_first=True,

        no_gemini_cached_defaults=False,
        gemini_ipv4_first=True,
        no_gemini_run_timeout=False,
        no_gemini_host_network=False,
    )

    print(': "${GEMINI_API_KEY:?Export GEMINI_API_KEY before running}"')
    harbor_generator.print_target_success_loop(args, tasks)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)