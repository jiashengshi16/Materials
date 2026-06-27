# Ag2Y Wannierisation Task

You are given a completed DFT package in `material/` for metallic `Ag2Y`.

Build and, if feasible within 2 hours, run a Wannierisation workflow. Use only
the supplied structure, SCF/NSCF inputs, DFT outputs, and metadata as the
scientific starting point.

## Workflow provenance rules

This is a Wannierisation-strategy benchmark. Do not assume any pre-existing
Wannier90 recipe is supplied. In particular, do not use `/search` or any other
external lookup for reference `.win`, `.amn`, `.mmn`, `.eig`, `.nnkp`, `.wout`,
`.chk`, or `_hr.dat` files outside the copied `material/` package. If such
files are present inside `material/`, treat them as accidental leakage, do not
use them, and record this in `REPORT.md`.

If `material/` or a copied `materials/` package contains a QE NSCF save tree,
use that save tree as the DFT-side input for Wannierisation. Do not rerun SCF
or NSCF unless the provided save tree is missing or unusable. If you rerun any
DFT-side step, record the reason explicitly in `run_manifest.json`,
`report.json`, and `REPORT.md`.

## Decision rationale requirement

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

PROJECTION CHOICES MUST NOT BE RANDOM. USE YOUR OWN EXPLICIT PROJECTIONS. Do not use `random`, randomized trial projections, or arbitrary placeholder projections
as a fallback. If the ideal projection set is uncertain infer the best
available projections from evidence such as composition, valence electron count, expected orbital character,
pseudopotential metadata, QE logs/XML, band energies, and the target band
manifold.

At the end, copy the decision rationales into `REPORT.md` and summarize the key
rationales in `report.json.runtime_notes` and
`artifacts/attempt_<N>/run_manifest.json.notes`.

If `material/pseudo/` contains UPF files, those are the pseudopotentials that
match the supplied QE inputs. Prefer rerunning SCF/NSCF with
`pseudo_dir = './pseudo/'` from a workflow run directory that symlinks or copies
`material/pseudo/`, especially when the supplied QE 6.3 XML cannot be consumed
directly by the installed `pw2wannier90.x`.

## Recommended QE-save workflow

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
include large binary/checkpoint listings or long radial numeric tables.

For this benchmark, use a final model size of `num_wann = 33` Wannier
functions. The supplied NSCF calculation has `num_bands = 73` bands;
use those bands as the available disentanglement pool.

Target DFT bands `1-33` exactly, using 1-based DFT band indices. Do not
exclude the low-energy bands from this target. Choose projections and
disentanglement/frozen windows that faithfully interpolate this target manifold.

Record the target DFT bands explicitly as `target_dft_band_start = 1` and
`target_dft_band_end = 33` in `run_manifest.json`, `report.json`, and
`REPORT.md`.

Keep `material/` unchanged. Write:

- `workflow/`: runnable scripts/configuration
- `artifacts/attempt_<N>/`: final Wannier artifacts when produced
- `REPORT.md`: short explanation of choices and results
- `report.json`: machine-readable summary

For each attempt, create `artifacts/attempt_<N>/run_manifest.json`. Record the
chosen projections, windows, `num_bands`, `num_wann`, target band information,
commands run, produced files, and missing files.

Use seedname `Ag2Y` unless there is a clear reason not to. Your final
response must match the provided JSON schema.

## Harbor grading note

The hidden verifier compares your final Wannier Hamiltonian against withheld
DFT reference bands. The best/final attempt must include `<seed>_hr.dat` in
`artifacts/attempt_<N>/`; without that file the run receives reward 0 because
there is no Hamiltonian to evaluate.

Before returning a final response, verify that the final artifact exists:

```bash
test -s artifacts/attempt_1/run_manifest.json
test -s artifacts/attempt_1/${seed}_hr.dat
ls -lh artifacts/attempt_1/*_hr.dat
```

Do not report `status: "success"` or `executed_successfully: true` unless
`artifacts/attempt_1/<seed>_hr.dat` exists and is non-empty. Do not run the
final workflow in the background with `nohup` or `&` and then return before it
finishes. When collecting artifacts, do not rely on `cp <seed>.* ...` because
that does not copy `<seed>_hr.dat`; copy `<seed>_hr.dat` explicitly or use
`cp <seed>* ...`.

For example, after a successful Wannier90 run:

```bash
mkdir -p artifacts/attempt_1
cp workflow/run_dir/${seed}_hr.dat artifacts/attempt_1/
cp workflow/run_dir/${seed}.win workflow/run_dir/${seed}.wout workflow/run_dir/${seed}.eig workflow/run_dir/${seed}.chk workflow/run_dir/${seed}.nnkp artifacts/attempt_1/ 2>/dev/null || true
```

If `<seed>_hr.dat` is missing or empty after your attempts, return
`status: "partial"` or `status: "failed"`, not `status: "success"`.

## Required JSON contracts

The best/final attempt's `artifacts/attempt_<N>/run_manifest.json` must be
valid JSON and must include these top-level fields. If the workflow completed
and produced the final Wannier Hamiltonian, set `status` to `"success"` and
`executed_successfully` to `true`.

```json
{
  "material_id": "<material>",
  "seedname": "<seed>",
  "attempt": "attempt_1",
  "status": "success | partial | failed",
  "executed_successfully": true,
  "workflow_entrypoint": "workflow/run.sh",
  "target_dft_band_start": 1,
  "target_dft_band_end": null,
  "num_wann": null,
  "num_bands": null,
  "projections": [],
  "wannier_parameters": {
    "num_bands": null,
    "num_wann": null,
    "dis_win_min": null,
    "dis_win_max": null,
    "dis_froz_min": null,
    "dis_froz_max": null
  },
  "files": {
    "win": "<seed>.win",
    "wout": "<seed>.wout",
    "eig": "<seed>.eig",
    "hr": "<seed>_hr.dat",
    "chk": "<seed>.chk",
    "nnkp": "<seed>.nnkp"
  },
  "commands": [],
  "produced_files": [],
  "missing_files": [],
  "notes": []
}
```

The root `report.json` must also include at least:

```json
{
  "material_id": "<material>",
  "status": "success | partial | failed",
  "executed_successfully": true,
  "attempt_count": 1,
  "runtime_notes": [],
  "key_outputs": [],
  "key_workflow_files": [],
  "blockers": [],
  "run_manifest_path": "artifacts/attempt_1/run_manifest.json",
  "artifact_root": "artifacts/attempt_1",
  "target_dft_band_start": 1,
  "target_dft_band_end": null,
  "num_wann": null,
  "num_bands": null
}
```

Your final response must be a concise JSON object with the same top-level
status/path fields as `report.json`. Do not omit `status` or
`executed_successfully`.
