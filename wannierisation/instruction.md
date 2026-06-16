# W Wannierisation Task

You are given a completed DFT package in `material/` for metallic `W`.

Build and, if feasible within 2 hours, run a Wannierisation workflow. Use only
the supplied structure, SCF/NSCF inputs, DFT outputs, and metadata as the
scientific starting point.

For this benchmark, use a final model size of `num_wann = 13` Wannier
functions. The supplied NSCF calculation has `num_bands = 21` bands;
use those bands as the available disentanglement pool.

Target DFT bands `1-13` exactly, using 1-based DFT band indices. Do not
exclude the low-energy bands from this target. Choose projections and
disentanglement/frozen windows that faithfully interpolate this target manifold.

Record the target DFT bands explicitly as `target_dft_band_start = 1` and
`target_dft_band_end = 13` in `run_manifest.json`, `report.json`, and
`REPORT.md`.

Keep `material/` unchanged. Write:

- `workflow/`: runnable scripts/configuration
- `artifacts/attempt_<N>/`: final Wannier artifacts when produced
- `REPORT.md`: short explanation of choices and results
- `report.json`: machine-readable summary

For each attempt, create `artifacts/attempt_<N>/run_manifest.json`. Record the
chosen projections, windows, `num_bands`, `num_wann`, target band information,
commands run, produced files, and missing files.

Use seedname `W` unless there is a clear reason not to. Your final
response must match the provided JSON schema.

## Harbor grading note

The hidden verifier compares your final Wannier Hamiltonian against withheld
DFT reference bands. The best/final attempt must include `<seed>_hr.dat` in
`artifacts/attempt_<N>/`; without that file the run receives reward 0 because
there is no Hamiltonian to evaluate.

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
