import csv
import json
import math
from collections import defaultdict
from pathlib import Path

# Assumes you run this script from the WannierisationBenchmarking folder
PROJECT_ROOT = Path.cwd()

JOBS_DIR = PROJECT_ROOT / "jobs"

INPUT_CSV = JOBS_DIR / "successful_run_errors.csv"
CASE1_OUTPUT_JSON = JOBS_DIR / "case1_materials.json"
CASE2_OUTPUT_JSON = JOBS_DIR / "case2_materials.json"
COMBINED_OUTPUT_JSON = JOBS_DIR / "filtered_materials.json"

RATIO_COLUMN = "gemini_to_reference_ratio"
MATERIAL_COLUMN = "material"
NUM_WANN_COLUMN = "num_wann"
RUN_ID_COLUMN = "run_id"


def parse_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def load_rows(csv_path):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {MATERIAL_COLUMN, NUM_WANN_COLUMN, RUN_ID_COLUMN, RATIO_COLUMN}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')

        for row in reader:
            ratio = parse_float(row.get(RATIO_COLUMN))
            if ratio is None:
                continue
            row['_ratio'] = ratio
            rows.append(row)
    return rows


def main():
    rows = load_rows(INPUT_CSV)

    rows_by_material = defaultdict(list)
    rows_by_material_num_wann = defaultdict(list)

    for row in rows:
        material = row[MATERIAL_COLUMN]
        num_wann = row[NUM_WANN_COLUMN]
        rows_by_material[material].append(row)
        rows_by_material_num_wann[(material, num_wann)].append(row)

    # Case 1:
    # Keep a material if at least one material+num_wann group has a run with ratio < 3
    # and another run in that same material+num_wann group has ratio > 10.
    case1_materials = set()
    for (material, _num_wann), group_rows in rows_by_material_num_wann.items():
        has_below_3 = any(row['_ratio'] < 3 for row in group_rows)
        has_above_10 = any(row['_ratio'] > 10 for row in group_rows)
        if has_below_3 and has_above_10:
            case1_materials.add(material)

    case1 = {}
    for material in sorted(case1_materials):
        material_rows = rows_by_material[material]
        min_row = min(material_rows, key=lambda row: row['_ratio'])
        max_row = max(material_rows, key=lambda row: row['_ratio'])
        case1[material] = [
            min_row['_ratio'],
            max_row['_ratio'],
            min_row[RUN_ID_COLUMN],
            max_row[RUN_ID_COLUMN],
        ]

    # Case 2:
    # Keep materials where no run has ratio below 5, i.e. every valid run has ratio >= 5.
    case2 = sorted(
        material
        for material, material_rows in rows_by_material.items()
        if all(row['_ratio'] >= 5 for row in material_rows)
    )

    combined = {
        'case1_materials': case1,
        'case2_materials': case2,
    }

    CASE1_OUTPUT_JSON.write_text(json.dumps(case1, indent=2), encoding='utf-8')
    CASE2_OUTPUT_JSON.write_text(json.dumps(case2, indent=2), encoding='utf-8')
    COMBINED_OUTPUT_JSON.write_text(json.dumps(combined, indent=2), encoding='utf-8')

    print(f'Wrote {len(case1)} case 1 materials to {CASE1_OUTPUT_JSON}')
    print(f'Wrote {len(case2)} case 2 materials to {CASE2_OUTPUT_JSON}')
    print(f'Wrote combined output to {COMBINED_OUTPUT_JSON}')


if __name__ == '__main__':
    main()

