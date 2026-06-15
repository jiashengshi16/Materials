#!/usr/bin/env python3
"""Extract exact conductor packages from the Materials Cloud AiiDA archive."""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "source_downloads" / "materials_cloud_cache"
DEFAULT_ARCHIVE = DEFAULT_CACHE_DIR / "automatic_wannier_provenance.aiida"
DEFAULT_DATA = DEFAULT_CACHE_DIR / "automatic_wannier_data.json"
DEFAULT_DISCOVER_DATA = DEFAULT_CACHE_DIR / "automated_wannier_discover_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "staging" / "aiida_conductor_exact"

GROUP_LABEL = "AutoWannier-with_conduction-SCDM+MLWF-0.4"
MC_RECORD = "0ctng-gre46"
MC_RECORD_URL = f"https://archive.materialscloud.org/record/{MC_RECORD}"

TARGETS = {
    "W": "W",
    "AlCo": "AlCo",
    "FeTi": "TiFe",
    "RuTi": "TiRu",
    "HfOs": "HfOs",
}


def formula_composition(formula: str) -> tuple[tuple[str, int], ...]:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not parts:
        raise ValueError(f"Could not parse formula {formula!r}")
    counts = [(symbol, int(count or "1")) for symbol, count in parts]
    divisor = 0
    for _symbol, count in counts:
        divisor = math.gcd(divisor, count)
    divisor = max(divisor, 1)
    return tuple(sorted((symbol, count // divisor) for symbol, count in counts))


def resolve_target_map(
    requested: list[str] | None,
    final_by_formula: dict[str, str],
    discover: dict[str, Any] | None = None,
    final_formula_by_structure_uuid: dict[str, str] | None = None,
) -> dict[str, str]:
    if not requested:
        return dict(TARGETS)

    formulas_by_composition: dict[tuple[tuple[str, int], ...], list[str]] = defaultdict(list)
    for formula in final_by_formula:
        formulas_by_composition[formula_composition(formula)].append(formula)

    target_map: dict[str, str] = {}
    for item in requested:
        if "=" in item:
            material_id, archive_formula = item.split("=", 1)
            if archive_formula not in final_by_formula:
                raise ValueError(f"{item}: archive formula {archive_formula!r} not found")
            target_map[material_id] = archive_formula
            continue

        if item in final_by_formula:
            target_map[item] = item
            continue

        if discover is not None and final_formula_by_structure_uuid is not None:
            structure_uuid = discover.get(item, {}).get("structure_uuid")
            if structure_uuid in final_formula_by_structure_uuid:
                target_map[item] = final_formula_by_structure_uuid[structure_uuid]
                continue

        matches = formulas_by_composition.get(formula_composition(item), [])
        if not matches:
            raise ValueError(f"{item}: no matching archive formula found")
        if len(matches) > 1:
            raise ValueError(f"{item}: ambiguous archive formula matches {matches}")
        target_map[item] = matches[0]
    return target_map


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def uuid_path(uuid: str) -> str:
    return f"nodes/{uuid[:2]}/{uuid[2:4]}/{uuid[4:]}/path"


def input_link(incoming: dict[str, list[dict[str, str]]], uuid: str, label: str) -> str:
    for link in incoming[uuid]:
        if link["label"] == label and link["type"].startswith("input"):
            return link["input"]
    raise KeyError(f"{uuid}: missing input link {label!r}")


def output_link(outgoing: dict[str, list[dict[str, str]]], uuid: str, label: str) -> str:
    for link in outgoing[uuid]:
        if link["label"] == label:
            return link["output"]
    raise KeyError(f"{uuid}: missing output link {label!r}")


def creator_of_remote(incoming: dict[str, list[dict[str, str]]], remote_uuid: str) -> str:
    for link in incoming[remote_uuid]:
        if link["label"] == "remote_folder" and link["type"] == "create":
            return link["input"]
    raise KeyError(f"{remote_uuid}: missing creator calc for remote folder")


def attrs_for(
    uuid_to_pk: dict[str, str], attrs: dict[str, dict[str, Any]], uuid: str
) -> dict[str, Any]:
    return attrs.get(uuid_to_pk[uuid], {})


def dict_attrs(
    uuid_to_pk: dict[str, str], attrs: dict[str, dict[str, Any]], uuid: str
) -> dict[str, Any]:
    data = attrs_for(uuid_to_pk, attrs, uuid)
    return data.get("dict") or data.get("dictionary") or data


def formula_from_structure(structure: dict[str, Any]) -> str:
    names = [
        site.get("kind_name") or site.get("name")
        for site in structure.get("sites", [])
        if site.get("kind_name") or site.get("name")
    ]
    counts = Counter(names)
    order: list[str] = []
    for name in names:
        if name not in order:
            order.append(name)
    divisor = 0
    for count in counts.values():
        divisor = math.gcd(divisor, count)
    divisor = max(divisor, 1)
    return "".join(
        name if counts[name] // divisor == 1 else f"{name}{counts[name] // divisor}"
        for name in order
    )


def species_from_structure(structure: dict[str, Any]) -> list[str]:
    species: list[str] = []
    for kind in structure.get("kinds", []):
        name = kind["name"]
        if name not in species:
            species.append(name)
    return species


def fmt_float(value: float) -> str:
    return f"{value:.10f}"


def namelist_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, float):
        return f"{value:.10e}".replace("e", "d")
    return str(value)


def sorted_namelist_keys(values: dict[str, Any]) -> list[str]:
    preferred = [
        "calculation",
        "restart_mode",
        "max_seconds",
        "outdir",
        "prefix",
        "pseudo_dir",
        "verbosity",
        "wf_collect",
        "tstress",
        "ibrav",
        "nat",
        "ntyp",
        "nbnd",
        "ecutwfc",
        "ecutrho",
        "occupations",
        "smearing",
        "degauss",
        "nosym",
        "noinv",
        "diagonalization",
        "diago_full_acc",
        "conv_thr",
    ]
    seen = set()
    ordered = []
    for key in preferred:
        if key in values:
            ordered.append(key)
            seen.add(key)
    ordered.extend(sorted(key for key in values if key not in seen))
    return ordered


def format_namelist(name: str, values: dict[str, Any]) -> list[str]:
    lines = [f"&{name}"]
    for key in sorted_namelist_keys(values):
        lines.append(f"  {key} = {namelist_value(values[key])}")
    lines.append("/")
    return lines


def qe_parameters(
    params: dict[str, Any], structure: dict[str, Any], include_defaults: bool = True
) -> dict[str, dict[str, Any]]:
    result = {section: dict(values) for section, values in params.items()}
    result.setdefault("CONTROL", {})
    result.setdefault("SYSTEM", {})
    result.setdefault("ELECTRONS", {})
    if include_defaults:
        result["CONTROL"].setdefault("outdir", "./out/")
        result["CONTROL"].setdefault("prefix", "aiida")
        result["CONTROL"].setdefault("pseudo_dir", "./pseudo/")
        result["CONTROL"].setdefault("verbosity", "high")
    result["SYSTEM"].setdefault("ibrav", 0)
    result["SYSTEM"]["nat"] = len(structure.get("sites", []))
    result["SYSTEM"]["ntyp"] = len(structure.get("kinds", []))
    return result


def pseudo_inputs(
    incoming: dict[str, list[dict[str, str]]],
    uuid_to_pk: dict[str, str],
    nodes: dict[str, dict[str, Any]],
    attrs: dict[str, dict[str, Any]],
    calc_uuid: str,
) -> dict[str, dict[str, Any]]:
    pseudos = {}
    for link in incoming[calc_uuid]:
        if not link["label"].startswith("pseudo_"):
            continue
        symbol = link["label"].removeprefix("pseudo_")
        pseudos[symbol] = attrs_for(uuid_to_pk, attrs, link["input"])
    return pseudos


def format_qe_input(
    params: dict[str, Any],
    structure: dict[str, Any],
    kpoints: dict[str, Any],
    pseudos: dict[str, dict[str, Any]],
    force_kpoints_list: bool,
) -> str:
    lines: list[str] = []
    qe_params = qe_parameters(params, structure)
    for section in ("CONTROL", "SYSTEM", "ELECTRONS", "IONS", "CELL"):
        if section in qe_params and qe_params[section]:
            lines.extend(format_namelist(section, qe_params[section]))

    lines.append("ATOMIC_SPECIES")
    for kind in structure["kinds"]:
        symbol = kind["name"]
        pseudo = pseudos[symbol]
        lines.append(f"{symbol:<6} {kind['mass']:12.6f} {pseudo['filename']}")

    lines.append("ATOMIC_POSITIONS angstrom")
    for site in structure["sites"]:
        x, y, z = site["position"]
        lines.append(f"{site['kind_name']:<6} {x:16.10f} {y:16.10f} {z:16.10f}")

    mesh = kpoints.get("mesh")
    offset = kpoints.get("offset", [0.0, 0.0, 0.0])
    if mesh and not force_kpoints_list:
        offset_int = [1 if abs(float(value) - 0.5) < 1e-12 else 0 for value in offset]
        lines.append("K_POINTS automatic")
        lines.append(
            f"{mesh[0]} {mesh[1]} {mesh[2]} {offset_int[0]} {offset_int[1]} {offset_int[2]}"
        )
    elif mesh:
        points = [
            (i / mesh[0], j / mesh[1], k / mesh[2])
            for i in range(mesh[0])
            for j in range(mesh[1])
            for k in range(mesh[2])
        ]
        lines.append("K_POINTS crystal")
        lines.append(str(len(points)))
        for x, y, z in points:
            lines.append(f"{x:18.10f} {y:18.10f} {z:18.10f} {1.0:18.10f}")
    else:
        raise ValueError("Only mesh KpointsData nodes are supported")

    lines.append("CELL_PARAMETERS angstrom")
    for row in structure["cell"]:
        lines.append("".join(f"{value:19.10f}" for value in row))
    return "\n".join(lines) + "\n"


def wannier_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def format_wannier90_win(
    params: dict[str, Any],
    structure: dict[str, Any],
    kpoints: dict[str, Any],
    kpoint_path: dict[str, Any],
) -> str:
    lines = [
        f"! Reconstructed from Materials Cloud AiiDA archive {MC_RECORD}.",
        f"! Final group: {GROUP_LABEL}.",
        "! Seed used by the archived AiiDA calculations: aiida.",
        "",
    ]
    for key in sorted(params):
        lines.append(f"{key} = {wannier_value(params[key])}")
    if "mesh" in kpoints:
        lines.append(f"mp_grid = {kpoints['mesh'][0]} {kpoints['mesh'][1]} {kpoints['mesh'][2]}")

    lines.extend(["", "begin unit_cell_cart", "Ang"])
    for row in structure["cell"]:
        lines.append("".join(f"{value:19.10f}" for value in row))
    lines.extend(["end unit_cell_cart", "", "begin atoms_cart", "Ang"])
    for site in structure["sites"]:
        x, y, z = site["position"]
        lines.append(f"{site['kind_name']:<6} {x:16.10f} {y:16.10f} {z:16.10f}")
    lines.extend(["end atoms_cart", "", "begin kpoints"])
    mesh = kpoints["mesh"]
    for i in range(mesh[0]):
        for j in range(mesh[1]):
            for k in range(mesh[2]):
                lines.append(f"{i / mesh[0]:18.10f} {j / mesh[1]:18.10f} {k / mesh[2]:18.10f}")
    lines.append("end kpoints")

    if kpoint_path:
        coords = kpoint_path.get("point_coords", {})
        lines.extend(["", "begin kpoint_path"])
        for start, end in kpoint_path.get("path", []):
            c1 = coords[start]
            c2 = coords[end]
            lines.append(
                f"{start:<10} {c1[0]:10.6f} {c1[1]:10.6f} {c1[2]:10.6f} "
                f"{end:<10} {c2[0]:10.6f} {c2[1]:10.6f} {c2[2]:10.6f}"
            )
        lines.append("end kpoint_path")
    return "\n".join(lines) + "\n"


def format_pw2wannier90_in(params: dict[str, Any]) -> str:
    inputpp = dict(params.get("inputpp", params))
    inputpp.setdefault("outdir", "./out/")
    inputpp.setdefault("prefix", "aiida")
    inputpp.setdefault("seedname", "aiida")
    lines = format_namelist("inputpp", inputpp)
    return "\n".join(lines) + "\n"


def lengths_angles(cell: list[list[float]]) -> tuple[list[float], list[float]]:
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def norm(a: list[float]) -> float:
        return math.sqrt(dot(a, a))

    a, b, c = cell
    lengths = [norm(a), norm(b), norm(c)]
    alpha = math.degrees(math.acos(dot(b, c) / (lengths[1] * lengths[2])))
    beta = math.degrees(math.acos(dot(a, c) / (lengths[0] * lengths[2])))
    gamma = math.degrees(math.acos(dot(a, b) / (lengths[0] * lengths[1])))
    return lengths, [alpha, beta, gamma]


def invert_3x3(m: list[list[float]]) -> list[list[float]]:
    a, b, c = m
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(det) < 1e-14:
        raise ValueError("singular cell")
    return [
        [
            (b[1] * c[2] - b[2] * c[1]) / det,
            (a[2] * c[1] - a[1] * c[2]) / det,
            (a[1] * b[2] - a[2] * b[1]) / det,
        ],
        [
            (b[2] * c[0] - b[0] * c[2]) / det,
            (a[0] * c[2] - a[2] * c[0]) / det,
            (a[2] * b[0] - a[0] * b[2]) / det,
        ],
        [
            (b[0] * c[1] - b[1] * c[0]) / det,
            (a[1] * c[0] - a[0] * c[1]) / det,
            (a[0] * b[1] - a[1] * b[0]) / det,
        ],
    ]


def cart_to_frac(position: list[float], cell: list[list[float]]) -> list[float]:
    # Positions are row-vector combinations of the cell rows.
    inv = invert_3x3([[cell[j][i] for j in range(3)] for i in range(3)])
    return [sum(inv[i][j] * position[j] for j in range(3)) for i in range(3)]


def format_cif(material_id: str, structure: dict[str, Any]) -> str:
    lengths, angles = lengths_angles(structure["cell"])
    lines = [
        f"data_{material_id}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        f"_cell_length_a   {lengths[0]:.10f}",
        f"_cell_length_b   {lengths[1]:.10f}",
        f"_cell_length_c   {lengths[2]:.10f}",
        f"_cell_angle_alpha   {angles[0]:.10f}",
        f"_cell_angle_beta    {angles[1]:.10f}",
        f"_cell_angle_gamma   {angles[2]:.10f}",
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
    ]
    counters: Counter[str] = Counter()
    for site in structure["sites"]:
        symbol = site["kind_name"]
        counters[symbol] += 1
        fx, fy, fz = cart_to_frac(site["position"], structure["cell"])
        lines.append(f"  {symbol}{counters[symbol]} {symbol} {fx:.10f} {fy:.10f} {fz:.10f}")
    return "\n".join(lines) + "\n"


def format_xsf(structure: dict[str, Any]) -> str:
    lines = ["CRYSTAL", "PRIMVEC"]
    for row in structure["cell"]:
        lines.append(" ".join(f"{value:16.10f}" for value in row))
    lines.append("PRIMCOORD")
    lines.append(f"{len(structure['sites'])} 1")
    for site in structure["sites"]:
        x, y, z = site["position"]
        lines.append(f"{site['kind_name']:<4} {x:16.10f} {y:16.10f} {z:16.10f}")
    return "\n".join(lines) + "\n"


def zip_members(zip_file: zipfile.ZipFile, node_uuid: str) -> list[str]:
    prefix = uuid_path(node_uuid) + "/"
    return [name for name in zip_file.namelist() if name.startswith(prefix) and name != prefix]


def extract_node_files(zip_file: zipfile.ZipFile, node_uuid: str, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    prefix = uuid_path(node_uuid) + "/"
    written = []
    for member in zip_members(zip_file, node_uuid):
        rel = member.removeprefix(prefix)
        if not rel or rel.endswith("/"):
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zip_file.read(member))
        written.append(str(target.relative_to(destination.parent.parent)))
    return sorted(written)


def extract_single_file(
    zip_file: zipfile.ZipFile, node_uuid: str, source_name: str, destination: Path
) -> bool:
    member = f"{uuid_path(node_uuid)}/{source_name}"
    if member not in zip_file.namelist():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(zip_file.read(member))
    return True


def clean_or_create(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite")
    if path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def trace_material(
    final_uuid: str,
    incoming: dict[str, list[dict[str, str]]],
    outgoing: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    final_retrieved = output_link(outgoing, final_uuid, "retrieved")
    final_output_parameters = output_link(outgoing, final_uuid, "output_parameters")
    structure = input_link(incoming, final_uuid, "structure")
    final_parameters = input_link(incoming, final_uuid, "parameters")
    final_kpoints = input_link(incoming, final_uuid, "kpoints")
    final_kpoint_path = input_link(incoming, final_uuid, "kpoint_path")
    p2w_remote = input_link(incoming, final_uuid, "remote_input_folder")
    p2w_calc = creator_of_remote(incoming, p2w_remote)
    p2w_parameters = input_link(incoming, p2w_calc, "parameters")
    p2w_retrieved = output_link(outgoing, p2w_calc, "retrieved")
    nnkp_file = input_link(incoming, p2w_calc, "nnkp_file")
    nscf_remote = input_link(incoming, p2w_calc, "parent_calc_folder")
    nscf_calc = creator_of_remote(incoming, nscf_remote)
    nscf_parameters = input_link(incoming, nscf_calc, "parameters")
    nscf_kpoints = input_link(incoming, nscf_calc, "kpoints")
    nscf_settings = input_link(incoming, nscf_calc, "settings")
    nscf_retrieved = output_link(outgoing, nscf_calc, "retrieved")
    scf_remote = input_link(incoming, nscf_calc, "parent_calc_folder")
    scf_calc = creator_of_remote(incoming, scf_remote)
    scf_parameters = input_link(incoming, scf_calc, "parameters")
    scf_kpoints = input_link(incoming, scf_calc, "kpoints")
    scf_settings = input_link(incoming, scf_calc, "settings")
    scf_retrieved = output_link(outgoing, scf_calc, "retrieved")
    return {
        "final_wannier90_calc": final_uuid,
        "final_wannier90_retrieved": final_retrieved,
        "final_wannier90_output_parameters": final_output_parameters,
        "structure": structure,
        "final_wannier90_parameters": final_parameters,
        "final_wannier90_kpoints": final_kpoints,
        "final_wannier90_kpoint_path": final_kpoint_path,
        "pw2wannier90_calc": p2w_calc,
        "pw2wannier90_parameters": p2w_parameters,
        "pw2wannier90_retrieved": p2w_retrieved,
        "nnkp_file": nnkp_file,
        "nscf_calc": nscf_calc,
        "nscf_parameters": nscf_parameters,
        "nscf_kpoints": nscf_kpoints,
        "nscf_settings": nscf_settings,
        "nscf_retrieved": nscf_retrieved,
        "scf_calc": scf_calc,
        "scf_parameters": scf_parameters,
        "scf_kpoints": scf_kpoints,
        "scf_settings": scf_settings,
        "scf_retrieved": scf_retrieved,
    }


def relative_files(material_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(material_dir)) for path in material_dir.rglob("*") if path.is_file())


def update_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    by_material = {row["material"]: row for row in existing}
    for row in rows:
        by_material[row["material"]] = row
    merged = [by_material[key] for key in sorted(by_material)]
    summary_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--discover-data", type=Path, default=DEFAULT_DISCOVER_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--materials",
        nargs="+",
        help=(
            "Material IDs to extract. Formula-order aliases are resolved by "
            "composition; use material_id=archive_formula to force a mapping."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = load_json(args.data)
    discover = load_json(args.discover_data)
    nodes = data["export_data"]["Node"]
    attrs = data["node_attributes"]
    uuid_to_pk = {node["uuid"]: pk for pk, node in nodes.items()}
    incoming: dict[str, list[dict[str, str]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, str]]] = defaultdict(list)
    for link in data["links_uuid"]:
        incoming[link["output"]].append(link)
        outgoing[link["input"]].append(link)

    group_uuid = None
    for group in data["export_data"]["Group"].values():
        if group["label"] == GROUP_LABEL:
            group_uuid = group["uuid"]
            break
    if group_uuid is None:
        raise ValueError(f"missing group {GROUP_LABEL}")

    final_by_formula: dict[str, str] = {}
    final_formula_by_structure_uuid: dict[str, str] = {}
    for final_uuid in data["groups_uuid"][group_uuid]:
        structure_uuid = input_link(incoming, final_uuid, "structure")
        structure = attrs_for(uuid_to_pk, attrs, structure_uuid)
        archive_formula = formula_from_structure(structure)
        final_by_formula[archive_formula] = final_uuid
        final_formula_by_structure_uuid[structure_uuid] = archive_formula
    target_map = resolve_target_map(
        args.materials,
        final_by_formula,
        discover=discover,
        final_formula_by_structure_uuid=final_formula_by_structure_uuid,
    )
    archive_display_path = str(args.archive.resolve().relative_to(ROOT))

    rows = []
    with zipfile.ZipFile(args.archive) as zip_file:
        zip_names = set(zip_file.namelist())
        for material_id, archive_formula in target_map.items():
            final_uuid = final_by_formula[archive_formula]
            dft_reference_uuid = discover[material_id]["bands"]["DFT_uuid"]
            trace = trace_material(final_uuid, incoming, outgoing)
            material_dir = args.output_dir / material_id
            clean_or_create(material_dir, args.overwrite)

            structure = attrs_for(uuid_to_pk, attrs, trace["structure"])
            final_params = dict_attrs(uuid_to_pk, attrs, trace["final_wannier90_parameters"])
            p2w_params = dict_attrs(uuid_to_pk, attrs, trace["pw2wannier90_parameters"])
            nscf_params = dict_attrs(uuid_to_pk, attrs, trace["nscf_parameters"])
            scf_params = dict_attrs(uuid_to_pk, attrs, trace["scf_parameters"])
            final_kpoints = attrs_for(uuid_to_pk, attrs, trace["final_wannier90_kpoints"])
            final_kpoint_path = attrs_for(uuid_to_pk, attrs, trace["final_wannier90_kpoint_path"])
            nscf_kpoints = attrs_for(uuid_to_pk, attrs, trace["nscf_kpoints"])
            scf_kpoints = attrs_for(uuid_to_pk, attrs, trace["scf_kpoints"])
            nscf_settings = dict_attrs(uuid_to_pk, attrs, trace["nscf_settings"])
            scf_pseudos = pseudo_inputs(incoming, uuid_to_pk, nodes, attrs, trace["scf_calc"])
            nscf_pseudos = pseudo_inputs(incoming, uuid_to_pk, nodes, attrs, trace["nscf_calc"])
            output_parameters = dict_attrs(
                uuid_to_pk, attrs, trace["final_wannier90_output_parameters"]
            )

            (material_dir / "structure").mkdir(parents=True, exist_ok=True)
            (material_dir / "structure" / f"{material_id}.xsf").write_text(
                format_xsf(structure), encoding="utf-8"
            )
            (material_dir / "structure" / f"{material_id}.cif").write_text(
                format_cif(material_id, structure), encoding="utf-8"
            )

            (material_dir / "scf" / "input").mkdir(parents=True, exist_ok=True)
            (material_dir / "scf" / "input" / "scf.in").write_text(
                format_qe_input(scf_params, structure, scf_kpoints, scf_pseudos, False),
                encoding="utf-8",
            )
            extract_node_files(zip_file, trace["scf_retrieved"], material_dir / "scf" / "output")

            (material_dir / "nscf" / "input").mkdir(parents=True, exist_ok=True)
            force_nscf_list = bool(nscf_settings.get("FORCE_KPOINTS_LIST", False))
            (material_dir / "nscf" / "input" / "nscf.in").write_text(
                format_qe_input(
                    nscf_params,
                    structure,
                    nscf_kpoints,
                    nscf_pseudos,
                    force_nscf_list,
                ),
                encoding="utf-8",
            )
            extract_node_files(zip_file, trace["nscf_retrieved"], material_dir / "nscf" / "output")

            dft_offmesh_reference = material_dir / "dft" / "offmesh" / "reference"
            dft_offmesh_files = extract_node_files(zip_file, dft_reference_uuid, dft_offmesh_reference)
            dft_offmesh_attrs = attrs_for(uuid_to_pk, attrs, dft_reference_uuid)
            dft_offmesh_metadata = {
                "source": {
                    "provider": "Materials Cloud",
                    "record": MC_RECORD,
                    "record_url": MC_RECORD_URL,
                    "archive": archive_display_path,
                    "discover_dft_uuid": dft_reference_uuid,
                },
                "node": {
                    "uuid": dft_reference_uuid,
                    "pk": uuid_to_pk[dft_reference_uuid],
                    "node_type": nodes[uuid_to_pk[dft_reference_uuid]]["node_type"],
                },
                "arrays": {
                    "bands": dft_offmesh_attrs.get("array|bands"),
                    "kpoints": dft_offmesh_attrs.get("array|kpoints"),
                    "weights": dft_offmesh_attrs.get("array|weights"),
                    "occupations": dft_offmesh_attrs.get("array|occupations"),
                },
                "labels": dft_offmesh_attrs.get("labels", []),
                "label_numbers": dft_offmesh_attrs.get("label_numbers", []),
                "cell": dft_offmesh_attrs.get("cell"),
                "units": dft_offmesh_attrs.get("units"),
                "files": sorted(
                    str(path.relative_to(material_dir))
                    for path in dft_offmesh_reference.rglob("*")
                    if path.is_file() and path.name != "metadata.json"
                ),
            }
            (dft_offmesh_reference / "metadata.json").write_text(
                json.dumps(dft_offmesh_metadata, indent=2) + "\n", encoding="utf-8"
            )

            wannier_input = material_dir / "wannier" / "input"
            generated = wannier_input / "generated"
            output_p2w = material_dir / "wannier" / "output" / "pw2wannier90"
            output_w90 = material_dir / "wannier" / "output" / "wannier90"
            wannier_input.mkdir(parents=True, exist_ok=True)
            (wannier_input / "wannier90.win").write_text(
                format_wannier90_win(final_params, structure, final_kpoints, final_kpoint_path),
                encoding="utf-8",
            )
            (wannier_input / "pw2wannier90.in").write_text(
                format_pw2wannier90_in(p2w_params), encoding="utf-8"
            )
            nnkp_written = extract_single_file(
                zip_file, trace["nnkp_file"], "aiida.nnkp", generated / "aiida.nnkp"
            )
            extract_node_files(zip_file, trace["pw2wannier90_retrieved"], output_p2w)
            extract_node_files(zip_file, trace["final_wannier90_retrieved"], output_w90)

            generated_present = sorted(path.name for path in generated.glob("*")) if generated.exists() else []
            generated_expected = ["aiida.nnkp", "aiida.eig", "aiida.mmn", "aiida.amn"]
            unavailable = [name for name in generated_expected if name not in generated_present]
            species = species_from_structure(structure)
            summary_row = {
                "material": material_id,
                "status": "ok_aiida_conductor",
                "num_bands": final_params["num_bands"],
                "num_wann": final_params["num_wann"],
                "archive_formula": archive_formula,
                "source_group": GROUP_LABEL,
            }
            rows.append(summary_row)

            manifest = {
                "source": {
                    "provider": "Materials Cloud",
                    "record": MC_RECORD,
                    "record_url": MC_RECORD_URL,
                    "archive": archive_display_path,
                    "archive_export_version": "0.7",
                    "group_label": GROUP_LABEL,
                },
                "archive_nodes": trace,
                "dft_reference": {
                    "offmesh_bands_uuid": dft_reference_uuid,
                    "offmesh_reference_dir": "dft/offmesh/reference",
                    "offmesh_reference_files": sorted(
                        str(path.relative_to(material_dir))
                        for path in dft_offmesh_reference.rglob("*")
                        if path.is_file()
                    ),
                    "offmesh_bands_shape": dft_offmesh_attrs.get("array|bands"),
                    "offmesh_kpoints_shape": dft_offmesh_attrs.get("array|kpoints"),
                    "labels": dft_offmesh_attrs.get("labels", []),
                    "label_numbers": dft_offmesh_attrs.get("label_numbers", []),
                    "units": dft_offmesh_attrs.get("units"),
                },
                "target": {
                    "material_id": material_id,
                    "archive_formula": archive_formula,
                    "seedname": "aiida",
                    "num_wann": final_params["num_wann"],
                    "num_bands": final_params["num_bands"],
                    "scf_mesh": scf_kpoints.get("mesh"),
                    "nscf_mesh": nscf_kpoints.get("mesh"),
                },
                "input_files": [
                    "input/wannier90.win",
                    "input/pw2wannier90.in",
                    *([f"input/generated/{name}" for name in generated_present]),
                ],
                "outputs_included": {
                    "scf": sorted(
                        str(path.relative_to(material_dir))
                        for path in (material_dir / "scf" / "output").rglob("*")
                        if path.is_file()
                    ),
                    "nscf": sorted(
                        str(path.relative_to(material_dir))
                        for path in (material_dir / "nscf" / "output").rglob("*")
                        if path.is_file()
                    ),
                    "pw2wannier90": sorted(
                        str(path.relative_to(material_dir))
                        for path in output_p2w.rglob("*")
                        if path.is_file()
                    ),
                    "wannier90": sorted(
                        str(path.relative_to(material_dir))
                        for path in output_w90.rglob("*")
                        if path.is_file()
                    ),
                },
                "generated_files": {
                    "included": generated_present,
                    "unavailable_in_archive_retrieved_data": unavailable,
                    "nnkp_from_preprocess_singlefile": nnkp_written,
                },
                "final_output_parameters": output_parameters,
            }
            (material_dir / "wannier" / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            metadata = {
                "material_id": material_id,
                "status": "ok_aiida_conductor",
                "formula": material_id,
                "archive_formula": archive_formula,
                "species": species,
                "num_atoms": len(structure["sites"]),
                "num_species": len(species),
                "structure_files": [
                    f"structure/{material_id}.cif",
                    f"structure/{material_id}.xsf",
                ],
                "source": {
                    "provider": "Materials Cloud",
                    "record": MC_RECORD,
                    "record_url": MC_RECORD_URL,
                    "archive": archive_display_path,
                    "group_label": GROUP_LABEL,
                },
                "reference_params": {
                    "num_bands": final_params["num_bands"],
                    "num_wann": final_params["num_wann"],
                    "qe_nbnd": nscf_params.get("SYSTEM", {}).get("nbnd"),
                    "scf_mesh": scf_kpoints.get("mesh"),
                    "nscf_mesh": nscf_kpoints.get("mesh"),
                    "num_wann_source": "AiiDA final Wannier90 parameters",
                    "num_bands_source": "AiiDA final Wannier90 parameters and NSCF nbnd",
                },
                "artifact_status": {
                    "dft_outputs_included": True,
                    "dft_offmesh_reference_included": True,
                    "wannier_outputs_included": True,
                    "generated_wannier_files_included": generated_present,
                    "generated_wannier_files_unavailable": unavailable,
                },
                "all_files": relative_files(material_dir),
            }
            (material_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

    update_summary(args.output_dir, rows)
    print(f"Wrote {len(rows)} AiiDA conductor packages to {args.output_dir}")
    for row in rows:
        print(
            f"{row['material']:5s} archive_formula={row['archive_formula']:5s} "
            f"num_wann={row['num_wann']:3d} num_bands={row['num_bands']:3d}"
        )


if __name__ == "__main__":
    main()
