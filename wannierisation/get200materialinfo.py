#!/usr/bin/env python3
"""Build a per-material descriptor CSV for the 200 Wannierisation materials.

All paths and feature definitions are intentionally hardcoded for this repo.
The band-index boundary is the gap between DFT bands num_wann and num_wann + 1
using 1-based band labels, i.e. energies[:, num_wann] - energies[:, num_wann-1]
in zero-based Python indexing.
"""

from __future__ import annotations

import csv
import json
import math
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALS_FILE = REPO_ROOT / "staging" / "with_conduction_200_materials.txt"
HARBOR_ROOT = REPO_ROOT / "harbor_datasets" / "wannier_200"
AUTOWANNIER_JSON = REPO_ROOT / "200materials" / "automated_wannier_discover_data.json"
FERMI_JSON = REPO_ROOT / "200materials" / "fermi_energies.json"
CONDUCTIVITY_CSV = REPO_ROOT / "material_conductivity_labels.csv"
XSF_TAR = REPO_ROOT / "200materials" / "xsf.tar.gz"
OUTPUT_CSV = REPO_ROOT / "200materialinfo.csv"

HARTREE_TO_EV = 27.211386245988
BOUNDARY_CROWDING_EV = 0.25
FERMI_DENSITY_HALF_WINDOW_EV = 0.5
COORDINATION_RADIUS_SCALE = 1.25


ELEMENTS = {
    "H": (1, "s", "nonmetal", 0.31),
    "He": (2, "s", "noble_gas", 0.28),
    "Li": (3, "s", "alkali_metal", 1.28),
    "Be": (4, "s", "alkaline_earth_metal", 0.96),
    "B": (5, "p", "metalloid", 0.84),
    "C": (6, "p", "nonmetal", 0.76),
    "N": (7, "p", "nonmetal", 0.71),
    "O": (8, "p", "nonmetal", 0.66),
    "F": (9, "p", "halogen", 0.57),
    "Ne": (10, "p", "noble_gas", 0.58),
    "Na": (11, "s", "alkali_metal", 1.66),
    "Mg": (12, "s", "alkaline_earth_metal", 1.41),
    "Al": (13, "p", "post_transition_metal", 1.21),
    "Si": (14, "p", "metalloid", 1.11),
    "P": (15, "p", "nonmetal", 1.07),
    "S": (16, "p", "nonmetal", 1.05),
    "Cl": (17, "p", "halogen", 1.02),
    "Ar": (18, "p", "noble_gas", 1.06),
    "K": (19, "s", "alkali_metal", 2.03),
    "Ca": (20, "s", "alkaline_earth_metal", 1.76),
    "Sc": (21, "d", "transition_metal", 1.70),
    "Ti": (22, "d", "transition_metal", 1.60),
    "V": (23, "d", "transition_metal", 1.53),
    "Cr": (24, "d", "transition_metal", 1.39),
    "Mn": (25, "d", "transition_metal", 1.39),
    "Fe": (26, "d", "transition_metal", 1.32),
    "Co": (27, "d", "transition_metal", 1.26),
    "Ni": (28, "d", "transition_metal", 1.24),
    "Cu": (29, "d", "transition_metal", 1.32),
    "Zn": (30, "d", "transition_metal", 1.22),
    "Ga": (31, "p", "post_transition_metal", 1.22),
    "Ge": (32, "p", "metalloid", 1.20),
    "As": (33, "p", "metalloid", 1.19),
    "Se": (34, "p", "nonmetal", 1.20),
    "Br": (35, "p", "halogen", 1.20),
    "Kr": (36, "p", "noble_gas", 1.16),
    "Rb": (37, "s", "alkali_metal", 2.20),
    "Sr": (38, "s", "alkaline_earth_metal", 1.95),
    "Y": (39, "d", "transition_metal", 1.90),
    "Zr": (40, "d", "transition_metal", 1.75),
    "Nb": (41, "d", "transition_metal", 1.64),
    "Mo": (42, "d", "transition_metal", 1.54),
    "Tc": (43, "d", "transition_metal", 1.47),
    "Ru": (44, "d", "transition_metal", 1.46),
    "Rh": (45, "d", "transition_metal", 1.42),
    "Pd": (46, "d", "transition_metal", 1.39),
    "Ag": (47, "d", "transition_metal", 1.45),
    "Cd": (48, "d", "transition_metal", 1.44),
    "In": (49, "p", "post_transition_metal", 1.42),
    "Sn": (50, "p", "post_transition_metal", 1.39),
    "Sb": (51, "p", "metalloid", 1.39),
    "Te": (52, "p", "metalloid", 1.38),
    "I": (53, "p", "halogen", 1.39),
    "Xe": (54, "p", "noble_gas", 1.40),
    "Cs": (55, "s", "alkali_metal", 2.44),
    "Ba": (56, "s", "alkaline_earth_metal", 2.15),
    "La": (57, "f", "lanthanide", 2.07),
    "Hf": (72, "d", "transition_metal", 1.75),
    "Ta": (73, "d", "transition_metal", 1.70),
    "W": (74, "d", "transition_metal", 1.62),
    "Re": (75, "d", "transition_metal", 1.51),
    "Os": (76, "d", "transition_metal", 1.44),
    "Ir": (77, "d", "transition_metal", 1.41),
    "Pt": (78, "d", "transition_metal", 1.36),
    "Au": (79, "d", "transition_metal", 1.36),
    "Hg": (80, "d", "transition_metal", 1.32),
    "Tl": (81, "p", "post_transition_metal", 1.45),
    "Pb": (82, "p", "post_transition_metal", 1.46),
    "Bi": (83, "p", "post_transition_metal", 1.48),
}


def parse_formula(formula: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for symbol, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[symbol] += int(count_text or "1")
    if not counts:
        raise ValueError(f"Could not parse formula: {formula}")
    return counts


def fraction_json(counts: Counter[str], attr_index: int) -> str:
    total = sum(counts.values())
    fractions: Counter[str] = Counter()
    for symbol, count in counts.items():
        fractions[ELEMENTS[symbol][attr_index]] += count / total
    return json.dumps(dict(sorted(fractions.items())), sort_keys=True)


def load_conductivity() -> dict[str, dict[str, str]]:
    with CONDUCTIVITY_CSV.open(newline="") as handle:
        return {row["formula"]: row for row in csv.DictReader(handle)}


def load_xsf_structures() -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    structures = {}
    with tarfile.open(XSF_TAR, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".xsf"):
                continue
            material = Path(member.name).stem
            text = archive.extractfile(member).read().decode()
            structures[material] = parse_xsf(text)
    return structures


def parse_xsf(text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    primvec_i = lines.index("PRIMVEC 1")
    cell = np.array(
        [[float(x) for x in lines[primvec_i + offset].split()] for offset in (1, 2, 3)],
        dtype=float,
    )
    primcoord_i = lines.index("PRIMCOORD 1")
    nat = int(lines[primcoord_i + 1].split()[0])
    symbols = []
    positions = []
    for line in lines[primcoord_i + 2 : primcoord_i + 2 + nat]:
        fields = line.split()
        symbols.append(atomic_symbol(fields[0]))
        positions.append([float(x) for x in fields[1:4]])
    return symbols, np.array(positions, dtype=float), cell


def atomic_symbol(value: str) -> str:
    if not value.isdigit():
        return value
    atomic_number = int(value)
    for symbol, (number, _, _, _) in ELEMENTS.items():
        if number == atomic_number:
            return symbol
    raise ValueError(f"Unknown atomic number in XSF: {value}")


def structural_features(
    symbols: list[str], positions: np.ndarray, cell: np.ndarray
) -> tuple[float, float, float]:
    nat = len(symbols)
    volume = abs(float(np.linalg.det(cell)))
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell
    coord_counts = [0] * nat
    nearest = [math.inf] * nat

    for i in range(nat):
        for j in range(i + 1, nat):
            delta_frac = frac[j] - frac[i]
            delta_frac -= np.round(delta_frac)
            distance = float(np.linalg.norm(delta_frac @ cell))
            nearest[i] = min(nearest[i], distance)
            nearest[j] = min(nearest[j], distance)
            cutoff = COORDINATION_RADIUS_SCALE * (
                ELEMENTS[symbols[i]][3] + ELEMENTS[symbols[j]][3]
            )
            if distance <= cutoff:
                coord_counts[i] += 1
                coord_counts[j] += 1

    mean_nearest = float(np.mean(nearest)) if nat > 1 else 0.0
    avg_coordination = float(np.mean(coord_counts)) if nat else 0.0
    return avg_coordination, mean_nearest, volume / nat


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def get_metadata(material: str) -> dict:
    return read_json(HARBOR_ROOT / material / "environment" / "material" / "metadata.json")


def get_grading_metadata(material: str) -> dict:
    return read_json(HARBOR_ROOT / material / "tests" / "reference" / "grading_metadata.json")


def scf_symmetry_ops(material: str) -> int | None:
    material_root = HARBOR_ROOT / material / "environment" / "material"
    scf_candidates = [
        material_root / "qe_save" / "logs" / "scf.out",
        material_root / "scf" / "output" / "aiida.out",
    ]
    scf_log = next((path for path in scf_candidates if path.exists()), None)
    if scf_log is None:
        return None
    text = scf_log.read_text(errors="replace")
    match = re.search(r"(\d+)\s+Sym\.\s+Ops\.", text)
    if match:
        return int(match.group(1))
    match = re.search(r"NUMBER_OF_SYMMETRIES.*?(\d+)", text, re.S)
    if match:
        return int(match.group(1))
    if "No symmetry found" in text:
        return 1
    return None


def parse_qe_eigenvalues_ev(material: str) -> tuple[np.ndarray, float | None]:
    material_root = HARBOR_ROOT / material / "environment" / "material"
    xml_candidates = [
        material_root / "qe_save" / "out" / "aiida.save" / "data-file-schema.xml",
        material_root / "qe_save" / "out" / "aiida.xml",
        material_root / "nscf" / "output" / "data-file.xml",
    ]
    xml_path = next((path for path in xml_candidates if path.exists()), None)
    if xml_path is None:
        raise FileNotFoundError(f"No QE XML found for {material}")
    root = ET.parse(xml_path).getroot()
    eigenvalue_rows = []
    for node in root.iter():
        if strip_namespace(node.tag) == "eigenvalues" and node.text:
            eigenvalue_rows.append([float(x) * HARTREE_TO_EV for x in node.text.split()])
    fermi = None
    for node in root.iter():
        if strip_namespace(node.tag) == "fermi_energy" and node.text:
            fermi = float(node.text) * HARTREE_TO_EV
            break
    if not eigenvalue_rows:
        output_path = material_root / "nscf" / "output" / "aiida.out"
        eigenvalue_rows = parse_qe_output_band_blocks(output_path)
        if not eigenvalue_rows:
            raise ValueError(f"No eigenvalues found in {xml_path} or {output_path}")
    return np.array(eigenvalue_rows, dtype=float), fermi


def parse_qe_output_band_blocks(output_path: Path) -> list[list[float]]:
    rows = []
    current = None
    for line in output_path.read_text(errors="replace").splitlines():
        if "bands (ev):" in line:
            current = []
            continue
        if current is None:
            continue
        if "occupation numbers" in line:
            if current:
                rows.append(current)
            current = None
            continue
        stripped = line.strip()
        if not stripped:
            continue
        try:
            current.extend(float(value) for value in stripped.split())
        except ValueError:
            current = None
    return rows


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def band_features(
    energies_ev: np.ndarray, fermi_ev: float, num_wann: int, num_atoms: int
) -> dict[str, float | int | None]:
    band_mins = energies_ev.min(axis=0)
    band_maxs = energies_ev.max(axis=0)
    bands_crossing = int(np.count_nonzero((band_mins <= fermi_ev) & (fermi_ev <= band_maxs)))
    density_count = np.count_nonzero(np.abs(energies_ev - fermi_ev) <= FERMI_DENSITY_HALF_WINDOW_EV)
    density = float(density_count / energies_ev.shape[0] / num_atoms)

    if num_wann >= energies_ev.shape[1]:
        return {
            "band_density_ef_0p5_per_atom": density,
            "bands_crossing_ef": bands_crossing,
            "boundary_crowding_fraction_0p25": None,
            "boundary_gap_p10_ev": None,
        }

    boundary_gaps = energies_ev[:, num_wann] - energies_ev[:, num_wann - 1]

    return {
        "band_density_ef_0p5_per_atom": density,
        "bands_crossing_ef": bands_crossing,
        "boundary_crowding_fraction_0p25": float(
            np.count_nonzero(boundary_gaps <= BOUNDARY_CROWDING_EV) / len(boundary_gaps)
        ),
        "boundary_gap_p10_ev": float(np.percentile(boundary_gaps, 10)),
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def main() -> None:
    materials = [line.strip() for line in MATERIALS_FILE.read_text().splitlines() if line.strip()]
    conductivity = load_conductivity()
    xsf_structures = load_xsf_structures()
    fermi_from_json = read_json(FERMI_JSON)
    autowannier = read_json(AUTOWANNIER_JSON)

    fieldnames = [
        "material",
        "block_fractions",
        "category_fractions",
        "num_wann",
        "wann_per_atom",
        "num_atoms",
        "metallicity",
        "band_density_ef_0p5_per_atom",
        "bands_crossing_ef",
        "boundary_crowding_fraction_0p25",
        "boundary_gap_p10_ev",
        "avg_coordination",
        "scf_symmetry_ops",
        "mean_nearest_neighbor_ang",
        "volume_per_atom_ang3",
        "num_bands",
        "fermi_energy_ev",
        "structure_uuid",
        "dft_uuid",
    ]

    rows = []
    for material in materials:
        counts = parse_formula(material)
        metadata = get_metadata(material)
        grading = get_grading_metadata(material)
        num_atoms = int(metadata["num_atoms"])
        num_wann = int(grading["num_wann"])
        num_bands = int(grading["num_bands"])
        fermi_ev = float(fermi_from_json.get(material, grading["fermi_energy_eV"]))

        symbols, positions, cell = xsf_structures[material]
        avg_coordination, mean_nearest, volume_per_atom = structural_features(symbols, positions, cell)

        energies_ev, xml_fermi = parse_qe_eigenvalues_ev(material)
        if material not in fermi_from_json and xml_fermi is not None:
            fermi_ev = xml_fermi
        band_info = band_features(energies_ev, fermi_ev, num_wann, num_atoms)

        conductive = conductivity[material]["is_conductive"]
        row = {
            "material": material,
            "block_fractions": fraction_json(counts, 1),
            "category_fractions": fraction_json(counts, 2),
            "num_wann": num_wann,
            "wann_per_atom": num_wann / num_atoms,
            "num_atoms": num_atoms,
            "metallicity": "metallic" if conductive == "True" else "not_metallic",
            "band_density_ef_0p5_per_atom": band_info["band_density_ef_0p5_per_atom"],
            "bands_crossing_ef": band_info["bands_crossing_ef"],
            "boundary_crowding_fraction_0p25": band_info["boundary_crowding_fraction_0p25"],
            "boundary_gap_p10_ev": band_info["boundary_gap_p10_ev"],
            "avg_coordination": avg_coordination,
            "scf_symmetry_ops": scf_symmetry_ops(material),
            "mean_nearest_neighbor_ang": mean_nearest,
            "volume_per_atom_ang3": volume_per_atom,
            "num_bands": num_bands,
            "fermi_energy_ev": fermi_ev,
            "structure_uuid": autowannier[material]["structure_uuid"],
            "dft_uuid": autowannier[material]["bands"]["DFT_uuid"],
        }
        rows.append({key: fmt(row[key]) for key in fieldnames})

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
