import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

JSON_PATH = Path("/home/jiasheng/WannierisationBenchmarking/material_similarity_candidates.json")
ORIGINAL_CSV_PATH = Path("/home/jiasheng/WannierisationBenchmarking/jobs/successful_run_errors.csv")
NEW_RUN_CSV_PATH = Path("/home/jiasheng/WannierisationBenchmarking/jobs/gemini_self_debug_reviews_chemical_similarity/all_error_ratios_by_material.csv")

OUT_PNG = Path("/home/jiasheng/WannierisationBenchmarking/jobs/gemini_self_debug_reviews_chemical_similarity/material_dependency_graph.png")
OUT_SVG = Path("/home/jiasheng/WannierisationBenchmarking/jobs/gemini_self_debug_reviews_chemical_similarity/material_dependency_graph.svg")

TARGET_GREEN = "#7BC96F"
CANDIDATE_ONLY_FILL_SIZE = 950

# Target nodes are now drawn in TWO layers:
#   1. TARGET_BASE_SIZE: full-size green disk showing this is a target material
#   2. NEW_RUN_INNER_FILL_SIZE: smaller blue/red disk showing avg_new_run_error_ratio
# Shrink THIS value if the blue/red center is too large.
TARGET_BASE_SIZE = 1050
NEW_RUN_INNER_FILL_SIZE = 200

CANDIDATE_ONLY_RING_SIZE = 1060
TARGET_AND_CANDIDATE_RING_SIZE = 1160
RATIO_RING_LINEWIDTH = 2.4


# ----------------------------
# Helper functions
# ----------------------------
def material_ratio_dict(df: pd.DataFrame, material_col: str, ratio_col: str) -> dict:
    """Return mean finite positive ratio per material."""
    return (
        df[[material_col, ratio_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[material_col, ratio_col])
        .groupby(material_col)[ratio_col]
        .mean()
        .dropna()
        .to_dict()
    )


def make_log_ratio_norm(ratio_by_material: dict, materials) -> mpl.colors.TwoSlopeNorm:
    """
    Build a log10 ratio norm centered at ratio = 1.

    ratio < 1 -> blue side
    ratio = 1 -> white center
    ratio > 1 -> red side
    """
    logs = np.array([
        np.log10(ratio_by_material[m])
        for m in materials
        if (
            m in ratio_by_material
            and np.isfinite(ratio_by_material[m])
            and ratio_by_material[m] > 0
        )
    ])

    if len(logs) == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = min(float(np.nanmin(logs)), 0.0)
        vmax = max(float(np.nanmax(logs)), 0.0)

        # TwoSlopeNorm requires vmin < vcenter < vmax.
        if vmin == 0.0:
            vmin = -1e-6
        if vmax == 0.0:
            vmax = 1e-6

    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def set_ratio_colorbar_ticks(cbar, norm: mpl.colors.TwoSlopeNorm):
    """Place readable ratio ticks on a log10-ratio colorbar."""
    tick_logs = np.linspace(norm.vmin, norm.vmax, 5)
    tick_logs = np.unique(np.r_[tick_logs, 0.0])  # force ratio = 1 tick
    tick_logs = tick_logs[(tick_logs >= norm.vmin) & (tick_logs <= norm.vmax)]

    cbar.set_ticks(tick_logs)
    cbar.set_ticklabels([f"{10 ** t:.3g}" for t in tick_logs])


# ----------------------------
# Load data
# ----------------------------
with open(JSON_PATH, "r") as f:
    candidates_by_target = json.load(f)

original_df = pd.read_csv(ORIGINAL_CSV_PATH)
new_run_df = pd.read_csv(NEW_RUN_CSV_PATH)

# ORIGINAL_CSV_PATH is the old successful-run CSV used by the original script.
# It provides the existing candidate outer-ring ratio.
original_required_cols = {"material", "gemini_to_reference_ratio"}
missing_original = original_required_cols - set(original_df.columns)
if missing_original:
    raise ValueError(
        f"Original successful-run CSV missing required columns: {missing_original}"
    )

# NEW_RUN_CSV_PATH is the new averaged CSV.
# It provides the avg_new_run_error_ratio for target-node inner fill colors.
new_run_required_cols = {"material", "avg_new_run_error_ratio"}
missing_new_run = new_run_required_cols - set(new_run_df.columns)
if missing_new_run:
    raise ValueError(
        f"New-run averages CSV missing required columns: {missing_new_run}"
    )

# Outer candidate ring = original successful-run average ratio from jobs/successful_run_errors.csv.
# Inner target fill = avg_new_run_error_ratio from the new averaged CSV.
candidate_ratio_col = "gemini_to_reference_ratio"
new_run_ratio_col = "avg_new_run_error_ratio"

candidate_ratio = material_ratio_dict(original_df, "material", candidate_ratio_col)
new_run_ratio = material_ratio_dict(new_run_df, "material", new_run_ratio_col)


# ----------------------------
# Build directed graph
# ----------------------------
G = nx.DiGraph()

# JSON keys are target materials.
target_nodes = set(candidates_by_target.keys())

# JSON list entries are candidate materials.
candidate_nodes = set()

for target, candidates in candidates_by_target.items():
    G.add_node(target)
    for cand in candidates:
        candidate_nodes.add(cand)
        G.add_node(cand)
        G.add_edge(target, cand)

# Drawing classes:
#   target_only_nodes: target materials that are not candidates
#   candidate_only_nodes: candidate materials that are not targets
#   target_and_candidate_nodes: target materials that also appear as candidates
#
# Visual encodings:
#   inner fill color = avg_new_run_error_ratio, for target materials with data
#   green border      = target material / JSON key
#   outer ring        = original successful-run ratio, for candidate-list materials
#   white fill        = candidate-only material without target/new-run encoding
#   gray ring         = missing original successful-run ratio
#
# This avoids putting two quantitative encodings on the same outer ring.
target_only_nodes = sorted(target_nodes - candidate_nodes)
candidate_only_nodes = sorted(candidate_nodes - target_nodes)
target_and_candidate_nodes = sorted(target_nodes & candidate_nodes)
ring_nodes = sorted(candidate_nodes)

all_nodes = set(G.nodes)
target_fill_nodes = sorted(target_nodes)


# ----------------------------
# Color mapping
# ----------------------------
# log10 scale centered at ratio = 1.
# ratio < 1 -> blue
# ratio = 1 -> white
# ratio > 1 -> red
# Missing/invalid candidate ratio -> gray outer ring
# Missing/invalid new-run ratio -> original green target fill
blue_white_red = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_white_red",
    ["#08306B", "#9ECAE1", "#FFFFFF", "#FCAE91", "#99000D"],
)

candidate_norm = make_log_ratio_norm(candidate_ratio, ring_nodes)
new_run_norm = make_log_ratio_norm(new_run_ratio, target_fill_nodes)


def candidate_ring_color(material: str):
    r = candidate_ratio.get(material)
    if r is None or not np.isfinite(r) or r <= 0:
        return "#BBBBBB"
    return blue_white_red(candidate_norm(np.log10(r)))


def target_fill_color(material: str):
    r = new_run_ratio.get(material)
    if r is None or not np.isfinite(r) or r <= 0:
        return TARGET_GREEN  # fallback: original target green
    return blue_white_red(new_run_norm(np.log10(r)))


# ----------------------------
# Layout
# ----------------------------
# Deterministic straight-edge, hard-non-overlap layout.
#
# What this does:
#   1. Get a global starting layout from Graphviz/sfdp, or spring_layout.
#   2. Spread green target nodes apart first.
#   3. Put single-parent candidate-only nodes on radial/concentric slots around
#      their target.
#   4. Put multi-parent candidate-only nodes near the centroid of their green
#      parents.
#   5. Run a HARD disk-collision solver on every drawn node, using the largest
#      node marker size for that node type plus label padding.
#   6. Run a straight-edge clearance pass: if an edge segment passes through an
#      unrelated node disk, move the node/endpoints; the edge is NEVER bent.
#   7. Run the hard collision solver again.
#
# Important: edges below are still drawn with nx.draw_networkx_edges(...). There
# are no curved, routed, or polyline edges in this version.
try:
    pos0 = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
except Exception:
    try:
        pos0 = nx.nx_pydot.graphviz_layout(G, prog="sfdp")
    except Exception:
        pos0 = nx.spring_layout(G, seed=7, k=3.5, iterations=2000)

pos = {n: np.asarray(xy, dtype=float).copy() for n, xy in pos0.items()}


def _layout_span(pos: dict) -> float:
    """Return the larger x/y span of the current layout."""
    coords = np.asarray(list(pos.values()), dtype=float)
    span = coords.max(axis=0) - coords.min(axis=0)
    return float(max(span[0], span[1], 1.0))


def _unit(v: np.ndarray, fallback_angle: float = 0.0) -> np.ndarray:
    """Return a unit vector, using fallback_angle if v is nearly zero."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([np.cos(fallback_angle), np.sin(fallback_angle)])
    return v / norm


def _segment_distance_to_point(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Distance from p to segment a-b, the segment parameter, and closest point."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a)), 0.0, a.copy()
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest)), t, closest


def _node_size_for_clearance(node: str) -> float:
    """
    Largest marker size used for this node. This is only used to estimate
    relative disk radii for the layout; matplotlib marker sizes are in points^2,
    so the absolute conversion is intentionally approximate.
    """
    if node in target_and_candidate_nodes:
        return max(TARGET_AND_CANDIDATE_RING_SIZE, TARGET_BASE_SIZE)
    if node in target_only_nodes:
        return TARGET_BASE_SIZE
    if node in candidate_only_nodes:
        return max(CANDIDATE_ONLY_RING_SIZE, CANDIDATE_ONLY_FILL_SIZE)
    return TARGET_BASE_SIZE


def _node_radii(pos: dict, radius_scale_frac: float = 0.0105) -> dict[str, float]:
    """
    Convert marker sizes into conservative data-coordinate radii.

    Increase radius_scale_frac if labels still feel tight. The defaults are
    deliberately conservative because the plot uses large markers and bold text.
    """
    span = _layout_span(pos)
    return {
        n: radius_scale_frac * span * np.sqrt(_node_size_for_clearance(n) / 1000.0)
        for n in pos
    }


def _spread_targets(pos: dict, min_dist_frac: float = 0.18, iterations: int = 1000):
    """Spread only green target nodes before attaching candidate rings."""
    targets = [n for n in sorted(target_nodes) if n in pos]
    if len(targets) < 2:
        return

    coords = np.asarray([pos[n] for n in targets], dtype=float)
    center0 = coords.mean(axis=0)
    min_dist = min_dist_frac * _layout_span(pos)
    rng = np.random.default_rng(7)

    for _ in range(iterations):
        delta = np.zeros_like(coords)
        max_move = 0.0
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                diff = coords[j] - coords[i]
                dist = float(np.linalg.norm(diff))
                if dist < 1e-12:
                    theta = rng.uniform(0.0, 2.0 * np.pi)
                    direction = np.array([np.cos(theta), np.sin(theta)])
                    dist = 1e-12
                else:
                    direction = diff / dist
                if dist < min_dist:
                    push = 0.52 * (min_dist - dist) * direction
                    delta[i] -= push
                    delta[j] += push
        coords += delta
        coords += center0 - coords.mean(axis=0)
        max_move = float(np.max(np.linalg.norm(delta, axis=1)))
        if max_move < 1e-3:
            break

    for n, xy in zip(targets, coords):
        pos[n] = xy


def _initial_radial_candidate_layout(
    pos: dict,
    old_pos: dict,
    base_radius_frac: float = 0.105,
    ring_gap_frac: float = 0.070,
    max_per_ring: int = 7,
):
    """
    Put each target's private candidate-only neighbors on concentric rings.
    Shared candidates are handled separately, since they cannot be centered on
    one target without lying to the graph structure.
    """
    span = _layout_span(pos)
    base_radius = base_radius_frac * span
    ring_gap = ring_gap_frac * span
    target_positions = {t: np.asarray(pos[t], dtype=float) for t in target_nodes if t in pos}
    target_centroid = np.asarray(list(target_positions.values()), dtype=float).mean(axis=0)
    candidate_only_set = set(candidate_only_nodes)
    shared = set()

    for target in sorted(target_nodes):
        if target not in pos:
            continue
        center = np.asarray(pos[target], dtype=float)
        candidates = [n for n in G.successors(target) if n in candidate_only_set and n in pos]
        private = []
        for n in candidates:
            parents = [p for p in G.predecessors(n) if p in target_nodes]
            if len(parents) == 1:
                private.append(n)
            else:
                shared.add(n)

        if not private:
            continue

        away = _unit(center - target_centroid, fallback_angle=np.deg2rad(90.0))
        start_angle = float(np.arctan2(away[1], away[0]))

        def old_angle(n: str) -> float:
            v = np.asarray(old_pos[n], dtype=float) - center
            return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-12 else 0.0

        # Stable order from original layout, then name.
        private = sorted(private, key=lambda n: (old_angle(n), n))

        idx = 0
        ring = 0
        while idx < len(private):
            count = min(max_per_ring + 4 * ring, len(private) - idx)
            radius = base_radius + ring * ring_gap
            # Rotate each ring to avoid spokes lining up when there are many nodes.
            offset = start_angle + ring * np.deg2rad(17.0)
            for k in range(count):
                theta = offset + 2.0 * np.pi * k / count
                node = private[idx]
                pos[node] = center + radius * np.array([np.cos(theta), np.sin(theta)])
                idx += 1
            ring += 1

    # Put shared candidate-only nodes near the centroid of their target parents,
    # then the force/clearance passes will separate them cleanly.
    for node in sorted(shared):
        parents = [p for p in G.predecessors(node) if p in target_positions]
        if not parents:
            continue
        parent_xy = np.asarray([target_positions[p] for p in parents], dtype=float)
        centroid = parent_xy.mean(axis=0)
        away = _unit(centroid - target_centroid, fallback_angle=0.0)
        pos[node] = centroid + 0.50 * base_radius * away


def _hard_resolve_node_collisions(
    pos: dict,
    min_gap_frac: float = 0.018,
    iterations: int = 3000,
    damping: float = 0.86,
):
    """
    Hard disk-packing pass. This is the part that prevents intersecting nodes.
    It treats every visible marker as a disk, with extra gap for labels.
    """
    nodes = list(pos)
    coords = np.asarray([pos[n] for n in nodes], dtype=float)
    rng = np.random.default_rng(17)

    for _ in range(iterations):
        tmp = {n: coords[i] for i, n in enumerate(nodes)}
        radii = _node_radii(tmp)
        span = _layout_span(tmp)
        min_gap = min_gap_frac * span
        delta = np.zeros_like(coords)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                diff = coords[j] - coords[i]
                dist = float(np.linalg.norm(diff))
                if dist < 1e-12:
                    theta = rng.uniform(0.0, 2.0 * np.pi)
                    direction = np.array([np.cos(theta), np.sin(theta)])
                    dist = 1e-12
                else:
                    direction = diff / dist

                required = radii[nodes[i]] + radii[nodes[j]] + min_gap
                if dist < required:
                    push = 0.53 * (required - dist) * direction
                    delta[i] -= push
                    delta[j] += push

        max_move = float(np.max(np.linalg.norm(delta, axis=1)))
        coords += damping * delta
        if max_move < 1e-3:
            break

    for n, xy in zip(nodes, coords):
        pos[n] = xy


def _straight_edge_clearance_pass(
    pos: dict,
    iterations: int = 900,
    clearance_gap_frac: float = 0.013,
):
    """
    Move nodes so straight edge segments do not run through unrelated node disks.
    This does not route or bend edges; it only changes node positions.
    """
    nodes = list(pos)
    node_index = {n: i for i, n in enumerate(nodes)}
    coords = np.asarray([pos[n] for n in nodes], dtype=float)
    edges = [(u, v) for u, v in G.edges() if u in node_index and v in node_index]
    rng = np.random.default_rng(23)

    for _ in range(iterations):
        tmp = {n: coords[i] for i, n in enumerate(nodes)}
        radii = _node_radii(tmp)
        span = _layout_span(tmp)
        gap = clearance_gap_frac * span
        delta = np.zeros_like(coords)

        for u, v in edges:
            iu = node_index[u]
            iv = node_index[v]
            a = coords[iu]
            b = coords[iv]
            ab = b - a
            if np.linalg.norm(ab) < 1e-12:
                continue

            for w in nodes:
                if w == u or w == v:
                    continue
                iw = node_index[w]
                p = coords[iw]
                dist, t, closest = _segment_distance_to_point(a, b, p)
                if not (0.10 < t < 0.90):
                    continue

                # Strongest protection for green nodes, but apply to all nodes.
                multiplier = 1.65 if w in target_nodes else 1.0
                required = multiplier * radii[w] + gap
                if dist < required:
                    normal = _unit(p - closest, fallback_angle=rng.uniform(0, 2 * np.pi))
                    push = 0.34 * (required - dist) * normal

                    # Move the blocking node away from the segment, and move
                    # endpoints slightly the other way. This keeps edges straight
                    # while opening clearance.
                    delta[iw] += 1.35 * push
                    delta[iu] -= 0.20 * (1.0 - t) * push
                    delta[iv] -= 0.20 * t * push

        max_move = float(np.max(np.linalg.norm(delta, axis=1)))
        coords += delta
        if max_move < 1e-3:
            break

    for n, xy in zip(nodes, coords):
        pos[n] = xy


def _normalize_layout(pos: dict, target_span: float = 1000.0):
    """Normalize layout size so the saved figure has stable margins."""
    nodes = list(pos)
    coords = np.asarray([pos[n] for n in nodes], dtype=float)
    center = coords.mean(axis=0)
    coords -= center
    span = max(float(coords[:, 0].max() - coords[:, 0].min()), float(coords[:, 1].max() - coords[:, 1].min()), 1.0)
    coords *= target_span / span
    for n, xy in zip(nodes, coords):
        pos[n] = xy


# Build the layout.
_normalize_layout(pos, target_span=1000.0)
_spread_targets(pos, min_dist_frac=0.20, iterations=1200)
_initial_radial_candidate_layout(pos, pos0, base_radius_frac=0.120, ring_gap_frac=0.078, max_per_ring=7)

# The sequence below is intentional: collision -> edge clearance -> collision.
# The last collision pass is what guarantees nodes do not sit on top of each other.
_hard_resolve_node_collisions(pos, min_gap_frac=0.021, iterations=3500)
_straight_edge_clearance_pass(pos, iterations=1200, clearance_gap_frac=0.016)
_hard_resolve_node_collisions(pos, min_gap_frac=0.024, iterations=4000)
_normalize_layout(pos, target_span=1000.0)

# NetworkX drawing accepts tuples/lists; keep final positions simple.
pos = {n: tuple(xy) for n, xy in pos.items()}


# ----------------------------
# Draw
# ----------------------------
plt.figure(figsize=(22, 16))
ax = plt.gca()

ax.set_title(
    "Material similarity dependency graph\n"
    "Inner fill = avg new-run error ratio, Green outline = target material, "
    "Outer ring = original successful-run ratio",
    fontsize=15,
    pad=18,
)

# Edges: target -> candidate
# Draw as plain line segments instead of FancyArrowPatch arrows.
# FancyArrowPatch can crash with StopIteration when matplotlib tries to clip
# very short/curved arrows around large nodes. Lines are stable and keep the
# dependency edges visible.
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    arrows=False,
    width=1.25,
    alpha=0.55,
    edge_color="#666666",
)

# ------------------------------------------------------------------
# Draw original successful-run ratio rings as separate larger hollow nodes.
# This keeps the outer ring available for the existing candidate ratio.
# The new-run ratio is encoded in the inner fill of target nodes.
# ------------------------------------------------------------------

# Candidate-only ratio rings: ring around white node
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=candidate_only_nodes,
    node_color="none",
    edgecolors=[candidate_ring_color(n) for n in candidate_only_nodes],
    linewidths=RATIO_RING_LINEWIDTH,
    node_size=CANDIDATE_ONLY_RING_SIZE,
    ax=ax,
)

# Target-and-candidate ratio rings: ring around target node
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=target_and_candidate_nodes,
    node_color="none",
    edgecolors=[candidate_ring_color(n) for n in target_and_candidate_nodes],
    linewidths=RATIO_RING_LINEWIDTH,
    node_size=TARGET_AND_CANDIDATE_RING_SIZE,
    ax=ax,
)

# Candidate-only fills: white center
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=candidate_only_nodes,
    node_color="#FFFFFF",
    edgecolors="#DDDDDD",
    linewidths=0.8,
    node_size=CANDIDATE_ONLY_FILL_SIZE,
    ax=ax,
)

# ------------------------------------------------------------------
# Target nodes are drawn in TWO layers so the new-run color is smaller:
#   Layer 1: full-size green base disk = target material
#   Layer 2: smaller blue/red inner disk = avg_new_run_error_ratio
#
# To shrink the blue/red section inside green nodes, decrease
# NEW_RUN_INNER_FILL_SIZE near the top of the script.
# ------------------------------------------------------------------

# Target-only green base disks
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=target_only_nodes,
    node_color=TARGET_GREEN,
    edgecolors=TARGET_GREEN,
    linewidths=1.2,
    node_size=TARGET_BASE_SIZE,
    ax=ax,
)

# Target-and-candidate green base disks, inside the outer candidate ratio ring
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=target_and_candidate_nodes,
    node_color=TARGET_GREEN,
    edgecolors=TARGET_GREEN,
    linewidths=1.2,
    node_size=TARGET_BASE_SIZE,
    ax=ax,
)

# Target-only smaller blue/red inner disks for avg_new_run_error_ratio
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=target_only_nodes,
    node_color=[target_fill_color(n) for n in target_only_nodes],
    edgecolors="none",
    linewidths=0.0,
    node_size=NEW_RUN_INNER_FILL_SIZE,
    ax=ax,
)

# Target-and-candidate smaller blue/red inner disks for avg_new_run_error_ratio
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=target_and_candidate_nodes,
    node_color=[target_fill_color(n) for n in target_and_candidate_nodes],
    edgecolors="none",
    linewidths=0.0,
    node_size=NEW_RUN_INNER_FILL_SIZE,
    ax=ax,
)

# Labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=8,
    font_weight="bold",
    ax=ax,
)


# ----------------------------
# Colorbars
# ----------------------------
# Candidate/original-run outer-ring colorbar
candidate_sm = mpl.cm.ScalarMappable(norm=candidate_norm, cmap=blue_white_red)
candidate_sm.set_array([])

candidate_cbar = plt.colorbar(candidate_sm, ax=ax, shrink=0.72, pad=0.01)
set_ratio_colorbar_ticks(candidate_cbar, candidate_norm)
candidate_cbar.set_label(
    "Outer ring: average gemini_to_reference_ratio from successful_run_errors.csv",
    fontsize=11,
)

# New-run inner-fill colorbar
new_run_sm = mpl.cm.ScalarMappable(norm=new_run_norm, cmap=blue_white_red)
new_run_sm.set_array([])

new_run_cbar = plt.colorbar(new_run_sm, ax=ax, shrink=0.72, pad=0.075)
set_ratio_colorbar_ticks(new_run_cbar, new_run_norm)
new_run_cbar.set_label(
    "Inner fill: avg_new_run_error_ratio for target materials",
    fontsize=11,
)


# ----------------------------
# Legend
# ----------------------------
target_new_run_patch = mpl.lines.Line2D(
    [0], [0],
    marker="o",
    color="w",
    markerfacecolor="#FCAE91",
    markeredgecolor=TARGET_GREEN,
    markeredgewidth=2,
    markersize=13,
    label="Target material: green disk; smaller center colored by avg_new_run_error_ratio",
)

candidate_patch = mpl.lines.Line2D(
    [0], [0],
    marker="o",
    color="w",
    markerfacecolor="#FFFFFF",
    markeredgecolor="#99000D",
    markeredgewidth=3,
    markersize=13,
    label="Candidate-only material: outer ring colored by original successful-run ratio",
)

both_patch = mpl.lines.Line2D(
    [0], [0],
    marker="o",
    color="w",
    markerfacecolor="#FCAE91",
    markeredgecolor="#99000D",
    markeredgewidth=3,
    markersize=13,
    label="Target + candidate: green disk; smaller center = new-run ratio; outer ring = original ratio",
)

missing_new_run_patch = mpl.lines.Line2D(
    [0], [0],
    marker="o",
    color="w",
    markerfacecolor=TARGET_GREEN,
    markeredgecolor=TARGET_GREEN,
    markeredgewidth=2,
    markersize=13,
    label="Target with no valid new-run ratio: all green fill",
)

unknown_candidate_patch = mpl.lines.Line2D(
    [0], [0],
    marker="o",
    color="w",
    markerfacecolor="#FFFFFF",
    markeredgecolor="#BBBBBB",
    markeredgewidth=3,
    markersize=13,
    label="Candidate with no valid original successful-run ratio: gray outer ring",
)

ax.legend(
    handles=[
        target_new_run_patch,
        candidate_patch,
        both_patch,
        missing_new_run_patch,
        unknown_candidate_patch,
    ],
    loc="upper left",
    frameon=True,
)

ax.axis("off")
plt.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_SVG, bbox_inches="tight")

print(f"Saved: {OUT_PNG.resolve()}")
print(f"Saved: {OUT_SVG.resolve()}")
print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print(f"Target-only nodes with new-run fill where available: {len(target_only_nodes)}")
print(f"White candidate-only nodes with original successful-run outer ring: {len(candidate_only_nodes)}")
print(f"Target-and-candidate nodes with new-run fill and original successful-run outer ring: {len(target_and_candidate_nodes)}")
print(f"Candidate nodes expected to have outer ratio ring: {len(ring_nodes)}")
print(f"Target nodes expected to have new-run fill if present in CSV: {len(target_fill_nodes)}")

print("\nLowest original successful-run ratios among ring nodes:")
valid_candidate_ratios = [
    (m, candidate_ratio[m])
    for m in ring_nodes
    if m in candidate_ratio and np.isfinite(candidate_ratio[m]) and candidate_ratio[m] > 0
]
for material, ratio in sorted(valid_candidate_ratios, key=lambda x: x[1])[:15]:
    print(
        f"{material:20s} ratio={ratio:.6g}, "
        f"log10={np.log10(ratio): .3f}, "
        f"color={mpl.colors.to_hex(candidate_ring_color(material))}"
    )

print("\nHighest original successful-run ratios among ring nodes:")
for material, ratio in sorted(valid_candidate_ratios, key=lambda x: x[1], reverse=True)[:15]:
    print(
        f"{material:20s} ratio={ratio:.6g}, "
        f"log10={np.log10(ratio): .3f}, "
        f"color={mpl.colors.to_hex(candidate_ring_color(material))}"
    )

print("\nLowest new-run ratios among target nodes:")
valid_new_run_ratios = [
    (m, new_run_ratio[m])
    for m in target_fill_nodes
    if m in new_run_ratio and np.isfinite(new_run_ratio[m]) and new_run_ratio[m] > 0
]
for material, ratio in sorted(valid_new_run_ratios, key=lambda x: x[1])[:15]:
    print(
        f"{material:20s} ratio={ratio:.6g}, "
        f"log10={np.log10(ratio): .3f}, "
        f"color={mpl.colors.to_hex(target_fill_color(material))}"
    )

print("\nHighest new-run ratios among target nodes:")
for material, ratio in sorted(valid_new_run_ratios, key=lambda x: x[1], reverse=True)[:15]:
    print(
        f"{material:20s} ratio={ratio:.6g}, "
        f"log10={np.log10(ratio): .3f}, "
        f"color={mpl.colors.to_hex(target_fill_color(material))}"
    )
