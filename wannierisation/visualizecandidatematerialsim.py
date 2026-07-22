from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

CANDIDATES_CSV_PATH = Path(
    "/home/jiasheng/WannierisationBenchmarking/include_only_candidates.csv"
)
ERROR_RATIOS_CSV_PATH = Path(
    "/home/jiasheng/WannierisationBenchmarking/jobsGeminiReviewsDeepseek/"
    "ChemSimReruns/all_error_ratios_by_material.csv"
)

OUT_PNG = Path(
    "/home/jiasheng/WannierisationBenchmarking/jobsGeminiReviewsDeepseek/"
    "ChemSimReruns/material_dependency_graph.png"
)
OUT_SVG = Path(
    "/home/jiasheng/WannierisationBenchmarking/jobsGeminiReviewsDeepseek/"
    "ChemSimReruns/material_dependency_graph.svg"
)
LOWER_OUT_PNG = Path(
    "/home/jiasheng/WannierisationBenchmarking/jobsGeminiReviewsDeepseek/"
    "ChemSimReruns/material_dependency_graph_lower_error_ratio.png"
)

TARGET_GREEN = "#7BC96F"
CANDIDATE_ONLY_FILL_SIZE = 950
TARGET_BASE_SIZE = 1050
CANDIDATE_ONLY_RING_SIZE = 1060
TARGET_AND_CANDIDATE_RING_SIZE = 1160
RATIO_RING_LINEWIDTH = 2.8
MISSING_RING_COLOR = "#BBBBBB"
RING_BACKING_COLOR = "#777777"


def material_ratio_dict(
    df: pd.DataFrame,
    material_col: str,
    ratio_col: str,
    agg: str = "mean",
) -> dict:
    """Return one finite positive ratio per material, using mean or min aggregation."""
    valid = (
        df[[material_col, ratio_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[material_col, ratio_col])
    )
    valid = valid[valid[ratio_col] > 0]

    if agg == "mean":
        grouped = valid.groupby(material_col)[ratio_col].mean()
    elif agg == "min":
        grouped = valid.groupby(material_col)[ratio_col].min()
    else:
        raise ValueError(f"Unsupported aggregation {agg!r}; use 'mean' or 'min'.")

    return grouped.dropna().to_dict()


def comparison_ratio_dict(
    original_ratio_by_material: dict,
    new_ratio_by_material: dict,
) -> dict:
    """
    Return new/original for each material.

    Interpretation:
      - 1.0  => no change (white)
      - <1.0 => new run is better / lower error (blue)
      - >1.0 => new run is worse / higher error (red)
    """
    out = {}
    for material in sorted(set(original_ratio_by_material) | set(new_ratio_by_material)):
        original = original_ratio_by_material.get(material)
        new = new_ratio_by_material.get(material)
        if (
            original is None
            or new is None
            or not np.isfinite(original)
            or not np.isfinite(new)
            or original <= 0
            or new <= 0
        ):
            continue
        out[material] = float(new) / float(original)
    return out


def make_log_ratio_norm(
    ratio_by_material: dict,
    materials,
    percentile: float = 95,
    linthresh: float = 0.05,
) -> mpl.colors.SymLogNorm:
    """
    Symmetric log color norm around ratio = 1.

    Uses x = log10(ratio).
    linthresh controls how wide the near-white region around ratio=1 is.
    Smaller linthresh = faster transition from white to red/blue.
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
        max_abs = 1.0
    else:
        max_abs = float(np.nanpercentile(np.abs(logs), percentile))
        max_abs = max(max_abs, linthresh * 1.01)

    return mpl.colors.SymLogNorm(
        linthresh=linthresh,
        linscale=0.35,
        vmin=-max_abs,
        vmax=max_abs,
        base=10,
    )


def set_ratio_colorbar_ticks(cbar, norm):
    """Place readable ratio ticks on a log10-ratio colorbar."""
    max_abs = max(abs(float(norm.vmin)), abs(float(norm.vmax)))

    tick_logs = np.array([
        -max_abs,
        -max_abs / 2,
        0.0,
        max_abs / 2,
        max_abs,
    ])

    cbar.set_ticks(tick_logs)
    cbar.set_ticklabels([f"{10 ** t:.3g}" for t in tick_logs])


# ----------------------------
# Load data
# ----------------------------
candidate_pairs_df = pd.read_csv(CANDIDATES_CSV_PATH)
error_ratios_df = pd.read_csv(ERROR_RATIOS_CSV_PATH)

candidate_required_cols = {"target_material", "candidate_material"}
missing_candidate_cols = candidate_required_cols - set(candidate_pairs_df.columns)
if missing_candidate_cols:
    raise ValueError(
        f"Candidate CSV missing required columns: {missing_candidate_cols}. "
        f"Available columns: {list(candidate_pairs_df.columns)}"
    )

# The candidate CSV contains blank separator rows. Keep only actual
# target -> candidate pairs and normalize material names to strings.
candidate_pairs_df = (
    candidate_pairs_df[["target_material", "candidate_material"]]
    .dropna(subset=["target_material", "candidate_material"])
    .copy()
)
candidate_pairs_df["target_material"] = (
    candidate_pairs_df["target_material"].astype(str).str.strip()
)
candidate_pairs_df["candidate_material"] = (
    candidate_pairs_df["candidate_material"].astype(str).str.strip()
)
candidate_pairs_df = candidate_pairs_df[
    (candidate_pairs_df["target_material"] != "")
    & (candidate_pairs_df["candidate_material"] != "")
].drop_duplicates()


def first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Return the first matching column name, with a useful error if none exist."""
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find a {label} column in {ERROR_RATIOS_CSV_PATH}. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


if "material" not in error_ratios_df.columns:
    raise ValueError(
        f"{ERROR_RATIOS_CSV_PATH} must contain a 'material' column. "
        f"Available columns: {list(error_ratios_df.columns)}"
    )

# Both original-run and new-run ratios come from the same CSV.
original_run_ratio_col = first_existing_column(
    error_ratios_df,
    [
        "avg_original_run_error_ratio",
        "original_run_error_ratio",
        "avg_original_error_ratio",
        "original_error_ratio",
        "gemini_to_reference_ratio",
    ],
    "original-run error-ratio",
)
new_run_ratio_col = first_existing_column(
    error_ratios_df,
    [
        "avg_new_run_error_ratio",
        "new_run_error_ratio",
        "avg_new_error_ratio",
        "new_error_ratio",
    ],
    "new-run error-ratio",
)

print(f"Using original-run ratio column: {original_run_ratio_col}")
print(f"Using new-run ratio column: {new_run_ratio_col}")

# Average per-material values.
original_ratio_mean = material_ratio_dict(
    error_ratios_df, "material", original_run_ratio_col, agg="mean"
)
new_ratio_mean = material_ratio_dict(
    error_ratios_df, "material", new_run_ratio_col, agg="mean"
)

# Best (lowest) per-material values.
original_ratio_best = material_ratio_dict(
    error_ratios_df, "material", original_run_ratio_col, agg="min"
)
new_ratio_best = material_ratio_dict(
    error_ratios_df, "material", new_run_ratio_col, agg="min"
)

average_comparison_ratio = comparison_ratio_dict(original_ratio_mean, new_ratio_mean)
best_comparison_ratio = comparison_ratio_dict(original_ratio_best, new_ratio_best)


# ----------------------------
# Build directed graph
# ----------------------------
G = nx.DiGraph()

target_nodes = set(candidate_pairs_df["target_material"])
candidate_nodes = set(candidate_pairs_df["candidate_material"])

for row in candidate_pairs_df.itertuples(index=False):
    target = row.target_material
    cand = row.candidate_material
    G.add_node(target)
    G.add_node(cand)
    G.add_edge(target, cand)

target_only_nodes = sorted(target_nodes - candidate_nodes)
candidate_only_nodes = sorted(candidate_nodes - target_nodes)
target_and_candidate_nodes = sorted(target_nodes & candidate_nodes)
all_nodes = set(G.nodes)
ring_nodes = sorted(all_nodes)

blue_white_red = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_white_red",
    ["#08306B", "#9ECAE1", "#FFFFFF", "#FCAE91", "#99000D"],
)

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
        return max(TARGET_AND_CANDIDATE_RING_SIZE, TARGET_BASE_SIZE)
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

        private = sorted(private, key=lambda n: (old_angle(n), n))

        idx = 0
        ring = 0
        while idx < len(private):
            count = min(max_per_ring + 4 * ring, len(private) - idx)
            radius = base_radius + ring * ring_gap
            offset = start_angle + ring * np.deg2rad(17.0)
            for k in range(count):
                theta = offset + 2.0 * np.pi * k / count
                node = private[idx]
                pos[node] = center + radius * np.array([np.cos(theta), np.sin(theta)])
                idx += 1
            ring += 1

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

                multiplier = 1.65 if w in target_nodes else 1.0
                required = multiplier * radii[w] + gap
                if dist < required:
                    normal = _unit(p - closest, fallback_angle=rng.uniform(0, 2 * np.pi))
                    push = 0.34 * (required - dist) * normal
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
    span = max(
        float(coords[:, 0].max() - coords[:, 0].min()),
        float(coords[:, 1].max() - coords[:, 1].min()),
        1.0,
    )
    coords *= target_span / span
    for n, xy in zip(nodes, coords):
        pos[n] = xy


# Build the layout.
_normalize_layout(pos, target_span=1000.0)
_spread_targets(pos, min_dist_frac=0.20, iterations=1200)
_initial_radial_candidate_layout(pos, pos0, base_radius_frac=0.120, ring_gap_frac=0.078, max_per_ring=7)
_hard_resolve_node_collisions(pos, min_gap_frac=0.021, iterations=3500)
_straight_edge_clearance_pass(pos, iterations=1200, clearance_gap_frac=0.016)
_hard_resolve_node_collisions(pos, min_gap_frac=0.024, iterations=4000)
_normalize_layout(pos, target_span=1000.0)
pos = {n: tuple(xy) for n, xy in pos.items()}


# ----------------------------
# Draw
# ----------------------------
def draw_graph(
    comparison_ratio: dict,
    out_png: Path,
    out_svg: Path | None = None,
    aggregation_label: str = "average",
):
    comparison_norm = make_log_ratio_norm(comparison_ratio, ring_nodes)

    def outer_ring_color(material: str):
        r = comparison_ratio.get(material)
        if r is None or not np.isfinite(r) or r <= 0:
            return MISSING_RING_COLOR
        return blue_white_red(comparison_norm(np.log10(r)))

    plt.figure(figsize=(22, 16))
    ax = plt.gca()

    ax.set_title(
        "Material similarity dependency graph\n"
        f"Outer ring = {aggregation_label} new-run/original-run error ratio "
        "(white = no change, blue = better, red = worse)",
        fontsize=15,
        pad=18,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=False,
        width=1.25,
        alpha=0.55,
        edge_color="#666666",
    )

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

    def outer_ring_size(material: str) -> int:
        if material in target_nodes:
            return TARGET_AND_CANDIDATE_RING_SIZE
        return CANDIDATE_ONLY_RING_SIZE

    outer_ring_sizes = [outer_ring_size(n) for n in ring_nodes]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=ring_nodes,
        node_color="none",
        edgecolors=RING_BACKING_COLOR,
        linewidths=RATIO_RING_LINEWIDTH + 1.2,
        node_size=[s + 90 for s in outer_ring_sizes],
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=ring_nodes,
        node_color="none",
        edgecolors=[outer_ring_color(n) for n in ring_nodes],
        linewidths=RATIO_RING_LINEWIDTH,
        node_size=outer_ring_sizes,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    comparison_sm = mpl.cm.ScalarMappable(norm=comparison_norm, cmap=blue_white_red)
    comparison_sm.set_array([])

    comparison_cbar = plt.colorbar(comparison_sm, ax=ax, shrink=0.72, pad=0.02)
    set_ratio_colorbar_ticks(comparison_cbar, comparison_norm)
    comparison_cbar.set_label(
        f"Outer ring: {aggregation_label} new-run/original-run error ratio "
        "(1 = same, <1 = better, >1 = worse)",
        fontsize=11,
    )

    target_patch = mpl.lines.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=TARGET_GREEN,
        markeredgecolor="#99000D",
        markeredgewidth=3,
        markersize=13,
        label="Target material: green fill, outer ring colored by new-vs-original difference",
    )

    candidate_patch = mpl.lines.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="#FFFFFF",
        markeredgecolor="#99000D",
        markeredgewidth=3,
        markersize=13,
        label="Candidate-only material: white fill, outer ring colored by new-vs-original difference",
    )

    both_patch = mpl.lines.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=TARGET_GREEN,
        markeredgecolor="#99000D",
        markeredgewidth=3,
        markersize=13,
        label="Target + candidate: green fill, outer ring colored by new-vs-original difference",
    )

    unknown_patch = mpl.lines.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="#FFFFFF",
        markeredgecolor=MISSING_RING_COLOR,
        markeredgewidth=3,
        markersize=13,
        label="No valid original/new comparison available: gray outer ring",
    )

    ax.legend(
        handles=[target_patch, candidate_patch, both_patch, unknown_patch],
        loc="upper left",
        frameon=True,
    )

    ax.axis("off")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_svg, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png.resolve()}")
    if out_svg is not None:
        print(f"Saved: {out_svg.resolve()}")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"Target-only nodes: {len(target_only_nodes)}")
    print(f"Candidate-only nodes: {len(candidate_only_nodes)}")
    print(f"Target-and-candidate nodes: {len(target_and_candidate_nodes)}")

    valid_ring_nodes = [
        m for m in ring_nodes
        if m in comparison_ratio and np.isfinite(comparison_ratio[m]) and comparison_ratio[m] > 0
    ]
    missing_ring_nodes = [
        m for m in ring_nodes
        if m not in comparison_ratio or not np.isfinite(comparison_ratio.get(m, np.nan)) or comparison_ratio.get(m, np.nan) <= 0
    ]
    print(f"Nodes with valid {aggregation_label} comparison ratio: {len(valid_ring_nodes)} / {len(ring_nodes)}")
    if missing_ring_nodes:
        print(
            f"Nodes missing valid {aggregation_label} comparison ratio: "
            + ", ".join(missing_ring_nodes)
        )

    valid_comparison_ratios = [
        (m, comparison_ratio[m])
        for m in ring_nodes
        if m in comparison_ratio and np.isfinite(comparison_ratio[m]) and comparison_ratio[m] > 0
    ]

    print(f"\nMost improved materials by {aggregation_label} comparison ratio (new/original < 1):")
    for material, ratio in sorted(valid_comparison_ratios, key=lambda x: x[1])[:15]:
        print(
            f"{material:20s} new/original={ratio:.6g}, "
            f"log10={np.log10(ratio): .3f}, "
            f"color={mpl.colors.to_hex(outer_ring_color(material))}"
        )

    print(f"\nMost worsened materials by {aggregation_label} comparison ratio (new/original > 1):")
    for material, ratio in sorted(valid_comparison_ratios, key=lambda x: x[1], reverse=True)[:15]:
        print(
            f"{material:20s} new/original={ratio:.6g}, "
            f"log10={np.log10(ratio): .3f}, "
            f"color={mpl.colors.to_hex(outer_ring_color(material))}"
        )


# Average comparison graph.
draw_graph(
    comparison_ratio=average_comparison_ratio,
    out_png=OUT_PNG,
    out_svg=OUT_SVG,
    aggregation_label="average",
)

# Best/lower comparison graph.
draw_graph(
    comparison_ratio=best_comparison_ratio,
    out_png=LOWER_OUT_PNG,
    out_svg=None,
    aggregation_label="best",
)
