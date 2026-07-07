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
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
except Exception:
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="sfdp")
    except Exception:
        pos = nx.spring_layout(G, seed=7, k=1.2, iterations=300)


# ----------------------------
# Draw
# ----------------------------
plt.figure(figsize=(18, 14))
ax = plt.gca()

ax.set_title(
    "Material similarity dependency graph\n"
    "Inner fill = avg new-run error ratio, Green outline = target material, "
    "Outer ring = original successful-run ratio",
    fontsize=15,
    pad=18,
)

# Edges: target -> candidate
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=13,
    width=1.2,
    alpha=0.45,
    connectionstyle="arc3,rad=0.08",
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
