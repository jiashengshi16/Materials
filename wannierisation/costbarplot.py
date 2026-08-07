from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

models = [
    "Deepseekv4pro",
    "Deepseekv4flash",
    "Gemini 3.1 pro",
    "Gemini 3.5 Flash",
    "GPT 5.4 Mini",
    "KIMI3",
    "GLM 4.5 AIR",
    "Qwen3Coder Plus",
]

prices = [0.05, 0.03, 1.97, 1.78, 2.83, 1.17, 0.98, 1.10]

output_dir = Path("/Users/jshi/Documents/ICLR2027")
image_path = output_dir / "agent_model_price_per_run.png"
script_path = output_dir / "agent_model_price_chart.py"

script = '''from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

models = [
    "Deepseekv4pro",
    "Deepseekv4flash",
    "Gemini 3.1 pro",
    "Gemini 3.5 Flash",
    "GPT 5.4 Mini",
    "KIMI3",
    "GLM 4.5 AIR",
    "Qwen3Coder Plus",
]

prices = [0.05, 0.03, 1.97, 1.78, 2.83, 1.17, 0.98, 1.10]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(models, prices)

ax.set_title(
    "Avg $ per Run by Agent Model",
    fontsize=40,
    pad=18,
)

ax.set_xlabel(
    "Agent model",
    fontsize=18,
    labelpad=12,
)

ax.set_ylabel(
    "Avg $ per run",
    fontsize=20,
    labelpad=15,
)

ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.2f}"))
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

ax.tick_params(axis="x", rotation=35)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")

for bar, price in zip(bars, prices):
    ax.annotate(
        f"${price:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_ylim(0, max(prices) * 1.18)
fig.tight_layout()

output_path = Path("agent_model_price_per_run.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved chart to: {output_path.resolve()}")
'''

script_path.write_text(script, encoding="utf-8")

fig, ax = plt.subplots(figsize=(15, 8))
bars = ax.bar(models, prices)

ax.set_title(
    "Avg $ per Run by Agent Model",
    fontsize=40,
    pad=18,
    loc="left",
)

ax.set_xlabel(
    "Agent model",
    fontsize=30,
    labelpad=12,
)

ax.set_ylabel(
    "Avg $ per run",
    fontsize=40,
    labelpad=15,
)

ax.tick_params(axis="x", labelsize=20, rotation=35)
ax.tick_params(axis="y", labelsize=18)

for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")

for bar, price in zip(bars, prices):
    ax.annotate(
        f"${price:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

ax.set_ylim(0, max(prices) * 1.18)
fig.tight_layout()
output_path = Path("/Users/jshi/Documents/ICLR2027/agent_model_price_per_run.png")

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)
print(f"Image: {output_path}")
print(f"Script: {script_path}")

