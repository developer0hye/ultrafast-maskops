"""Render the README charts from the recorded measurements in bench/results.

    python bench/plot_readme.py            # writes docs/assets/*.svg

Every number comes from a JSON report checked into bench/results; nothing is
typed in here.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results"
ASSETS = ROOT / "docs/assets"
REFERENCE, OURS, SECOND = "#9aa0a6", "#d6336c", "#f59f00"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)


def load(name):
    return json.load(open(RESULTS / name))


def loader_medians(name):
    runs = load(name)["results"]
    out = {}
    for backend in ("reference", "geometry+masks"):
        mine = [r for r in runs if r["backend"] == backend]
        samples = mine[0]["samples"]
        out[backend] = {
            "total": statistics.median(r["mean_ms"] for r in mine),
            **{
                stage: statistics.median(r["stages_s"].get(stage, 0.0) for r in mine) * 1e3 / samples
                for stage in ("resample", "apply_segments", "masks")
            },
        }
    return out


def bar_pair(ax, labels, reference, ours, unit, title, note=None):
    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], reference, width, color=REFERENCE, label="Ultralytics reference")
    ax.bar([i + width / 2 for i in x], ours, width, color=OURS, label="ultrafast-maskops")
    top = max(reference)
    for i, (r, o) in enumerate(zip(reference, ours)):
        ax.text(i - width / 2, r + top * 0.02, f"{r:,.0f}", ha="center", va="bottom", color="#5f6368")
        ax.text(i + width / 2, o + top * 0.02, f"{o:,.0f}", ha="center", va="bottom", color=OURS, fontweight="bold")
        ax.text(i + width / 2, o + top * 0.13, f"{r / o:.1f}× faster", ha="center", va="bottom", color=OURS, fontsize=12,
                fontweight="bold")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel(unit)
    ax.set_ylim(0, top * 1.3)
    ax.set_title(title, loc="left", fontweight="bold")
    if note:
        ax.text(0, -0.22, note, transform=ax.transAxes, fontsize=9, color="#5f6368", va="top")


def mask_stage():
    m2, linux = load("mask-stage-sampled-m2-v1.json"), load("mask-stage-minimal-linux-v1.json")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_pair(
        ax,
        ["Apple M2", "Intel i5-10400"],
        [m2["reference"]["us_per_sample_median"], linux["reference"]["us_per_sample_median"]],
        [m2["sampled"]["us_per_sample_median"], linux["sampled"]["us_per_sample_median"]],
        "µs per Format call (13.7 instances)",
        "Mask rasterization (polygons2masks_overlap), 2,000 captured COCO calls",
        "Byte-identical output on every call. Medians of five alternating rounds\n(M2 at commit 490eed2, i5-10400 at 5d2d8b2).",
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "mask-stage.svg")
    plt.close(fig)


def loader_stages():
    hosts = {"Apple M2": loader_medians("geometry-loader-sampled-m2-v1.json"),
             "Intel i5-10400": loader_medians("geometry-loader-sampled-linux-v1.json")}
    stages = [("resample", "resample\nsegments"), ("apply_segments", "RandomPerspective\napply_segments"),
              ("masks", "Format\n_format_segments")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, (host, data) in zip(axes, hosts.items()):
        labels = [s[1] for s in stages]
        ref = [data["reference"][s[0]] for s in stages]
        ours = [data["geometry+masks"][s[0]] for s in stages]
        x = range(len(labels))
        width = 0.38
        ax.bar([i - width / 2 for i in x], ref, width, color=REFERENCE, label="Ultralytics reference")
        ax.bar([i + width / 2 for i in x], ours, width, color=OURS, label="ultrafast-maskops")
        top = max(ref)
        for i, (r, o) in enumerate(zip(ref, ours)):
            ax.text(i - width / 2, r + top * 0.02, f"{r:.2f}", ha="center", va="bottom", color="#5f6368", fontsize=9)
            ax.text(i + width / 2, o + top * 0.02, f"{o:.2f}", ha="center", va="bottom", color=OURS, fontsize=9, fontweight="bold")
            ax.text(i + width / 2, o + top * 0.12, f"{r / o:.1f}×", ha="center", va="bottom", color=OURS, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x), labels, fontsize=9)
        ax.set_ylim(0, top * 1.3)
        ax.set_ylabel("ms per augmented sample")
        total_r, total_o = data["reference"]["total"], data["geometry+masks"]["total"]
        ax.set_title(f"{host}: __getitem__ {total_r:.2f} → {total_o:.2f} ms ({total_r / total_o:.2f}×)", loc="left", fontweight="bold", fontsize=10.5)
    axes[0].legend(frameon=False, loc="upper left")
    fig.text(0.01, 0.005, "The three segment stages of one augmented COCO val2017 sample: single process, 1,000 samples, alternating rounds.\n"
             "The digest of every sample (image, classes, boxes, masks) is identical between the two.", fontsize=9, color="#5f6368")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ASSETS / "loader-stages.svg")
    plt.close(fig)


def coco_epoch():
    both = load("coco-epoch-hit-linux-v1.json")["summary"]
    only = load("coco-epoch-hit-maskops-linux-v1.json")["summary"]
    labels = ["Ultralytics\nreference", "ultrafast-maskops", "ultrafast-maskops +\nultrafast-yolo-dataset"]
    colors = [REFERENCE, OURS, SECOND]
    seconds = [only["reference"]["total_s"]["median"], only["maskops"]["total_s"]["median"], both["accelerated"]["total_s"]["median"]]
    memory = [only["reference"]["peak_memory_bytes"]["median"] / 1e9, only["maskops"]["peak_memory_bytes"]["median"] / 1e9,
              both["accelerated"]["peak_memory_bytes"]["median"] / 1e9]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, values, unit, title, fmt in (
        (ax1, seconds, "seconds: constructor + one epoch", "One augmented training epoch, 118,287 images", "{:.0f} s"),
        (ax2, memory, "peak memory, process + 8 workers (PSS)", "Memory during the epoch", "{:.2f} GB"),
    ):
        bars = ax.bar(range(3), values, 0.6, color=colors)
        top = max(values)
        for i, (bar, v) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center", va="bottom",
                    color="#5f6368" if i == 0 else colors[i], fontweight="normal" if i == 0 else "bold")
            if i:
                gain = f"{values[0] / v:.2f}× faster" if unit.startswith("seconds") else f"−{100 * (1 - v / values[0]):.0f}%"
                ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.12, gain, ha="center", va="bottom", color=colors[i],
                        fontsize=12, fontweight="bold")
        ax.set_xticks(range(3), labels, fontsize=9.5)
        ax.set_ylim(0, top * 1.3)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.text(0.01, 0.005, "Intel i5-10400, 8 DataLoader workers, batch 16, imgsz 640, default augmentations, label cache hit; identical batches.\n"
             "Time gains come from ultrafast-maskops; the memory reduction comes from ultrafast-yolo-dataset's packed labels.",
             fontsize=9, color="#5f6368")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ASSETS / "coco-epoch.svg")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    mask_stage()
    loader_stages()
    coco_epoch()
    print("wrote", sorted(p.name for p in ASSETS.glob("*.svg")))
