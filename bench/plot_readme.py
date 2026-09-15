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


def rfdetr_loader():
    data = load("rfdetr-loader-linux-v1.json")
    single, loader = data["single_process"], data["dataloader_img_per_s"]
    ref_stages, acc_stages = single["reference_stages_ms"], single["accelerated_stages_ms"]
    ref_total, acc_total = min(single["reference_ms"]), min(single["accelerated_ms"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    # Left: one sample, stacked by stage. The reference's mask work is inside prepare and transforms.
    labels = ["RF-DETR reference", "with ultrafast-maskops"]
    decode = [ref_total - ref_stages["prepare"] - ref_stages["transforms"],
              acc_total - acc_stages["prepare"] - acc_stages["transforms"] - acc_stages["materialize"]]
    parts = [
        ("JPEG decode", decode, ["#4b5563", "#c2255c"]),
        ("prepare (ConvertCoco)", [ref_stages["prepare"], acc_stages["prepare"]], ["#6b7280", "#e64980"]),
        ("transforms", [ref_stages["transforms"], acc_stages["transforms"]], [REFERENCE, "#f783ac"]),
        ("materialize masks", [0.0, acc_stages["materialize"]], ["#ffffff", "#ffd8e6"]),
    ]
    bottoms = [0.0, 0.0]
    for name, values, colors in parts:
        for i, (v, c) in enumerate(zip(values, colors)):
            if v:
                ax1.bar(i, v, 0.55, bottom=bottoms[i], color=c, edgecolor="white", linewidth=0.5,
                        label=name if i == 1 or name != "materialize masks" else None)
                if v > 0.6:
                    ax1.text(i, bottoms[i] + v / 2, f"{v:.1f}", ha="center", va="center", color="white", fontsize=9)
            bottoms[i] += v
    for i, total in enumerate((ref_total, acc_total)):
        ax1.text(i, bottoms[i] + ref_total * 0.02, f"{total:.1f} ms", ha="center", va="bottom",
                 color="#5f6368" if i == 0 else OURS, fontweight="normal" if i == 0 else "bold")
    ax1.text(1, bottoms[1] + ref_total * 0.13, f"{ref_total / acc_total:.2f}× faster", ha="center", va="bottom", color=OURS,
             fontsize=12, fontweight="bold")
    ax1.set_xticks([0, 1], labels)
    ax1.set_ylim(0, ref_total * 1.3)
    ax1.set_ylabel("ms per training sample (single process)")
    ax1.set_title("RF-DETR segmentation sample: decode, ConvertCoco, transforms", loc="left", fontweight="bold", fontsize=10.5)
    handles, names = ax1.get_legend_handles_labels()
    seen = {}
    for h, n in zip(handles, names):
        seen.setdefault(n, h)
    ax1.legend(seen.values(), seen.keys(), frameon=False, loc="upper right", fontsize=9)
    # Right: DataLoader throughput.
    values = [loader["reference"], loader["accelerated"]]
    bars = ax2.bar([0, 1], values, 0.55, color=[REFERENCE, OURS])
    top = max(values)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02, f"{v:.0f} img/s", ha="center", va="bottom",
                 color="#5f6368" if i == 0 else OURS, fontweight="normal" if i == 0 else "bold")
    ax2.text(1, values[1] + top * 0.13, f"{values[1] / values[0]:.2f}× faster", ha="center", va="bottom", color=OURS, fontsize=12,
             fontweight="bold")
    ax2.set_xticks([0, 1], labels)
    ax2.set_ylim(0, top * 1.3)
    ax2.set_ylabel(f"images per second, {loader['workers']} workers, batch {loader['batch']}")
    ax2.set_title("DataLoader throughput", loc="left", fontweight="bold", fontsize=10.5)
    fig.text(0.01, 0.005, f"Intel i5-10400, COCO val2017, RF-DETR default training transforms (square_resize_div_64, multi_scale), resolution {data['resolution']}.\n"
             f"{data['samples']} samples ({data['instances']} instances) drawn under the same seeds: images, boxes, labels and masks byte-identical.",
             fontsize=9, color="#5f6368")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ASSETS / "rfdetr-loader.svg")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    mask_stage()
    loader_stages()
    coco_epoch()
    rfdetr_loader()
    print("wrote", sorted(p.name for p in ASSETS.glob("*.svg")))
