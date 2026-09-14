"""Audit complete or explicitly partial frozen Linux DataLoader evidence.

No backend, framework or NumPy import. Full reports require a fresh-cache
reference verification; partial checkpoints deliberately produce no aggregates.
The identity file is an external trust anchor, not a signed attestation.
"""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def same(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    a, b = math.floor(x), math.ceil(x)
    return values[a] + (values[b] - values[a]) * (x - a)


def paired_summary(pairs):
    require(len(pairs) == 5, "five complete pairs required")
    estimates = []
    for indices in itertools.combinations_with_replacement(range(5), 5):
        weight = math.factorial(5)
        for index in range(5):
            weight //= math.factorial(indices.count(index))
        ratio = statistics.median(pairs[i][0] for i in indices) / statistics.median(pairs[i][1] for i in indices)
        estimates.extend([ratio] * weight)
    require(len(estimates) == 3125, "wrong bootstrap multiplicity")
    a, b = [statistics.median(pair[i] for pair in pairs) for i in (0, 1)]
    return {
        "reference_median": a,
        "candidate_median": b,
        "reference_over_candidate": a / b,
        "paired_bootstrap_ci95": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
        "raw_pairs": pairs,
    }


def audit(report_path, identity_path, runs_dir, overlap, partial=False, fresh_path=None):
    data, identity_data = report_path.read_bytes(), identity_path.read_bytes()
    report, identity = json.loads(data), json.loads(identity_data)
    require(
        identity["wheel_sha256"] == "2ca1875910cec38fd501af9108455609b6c926c8d14dcf570da7b9776a2e0c71",
        "wrong wheel identity",
    )
    require(identity["source_commit"] == "9c4ece56f8a15a251c36a07a0e02af238a7df81b", "wrong source revision")
    require(
        report["extension_sha256"]
        == identity["extension_sha256"]
        == "9da4432a4b3fe7c753568d82780886812360783bfbbf33e8727e1dde94f49cf2",
        "wrong extension",
    )
    require(report["source_sha256"] == identity["source_sha256"], "wrong source identity")
    require(report["corpus"] == identity["corpus"], "wrong corpus manifest")
    require(
        report["corpus"]["fingerprint"]
        == {
            "files": 9952,
            "bytes": 830426850,
            "sha256": "8b079ad2d8472ff17b6cca9b0250e831472f12ef500f2b065125f6f39406c208",
        },
        "wrong input fingerprint",
    )
    require(report["corpus"]["count"] == 5000 and report["corpus"]["task"] == "segment", "wrong corpus size/task")
    require(
        report["upstream_commit"] == identity["upstream_commit"] == "795a556942a12fe0124cf767888194a1d0b83e2e",
        "wrong upstream",
    )
    require(report["upstream_files"] == identity["upstream_files"], "wrong upstream file profile")
    plan = [
        (w, r, b)
        for w in (0, 2, 8)
        for r in range(5)
        for b in (("reference", "native") if r % 2 == 0 else ("native", "reference"))
    ]
    rows = report["results"]
    actual = [(r["workers"], r["round"], r["backend"]) for r in rows]
    require(rows and actual == plan[: len(rows)], "missing, extra or reordered samples")
    require(partial or len(rows) == 30, "incomplete requested sample count")
    raw_hashes, expected_output, expected_topology, values = {}, None, None, {}
    parent_fields = {
        "round",
        "parity",
        "sampled_family_peak_rss_bytes",
        "memory_sample_count",
        "max_sampled_processes",
        "memory_sample_max_gap_s",
    }
    for row in rows:
        w, r, b = row["workers"], row["round"], row["backend"]
        stem = f"{w}-{r}-{b}"
        raw_data = (runs_dir / (stem + ".json")).read_bytes()
        raw = json.loads(raw_data)
        require(
            canonical(raw) == canonical({k: v for k, v in row.items() if k not in parent_fields}),
            "raw/parent disagreement",
        )
        raw_hashes[stem + ".json"] = sha(raw_data)
        require(row["parity"] is True, "missing parity result")
        require(
            row["source_sha256"] == identity["source_sha256"]
            and row["extension_sha256"] == identity["extension_sha256"],
            "worker identity drift",
        )
        require(
            row["images"] == row["verification_images"] == 5000 and row["verification_batches"] == 625,
            "incomplete epoch or verification",
        )
        require(row["batch_size"] == 8 and row["imgsz"] == 640 and row["mask_ratio"] == 4, "wrong loader configuration")
        require(
            row["overlap"] is overlap and row["augment"] is False and row["image_cache"] is False,
            "wrong mask/augmentation/cache configuration",
        )
        require(
            all(
                number(row[k], 1e-12)
                for k in (
                    "epoch_s",
                    "first_batch_s",
                    "after_first_batch_s",
                    "images_per_s",
                    "after_first_batch_images_per_s",
                )
            ),
            "invalid timing",
        )
        require(same(row["epoch_s"] - row["first_batch_s"], row["after_first_batch_s"]), "wrong remainder time")
        require(same(row["images_per_s"], 5000 / row["epoch_s"]), "wrong throughput")
        require(
            same(row["after_first_batch_images_per_s"], 4992 / row["after_first_batch_s"]), "wrong remainder throughput"
        )
        arrivals = row["batch_arrivals_s"]
        require(len(arrivals) == 625 and all(number(x, 1e-12) for x in arrivals), "wrong batch arrivals")
        require(
            same(arrivals[0], row["first_batch_s"]) and sum(arrivals) <= row["epoch_s"] + 1e-9,
            "inconsistent arrival time",
        )
        require(
            same(row["batch_arrival_p50_s"], statistics.median(arrivals[1:]))
            and same(row["batch_arrival_p95_s"], quantile(arrivals[1:], 0.95)),
            "wrong arrival quantiles",
        )
        require(row["topology"]["images"] == 5000 and row["topology"]["instances"] == 36335, "wrong object population")
        expected_topology = expected_topology or row["topology"]
        require(row["topology"] == expected_topology, "topology drift")
        output = row["output_sha256"]
        require(
            isinstance(output, str) and len(output) == 64 and all(x in "0123456789abcdef" for x in output),
            "invalid output hash",
        )
        expected_output = expected_output or output
        require(output == expected_output, "full output parity mismatch")
        for k in (
            "parent_rss_before_epoch_bytes",
            "parent_peak_rss_through_epoch_bytes",
            "available_ram_before_epoch_bytes",
            "available_ram_bytes",
        ):
            require(number(row[k], 1), "invalid memory sample")
        require(
            row["parent_peak_rss_through_epoch_bytes"] >= row["parent_rss_before_epoch_bytes"],
            "invalid high-water sequence",
        )
        for k in ("loadavg_before_epoch", "loadavg"):
            require(len(row[k]) == 3 and all(number(x) for x in row[k]), "invalid load sample")
        require(
            all(
                number(row[k])
                for k in ("constructor_s_untimed", "host_swap_in_delta_epoch_bytes", "host_swap_out_delta_epoch_bytes")
            ),
            "invalid host metric",
        )
        mem_data = (runs_dir / (stem + ".memory.json")).read_bytes()
        samples = json.loads(mem_data)
        raw_hashes[stem + ".memory.json"] = sha(mem_data)
        require(
            samples
            and all(
                len(x) == 3 and number(x[0], 1) and type(x[1]) is int and x[1] > 0 and type(x[2]) is int and x[2] > 0
                for x in samples
            ),
            "invalid process-family samples",
        )
        gaps = [b[0] - a[0] for a, b in itertools.pairwise(samples)]
        require(all(x > 0 for x in gaps), "unordered memory samples")
        require(row["memory_sample_count"] == len(samples), "wrong sample count")
        require(row["sampled_family_peak_rss_bytes"] == max(x[1] for x in samples), "wrong family RSS peak")
        require(row["max_sampled_processes"] == max(x[2] for x in samples), "wrong family process count")
        require(row["memory_sample_max_gap_s"] == (max(gaps) if gaps else None), "wrong memory sampling gap")
        phase = (runs_dir / (stem + ".phase")).read_bytes()
        require(phase == b"done", "worker not terminal")
        raw_hashes[stem + ".phase"] = sha(phase)
        raw_hashes[stem + ".log"] = sha((runs_dir / (stem + ".log")).read_bytes())
        values[w, r, b] = row
    result = {
        "complete_requested_samples": len(rows) == 30,
        "overlap": overlap,
        "completed_workers": len(rows),
        "report_sha256": sha(data),
        "identity_sha256": sha(identity_data),
        "auditor_sha256": sha(Path(__file__).read_bytes()),
        "raw_sha256": raw_hashes,
        "output_sha256": expected_output,
        "limits": [
            "Audits retained artifacts; does not rerun full input fingerprinting or native code.",
            "Sampled family RSS sums shared pages and may miss peaks; maximum sampling gaps are retained.",
            "All complete worker outputs come from the separate untimed verification pass.",
            "Full aggregate intervals use exact paired enumeration, replacing the original random-bootstrap intervals.",
            "Partial checkpoints have no aggregate and do not establish cache freshness.",
        ],
    }
    if not partial:
        require(fresh_path is not None, "fresh-cache reference verification required")
        require(set(report["summary"]) == {"0", "2", "8"}, "missing aggregate groups")
        expected_files = {
            f"{w}-{r}-{b}{suffix}" for w, r, b in plan for suffix in (".json", ".memory.json", ".phase", ".log")
        }
        require({p.name for p in runs_dir.iterdir()} == expected_files, "missing or extra raw artifacts")
        fresh_data = fresh_path.read_bytes()
        fresh = json.loads(fresh_data)
        require(
            fresh["benchmark_sha256"] == sha(data)
            and fresh["matched_runs"] == 30
            and fresh["all_benchmark_outputs_match_fresh_reference"] is True,
            "fresh reference not bound to complete report",
        )
        require(
            fresh["script_sha256"] == "b17bbcbb736f867c8f066242304bd22e1f76cce95ace63b75f8adaec830e613d",
            "wrong fresh verification harness",
        )
        fr = fresh["fresh_reference"]
        fresh_runs = fresh_path.with_suffix(".runs")
        fresh_raw = (fresh_runs / "fresh-reference.json").read_bytes()
        require(canonical(json.loads(fresh_raw)) == canonical(fr), "fresh raw/parent mismatch")
        require((fresh_runs / "fresh-reference.phase").read_bytes() == b"done", "fresh reference not terminal")
        result["fresh_raw_sha256"] = {p.name: sha(p.read_bytes()) for p in fresh_runs.iterdir() if p.is_file()}
        require(
            set(result["fresh_raw_sha256"]) == {"fresh-reference.json", "fresh-reference.phase", "fresh-reference.log"},
            "wrong fresh artifacts",
        )
        require(
            fr["output_sha256"] == expected_output
            and fr["source_sha256"] == identity["source_sha256"]
            and fr["extension_sha256"] == identity["extension_sha256"],
            "fresh reference mismatch",
        )
        require(
            fr["images"] == fr["verification_images"] == 5000
            and fr["verification_batches"] == 625
            and fr["backend"] == "reference"
            and fr["workers"] == 0
            and fr["overlap"] is overlap,
            "incomplete or wrong fresh reference",
        )
        require(fresh["fresh_cache_sha256"] and len(fresh["fresh_cache_sha256"]) == 64, "missing rebuilt cache hash")
        result["fresh_reference_sha256"] = sha(fresh_data)
        metrics = ("epoch_s", "first_batch_s", "after_first_batch_s", "images_per_s", "sampled_family_peak_rss_bytes")
        result["summary"] = {}
        for w in (0, 2, 8):
            recorded = report["summary"][str(w)]
            require(set(recorded) == set(metrics), "wrong aggregate metrics")
            result["summary"][str(w)] = {}
            for metric in metrics:
                pairs = [(values[w, r, "reference"][metric], values[w, r, "native"][metric]) for r in range(5)]
                for i, b in enumerate(("reference", "native")):
                    sample = [p[i] for p in pairs]
                    require(
                        same(recorded[metric][b]["median"], statistics.median(sample))
                        and same(recorded[metric][b]["p95"], quantile(sample, 0.95)),
                        "wrong parent median/p95",
                    )
                result["summary"][str(w)][metric] = paired_summary(pairs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--overlap", choices=("yes", "no"), required=True)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "retain earlier receipts")
    result = audit(args.report, args.identity, args.runs_dir, args.overlap == "yes", args.allow_partial, args.fresh)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("completed_workers", "complete_requested_samples", "output_sha256")}))


if __name__ == "__main__":
    main()
