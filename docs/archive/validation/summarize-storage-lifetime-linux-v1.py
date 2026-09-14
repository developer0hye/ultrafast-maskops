"""Read raw diagnostic evidence; preserve failures instead of declaring qualification."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root, name):
    path = root / (name + ".json")
    data = json.loads(path.read_text())
    assert data["complete"]
    assert data["args"]["reset_point"] == "first-batch" and data["args"]["cycles"] == 6
    assert data["sharing_strategy"] == "file_system" and data["context"] == "spawn"
    assert data["workers"] == 2 and data["prefetch_factor"] == 2 and data["batch_size"] == 4
    assert data["performance_measurement"] is False
    assert 1 <= len(data["cycles"]) <= 6
    if data["protocol_passed"]:
        assert len(data["cycles"]) == 6
    else:
        assert data.get("error") and data.get("cleanup_error")
    traces = path.with_suffix(".traces")
    retired = []
    observations = []

    def verify_probe(probe, scope_pids):
        # v1 returns the mutable retired_pids list; subsequent appends changed
        # historical PID fields. Frozen per-probe trace hashes retain the scope.
        # Preserve this defect explicitly; never rewrite raw reports.
        assert probe["worker_pids"][: len(scope_pids)] == scope_pids
        records = {}
        for pid in scope_pids:
            trace = traces / f"{pid}.jsonl"
            if not trace.exists():
                assert trace.name not in probe["trace_sha256"]
                continue
            assert sha(trace) == probe["trace_sha256"][trace.name]
            for line in trace.read_text().splitlines():
                item = json.loads(line)
                assert item["pid"] == pid and item["handle"].startswith(f"/torch_{pid}_")
                records[item["handle"]] = item["storage_bytes"]
        assert len(records) == probe["observed_handles"]
        assert len({i["handle"] for i in probe["present"]}) == len(probe["present"])
        for item in probe["present"]:
            assert records[item["handle"]] == item["storage_bytes"]
        assert sum(i["storage_bytes"] for i in probe["present"]) == probe["present_storage_bytes"]
        return {
            "observed_handles": len(records),
            "present_handles": len(probe["present"]),
            "present_storage_bytes": probe["present_storage_bytes"],
        }

    for index, cycle in enumerate(data["cycles"]):
        assert cycle["index"] == index and cycle["verified_batches"] == 1
        old = cycle["old_workers"]
        assert len(old) == 2 and all(w["alive"] is False for w in old)
        if any(w["exitcode"] != 0 for w in old):
            assert not data["protocol_passed"] and index == len(data["cycles"]) - 1
        assert [w["pid"] for w in old] == cycle["old_worker_pids"]
        assert not set(retired) & set(cycle["old_worker_pids"])
        retired.extend(cycle["old_worker_pids"])
        if "retired_handles_after_reset" in cycle:
            assert not set(retired) & set(cycle["new_worker_pids"])
            observations.append(verify_probe(cycle["retired_handles_after_reset"], retired.copy()))
        else:
            assert not data["protocol_passed"]
    assert len(data["final_workers"]) == 2
    assert all(not w["alive"] for w in data["final_workers"])
    if data["protocol_passed"]:
        assert all(w["exitcode"] == 0 for w in data["final_workers"])
    final_scope = retired + [w["pid"] for w in data["final_workers"] if w["pid"] not in retired]
    final = verify_probe(data["handles_after_close_plus_3s"], final_scope)
    verify_probe(data["handles_after_close"], final_scope)
    result = {
        "name": name,
        "backend": data["args"]["backend"],
        "report_sha256": sha(path),
        "source_sha256": data["source_sha256"],
        "cycles": observations,
        "body_protocol_passed": data["protocol_passed"],
        "requested_cycles": 6,
        "completed_resets": len(observations),
        "worker_exitcodes": [w["exitcode"] for c in data["cycles"] for w in c["old_workers"]],
        "historical_probe_pid_list_alias_bug": True,
        "after_close_plus_3s": final,
    }
    process = path.with_suffix(".process.json")
    if process.exists():
        record = json.loads(process.read_text())
        assert record["report_sha256"] == sha(path) and record["log_sha256"] == sha(path.with_suffix(".log"))
        assert record["complete"] and not record["passed"] and record["timeout"]
        assert record["returncode"] == -15
        assert record["body_protocol_passed"] == data["protocol_passed"]
        assert record["handles_after_process_exit"]["present"] == []
        assert all(p.get("missing") or p.get("state") == "Z" for p in record["after_termination"])
        assert "resource_tracker.py" in path.with_suffix(".log").read_text()
        result["process"] = {
            "passed": False,
            "timeout": True,
            "returncode": -15,
            "post_termination_handles": 0,
            "receipt_sha256": sha(process),
        }
    else:
        record = json.loads((root / (name + "-post-termination.json")).read_text())
        assert record["all_related_terminal"] and record["handles"]["present"] == []
        result["process"] = {
            "passed": False,
            "supervision": "manual; original attach denied and cleanup-observer race preserved",
            "post_termination_handles": 0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    names = [
        "mask-storage-lifetime-linux-v1-reference-first-batch",
        "mask-storage-lifetime-linux-v2-native-first-batch",
        "mask-storage-lifetime-linux-v2-reference-first-batch",
    ]
    reports = [summarize(args.root, name) for name in names]
    assert all(r["source_sha256"] == reports[0]["source_sha256"] for r in reports)
    result = {
        "complete": True,
        "qualification_passed": False,
        "reports": reports,
        "scope": "Observed Linux file_system failures; instrumented small synthetic data, no speed/RSS or macOS claim.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "qualification_passed": False, "reports": len(reports)}))


if __name__ == "__main__":
    main()
