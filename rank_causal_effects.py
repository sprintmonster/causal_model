#!/usr/bin/env python3
"""Rank direct causal contributors for a target node using do-interventions.

This is a lightweight SCM-style analysis built on top of the current DAG.
For a chosen target, each direct parent is intervened on individually and the
change in P(target) is measured.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_causal_model import infer_marginals, resolve_node_id

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL = WORKSPACE / "final_graph" / "causal_model.json"
DEFAULT_OUT = WORKSPACE / "final_graph" / "causal_inference_report.json"


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_parent_index(model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    parents: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in model.get("nodes", []):
        for parent in node.get("parents", []):
            parents[node["id"]].append(parent)
    return parents


def select_target(model: dict[str, Any], target: str | None) -> str:
    if target:
        return resolve_node_id(model, target)

    parent_index = build_parent_index(model)
    candidates = [
        (len(parents), node_id)
        for node_id, parents in parent_index.items()
        if len(parents) >= 2
    ]
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    # Fallback to prohibition convention if the graph is sparse.
    return resolve_node_id(model, "event:prohibition_convention")


def intervention_report(model: dict[str, Any], target_id: str) -> dict[str, Any]:
    baseline = infer_marginals(model)
    parent_index = build_parent_index(model)
    target_node = next(node for node in model.get("nodes", []) if node.get("id") == target_id)
    parents = sorted(parent_index.get(target_id, []), key=lambda p: p["id"])

    ranked = []
    for parent in parents:
        pid = parent["id"]
        on = infer_marginals(model, intervention={pid: 1.0})
        off = infer_marginals(model, intervention={pid: 0.0})
        ranked.append(
            {
                "node_id": pid,
                "label": next((n.get("label", pid) for n in model.get("nodes", []) if n.get("id") == pid), pid),
                "edge_probability": float(parent.get("edge_probability", 0.0) or 0.0),
                "baseline_target_probability": round(baseline[target_id], 6),
                "do1_target_probability": round(on[target_id], 6),
                "do0_target_probability": round(off[target_id], 6),
                "effect_do1_minus_baseline": round(on[target_id] - baseline[target_id], 6),
                "effect_baseline_minus_do0": round(baseline[target_id] - off[target_id], 6),
                "parent_probability_baseline": round(baseline[pid], 6),
            }
        )

    ranked.sort(key=lambda x: (x["effect_do1_minus_baseline"], x["do1_target_probability"]), reverse=True)

    return {
        "target": {
            "node_id": target_node["id"],
            "label": target_node.get("label", target_node["id"]),
            "type": target_node.get("type"),
        },
        "baseline": {"P(target)": round(baseline[target_id], 6)},
        "direct_parents": len(parents),
        "ranking": ranked,
        "method": {
            "model": "binary noisy-or DAG",
            "intervention": "do(parent=1.0) and do(parent=0.0) for each direct parent",
            "rank_key": "effect_do1_minus_baseline",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--target", default="")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    model = load_model(args.model)
    target_id = select_target(model, args.target or None)
    report = intervention_report(model, target_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[target] {report['target']['node_id']} ({report['target']['label']})")
    print(f"[baseline] P(target)={report['baseline']['P(target)']:.6f}")
    if report["ranking"]:
        top = report["ranking"][0]
        print(
            f"[top] {top['label']} ({top['node_id']}) "
            f"do(1)->{top['do1_target_probability']:.6f} "
            f"delta={top['effect_do1_minus_baseline']:.6f}"
        )
    else:
        print("[top] no direct parents found")
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())