#!/usr/bin/env python3
"""Tiny runner: evaluate direct-parent interventions on a selected target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_causal_model import build_intervention_report, infer_marginals

WORKSPACE = Path(__file__).resolve().parent
MODEL_PATH = WORKSPACE / "final_graph" / "causal_model.json"
OUT_PATH = WORKSPACE / "final_graph" / "causal_demo_top_effects.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--target", default="topic:prohibition")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    model = json.loads(args.model.read_text(encoding="utf-8"))
    node_map = {n["id"]: n for n in model["nodes"]}

    if args.target not in node_map:
        raise SystemExit(f"target not found in model: {args.target}")

    target_node = node_map[args.target]
    parents = target_node.get("parents", [])
    if not parents:
        raise SystemExit(f"target has no direct parents: {args.target}")

    baseline = infer_marginals(model)
    out = []
    for p in parents:
        pid = p["id"]
        rep = build_intervention_report(
            model=model,
            target=args.target,
            intervene_node=pid,
            intervene_value=1.0,
        )
        out.append(
            {
                "intervene_node": pid,
                "edge_probability": p.get("edge_probability", 0.0),
                "baseline_target": baseline[args.target],
                "post_target": rep["post_intervention"]["P(target)"],
                "absolute_delta": rep["effect"]["absolute_delta"],
            }
        )

    out.sort(key=lambda x: x["absolute_delta"], reverse=True)
    out = out[: args.top]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"target": args.target, "top_effects": out}, ensure_ascii=False, indent=2), encoding="utf-8")

    if out:
        top = out[0]
        print(
            f"[top effect] do({top['intervene_node']}=1) => "
            f"P({args.target}) delta={top['absolute_delta']:.6f}"
        )
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
