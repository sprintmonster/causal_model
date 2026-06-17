#!/usr/bin/env python3
"""Stage C - merge batch-level causal graphs into one final graph."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import log1p
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
INPUT_DIR = WORKSPACE / "batch_graphs"
OUT_DIR = WORKSPACE / "final_graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "final_causal_graph.json"


def iter_graph_files(input_dir: Path):
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        yield path


def merge_graphs(graph_files: list[Path]) -> dict:
    merged_nodes: dict[str, dict] = {}
    merged_edges: dict[tuple[str, str, str], dict] = {}
    provenance: defaultdict[str, set[str]] = defaultdict(set)

    for path in graph_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        batch_id = data.get("batch", {}).get("id") or data.get("batch_id") or path.stem

        for node in data.get("nodes", []):
            node_id = node.get("id")
            if not node_id:
                continue
            current = merged_nodes.get(node_id)
            if current is None:
                current = dict(node)
                current.setdefault("attributes", {})
                current["attributes"].setdefault("batch_ids", [])
                merged_nodes[node_id] = current
            if batch_id not in current["attributes"].setdefault("batch_ids", []):
                current["attributes"]["batch_ids"].append(batch_id)
            if node.get("attributes"):
                na = node["attributes"]
                if isinstance(na, dict):
                    current["attributes"].update(na)
                else:
                    # accommodate malformed attribute values (e.g., a string)
                    current["attributes"].setdefault("note", na)
            current["attributes"]["support_count"] = len(current["attributes"]["batch_ids"])

        for edge in data.get("edges", []):
            src = edge.get("from")
            dst = edge.get("to")
            rel = edge.get("rel")
            if not src or not dst or not rel:
                continue
            key = (src, rel, dst)
            current = merged_edges.get(key)
            if current is None:
                current = dict(edge)
                current.setdefault("attributes", {})
                current["attributes"].setdefault("batch_ids", [])
                merged_edges[key] = current
            if batch_id not in current["attributes"].setdefault("batch_ids", []):
                current["attributes"]["batch_ids"].append(batch_id)
            if edge.get("attributes"):
                ea = edge["attributes"]
                if isinstance(ea, dict):
                    current["attributes"].update(ea)
                else:
                    current["attributes"].setdefault("note", ea)
            provenance[batch_id].add(key[0])
            provenance[batch_id].add(key[2])
            current["attributes"]["support_count"] = len(current["attributes"]["batch_ids"])

    n_batches = len(graph_files)
    for node in merged_nodes.values():
        attrs = node.setdefault("attributes", {})
        support_count = int(attrs.get("support_count", 0))
        attrs["support_probability"] = round((support_count + 1) / (n_batches + 2), 4)

    for edge in merged_edges.values():
        attrs = edge.setdefault("attributes", {})
        support_count = int(attrs.get("support_count", 0))
        attrs["support_probability"] = round((support_count + 1) / (n_batches + 2), 4)
        attrs["log_support_odds"] = round(log1p(support_count) - log1p(max(0, n_batches - support_count)), 4)

    return {
        "meta": {
            "source_dir": str(INPUT_DIR),
            "n_batches": n_batches,
            "n_nodes": len(merged_nodes),
            "n_edges": len(merged_edges),
        },
        "nodes": list(merged_nodes.values()),
        "edges": list(merged_edges.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    ap.add_argument("--output", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    graph_files = list(iter_graph_files(args.input_dir))
    if not graph_files:
        raise SystemExit(f"no batch graphs found in {args.input_dir}")

    merged = merge_graphs(graph_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] merged {len(graph_files)} batch graphs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())