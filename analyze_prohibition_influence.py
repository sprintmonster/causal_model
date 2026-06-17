#!/usr/bin/env python3
"""Rank upstream causes of prohibition using probability-weighted causal paths."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from math import prod
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
FINAL_GRAPH = WORKSPACE / "final_graph" / "final_causal_graph.json"
MERGE_SCRIPT = WORKSPACE / "merge_batch_graphs.py"
CAUSAL_RELS = {"CAUSES", "ENABLES", "MEDIATES"}


def load_graph(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_target(nodes: list[dict], target: str) -> str | None:
    target_norm = target.strip().lower()
    for node in nodes:
        node_id = str(node.get("id", "")).lower()
        label = str(node.get("label", "")).lower()
        if node_id == target_norm or label == target_norm:
            return node["id"]
    for node in nodes:
        node_id = str(node.get("id", "")).lower()
        label = str(node.get("label", "")).lower()
        if target_norm in node_id or target_norm in label:
            return node["id"]
    return None


def build_reverse_adjacency(edges: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    forward: defaultdict[str, list[dict]] = defaultdict(list)
    reverse: defaultdict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        rel = edge.get("rel")
        if rel not in CAUSAL_RELS:
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        attrs = edge.get("attributes") or {}
        prob = float(attrs.get("support_probability", 0.5))
        forward[src].append({"src": src, "dst": dst, "rel": rel, "prob": prob, "edge": edge})
        reverse[dst].append({"src": src, "dst": dst, "rel": rel, "prob": prob, "edge": edge})
    return forward, reverse


def enumerate_paths_to_target(reverse: dict[str, list[dict]], target: str, max_depth: int = 4) -> list[list[dict]]:
    paths: list[list[dict]] = []
    stack: list[tuple[str, list[dict], set[str], int]] = [(target, [], {target}, 0)]

    while stack:
        node, path_edges, visited, depth = stack.pop()
        parents = reverse.get(node, [])
        if not parents or depth >= max_depth:
            if path_edges:
                paths.append(list(reversed(path_edges)))
            continue
        extended = False
        for edge in parents:
            src = edge["src"]
            if src in visited:
                continue
            extended = True
            stack.append((src, path_edges + [edge], visited | {src}, depth + 1))
        if not extended and path_edges:
            paths.append(list(reversed(path_edges)))
    return paths


def path_probability(path_edges: list[dict]) -> float:
    if not path_edges:
        return 0.0
    return prod(edge.get("prob", 0.5) for edge in path_edges)


def rank_causes(graph: dict, target_node_id: str, max_depth: int = 4) -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_by_id = {node.get("id"): node for node in nodes if node.get("id")}
    _, reverse = build_reverse_adjacency(edges)
    paths = enumerate_paths_to_target(reverse, target_node_id, max_depth=max_depth)

    cause_scores: defaultdict[str, list[float]] = defaultdict(list)
    cause_paths: defaultdict[str, list[dict]] = defaultdict(list)

    for path in paths:
        if not path:
            continue
        source = path[0]["src"]
        score = path_probability(path)
        cause_scores[source].append(score)
        cause_paths[source].append({
            "path": [edge["src"] for edge in path] + [target_node_id],
            "rels": [edge["rel"] for edge in path],
            "probability": round(score, 6),
        })

    ranked = []
    for source, scores in cause_scores.items():
        node = node_by_id.get(source, {})
        ranked.append({
            "node_id": source,
            "label": node.get("label", source),
            "type": node.get("type"),
            "max_path_probability": round(max(scores), 6),
            "sum_path_probability": round(sum(scores), 6),
            "path_count": len(scores),
            "paths": sorted(cause_paths[source], key=lambda x: x["probability"], reverse=True)[:5],
        })

    ranked.sort(key=lambda item: (item["max_path_probability"], item["sum_path_probability"]), reverse=True)
    return {
        "target": {
            "node_id": target_node_id,
            "label": node_by_id.get(target_node_id, {}).get("label", target_node_id),
        },
        "ranking": ranked,
        "n_paths": len(paths),
        "n_sources": len(ranked),
    }


def maybe_build_final_graph(graph_path: Path) -> Path:
    if graph_path.exists():
        return graph_path
    import subprocess

    subprocess.run([str(Path("/home/jeonboyun/anaconda3/envs/RagEnv/bin/python")), str(MERGE_SCRIPT)], check=True)
    return graph_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=FINAL_GRAPH)
    ap.add_argument("--target", default="topic:prohibition")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--output", type=Path, default=WORKSPACE / "final_graph" / "prohibition_influence_report.json")
    args = ap.parse_args()

    graph_path = maybe_build_final_graph(args.graph)
    graph = load_graph(graph_path)
    target_node_id = resolve_target(graph.get("nodes", []), args.target)
    if not target_node_id:
        raise SystemExit(f"target not found: {args.target}")

    report = rank_causes(graph, target_node_id, max_depth=args.max_depth)
    report["graph_meta"] = graph.get("meta", {})
    report["probability_model"] = {
        "edge_probability": "(support_count + 1) / (n_batches + 2)",
        "path_probability": "product(edge probabilities)",
        "rank_metric": "max_path_probability then sum_path_probability",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if report["ranking"]:
        top = report["ranking"][0]
        print(f"[top] {top['label']} ({top['node_id']}) -> P={top['max_path_probability']:.4f}")
    else:
        print("[top] no upstream causes found")
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())