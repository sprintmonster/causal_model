#!/usr/bin/env python3
"""Build a DAG-based probabilistic causal model from the merged knowledge graph.

Model assumptions:
- Nodes are binary latent states (active/inactive)
- Node base rate comes from node support_probability
- Parent effects use edge support_probability with noisy-OR aggregation
- Cycles are removed by dropping backward edges using a stable node ordering
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_GRAPH = WORKSPACE / "final_graph" / "final_causal_graph.json"
DEFAULT_MODEL_OUT = WORKSPACE / "final_graph" / "causal_model.json"
DEFAULT_REPORT_OUT = WORKSPACE / "final_graph" / "causal_intervention_report.json"
DEFAULT_TARGET = "event:war-time-prohibition-enforcement-act"
CAUSAL_RELS = {"CAUSES", "ENABLES", "MEDIATES"}
CLUSTER_TYPES = {"event", "claim", "topic"}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_graph(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def node_support_prob(node: dict[str, Any], default: float = 0.01) -> float:
    attrs = node.get("attributes") or {}
    p = attrs.get("support_probability", default)
    try:
        return clamp(float(p), 1e-4, 0.999)
    except (TypeError, ValueError):
        return default


def edge_support_prob(edge: dict[str, Any], default: float = 0.01) -> float:
    attrs = edge.get("attributes") or {}
    p = attrs.get("support_probability", default)
    try:
        return clamp(float(p), 1e-4, 0.999)
    except (TypeError, ValueError):
        return default


def resolve_node_id(graph: dict[str, Any], target: str) -> str:
    target_norm = target.strip().lower()
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", "")).lower()
        label = str(node.get("label", "")).lower()
        if node_id == target_norm or label == target_norm:
            return node["id"]
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", "")).lower()
        label = str(node.get("label", "")).lower()
        if target_norm in node_id or target_norm in label:
            return node["id"]
    raise KeyError(f"node not found: {target}")


def extract_ancestor_subgraph(
    graph: dict[str, Any],
    target: str,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Keep the target and its upstream causal ancestors as a focused subgraph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_map = {n.get("id"): n for n in nodes if n.get("id")}
    target_id = resolve_node_id(graph, target)

    reverse: dict[str, list[str]] = defaultdict(list)
    incoming_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge.get("rel") not in CAUSAL_RELS:
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        reverse[dst].append(src)
        incoming_edges[dst].append(edge)

    keep = {target_id}
    queue = deque([(target_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for parent in reverse.get(node_id, []):
            if parent in keep:
                continue
            keep.add(parent)
            queue.append((parent, depth + 1))

    sub_nodes = [node_map[nid] for nid in node_map if nid in keep]
    sub_edges = [edge for edge in edges if edge.get("from") in keep and edge.get("to") in keep]

    return {
        "meta": {
            "source_graph_meta": graph.get("meta", {}),
            "focus_target": target_id,
            "ancestor_depth": max_depth,
            "subgraph_nodes": len(sub_nodes),
            "subgraph_edges": len(sub_edges),
        },
        "nodes": sub_nodes,
        "edges": sub_edges,
    }


def extract_cluster_subgraph(
    graph: dict[str, Any],
    query: str,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Build a wider prohibition-centered DAG from multiple seed nodes.

    Seeds are nodes whose id or label contains the query string, limited to
    event/claim/topic nodes so the result stays conceptually focused.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_map = {n.get("id"): n for n in nodes if n.get("id")}

    query_norm = query.strip().lower()
    seeds = []
    for node in nodes:
        node_id = str(node.get("id", "")).lower()
        label = str(node.get("label", "")).lower()
        node_type = str(node.get("type", "")).lower()
        if node_type not in CLUSTER_TYPES:
            continue
        if query_norm in node_id or query_norm in label:
            seeds.append(node["id"])

    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("rel") not in CAUSAL_RELS:
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        reverse[dst].append(src)

    keep = set(seeds)
    queue = deque([(seed, 0) for seed in seeds])
    while queue:
        node_id, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for parent in reverse.get(node_id, []):
            if parent in keep:
                continue
            keep.add(parent)
            queue.append((parent, depth + 1))

    sub_nodes = [node_map[nid] for nid in node_map if nid in keep]
    sub_edges = [edge for edge in edges if edge.get("from") in keep and edge.get("to") in keep]

    return {
        "meta": {
            "source_graph_meta": graph.get("meta", {}),
            "seed_query": query,
            "seed_count": len(seeds),
            "seed_sample": seeds[:25],
            "ancestor_depth": max_depth,
            "subgraph_nodes": len(sub_nodes),
            "subgraph_edges": len(sub_edges),
        },
        "nodes": sub_nodes,
        "edges": sub_edges,
    }


def build_dag_edges(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    min_edge_prob: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep causal direction and remove only cycle-closing edges to obtain a DAG."""
    node_ids = {n["id"] for n in nodes if n.get("id")}
    candidates: list[dict[str, Any]] = []
    dropped_rel = 0
    dropped_prob = 0
    dropped_missing = 0
    dropped_cycle = 0

    for e in edges:
        rel = e.get("rel")
        if rel not in CAUSAL_RELS:
            dropped_rel += 1
            continue
        src = e.get("from")
        dst = e.get("to")
        if not src or not dst or src not in node_ids or dst not in node_ids:
            dropped_missing += 1
            continue
        prob = edge_support_prob(e)
        if prob < min_edge_prob:
            dropped_prob += 1
            continue

        candidates.append(
            {
                "from": src,
                "to": dst,
                "rel": rel,
                "edge_probability": prob,
            }
        )

    # Prefer stronger edges first; weaker edges are dropped when they close cycles.
    candidates.sort(key=lambda x: (-float(x["edge_probability"]), x["from"], x["to"], x["rel"]))

    adj: dict[str, set[str]] = defaultdict(set)
    dag_edges: list[dict[str, Any]] = []

    def has_path(start: str, goal: str) -> bool:
        if start == goal:
            return True
        stack = [start]
        seen = {start}
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, set()):
                if nxt == goal:
                    return True
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        return False

    for e in candidates:
        src = e["from"]
        dst = e["to"]
        # If dst already reaches src, adding src->dst would create a cycle.
        if has_path(dst, src):
            dropped_cycle += 1
            continue
        dag_edges.append(e)
        adj[src].add(dst)

    stats = {
        "kept": len(dag_edges),
        "dropped_non_causal_rel": dropped_rel,
        "dropped_low_probability": dropped_prob,
        "dropped_missing_nodes": dropped_missing,
        "dropped_cycle_back_edges": dropped_cycle,
    }
    return dag_edges, stats


def topological_order(node_ids: list[str], dag_edges: list[dict[str, Any]]) -> list[str]:
    indeg = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = defaultdict(list)
    for e in dag_edges:
        s = e["from"]
        d = e["to"]
        adj[s].append(d)
        indeg[d] = indeg.get(d, 0) + 1

    q = deque([nid for nid in node_ids if indeg.get(nid, 0) == 0])
    order: list[str] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in adj.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(node_ids):
        # Should not happen after order-based pruning; fallback deterministic append.
        missing = [nid for nid in node_ids if nid not in set(order)]
        order.extend(sorted(missing))
    return order


def build_model(graph: dict[str, Any], min_edge_prob: float) -> dict[str, Any]:
    nodes = [n for n in graph.get("nodes", []) if n.get("id")]
    edges = graph.get("edges", [])
    node_by_id = {n["id"]: n for n in nodes}

    dag_edges, dag_stats = build_dag_edges(nodes, edges, min_edge_prob=min_edge_prob)

    parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in dag_edges:
        parents[e["to"]].append(
            {
                "id": e["from"],
                "rel": e["rel"],
                "edge_probability": e["edge_probability"],
            }
        )

    model_nodes: list[dict[str, Any]] = []
    for nid, n in node_by_id.items():
        base_rate = node_support_prob(n)
        model_nodes.append(
            {
                "id": nid,
                "label": n.get("label", nid),
                "type": n.get("type"),
                "base_rate": round(base_rate, 6),
                "parents": sorted(parents.get(nid, []), key=lambda p: p["id"]),
            }
        )

    topo = topological_order(list(node_by_id.keys()), dag_edges)

    model = {
        "meta": {
            "source_graph_meta": graph.get("meta", {}),
            "assumptions": {
                "node_state": "binary latent activation",
                "node_base_rate": "node.attributes.support_probability",
                "edge_effect": "edge.attributes.support_probability",
                "aggregation": "noisy-or",
            },
            "causal_relationships": sorted(CAUSAL_RELS),
            "min_edge_probability": min_edge_prob,
            "dag_stats": dag_stats,
            "n_model_nodes": len(model_nodes),
            "n_model_edges": len(dag_edges),
        },
        "topological_order": topo,
        "nodes": sorted(model_nodes, key=lambda x: x["id"]),
        "edges": dag_edges,
    }
    return model


def infer_marginals(model: dict[str, Any], intervention: dict[str, float] | None = None) -> dict[str, float]:
    """Compute approximate marginals with noisy-OR forward propagation."""
    intervention = intervention or {}

    node_map = {n["id"]: n for n in model["nodes"]}
    topo = model["topological_order"]
    probs: dict[str, float] = {}

    for nid in topo:
        if nid in intervention:
            probs[nid] = clamp(float(intervention[nid]), 0.0, 1.0)
            continue

        node = node_map[nid]
        base = clamp(float(node.get("base_rate", 0.01)), 1e-6, 0.999999)
        p_off = 1.0 - base

        for par in node.get("parents", []):
            pid = par.get("id")
            if not pid:
                continue
            pw = clamp(float(par.get("edge_probability", 0.01)), 1e-6, 0.999999)
            p_parent = probs.get(pid, node_map.get(pid, {}).get("base_rate", 0.01))
            # Expected noisy-OR contribution under parent marginal.
            p_off *= 1.0 - (pw * float(p_parent))

        probs[nid] = clamp(1.0 - p_off, 0.0, 1.0)

    return probs


def build_intervention_report(
    model: dict[str, Any],
    target: str,
    intervene_node: str,
    intervene_value: float,
) -> dict[str, Any]:
    base = infer_marginals(model)
    post = infer_marginals(model, intervention={intervene_node: intervene_value})

    if target not in base:
        raise KeyError(f"target not found in model: {target}")
    if intervene_node not in base:
        raise KeyError(f"intervene node not found in model: {intervene_node}")

    return {
        "target": target,
        "intervention": {
            "do": intervene_node,
            "value": intervene_value,
        },
        "baseline": {
            "P(target)": round(base[target], 6),
            "P(intervene_node)": round(base[intervene_node], 6),
        },
        "post_intervention": {
            "P(target)": round(post[target], 6),
            "P(intervene_node)": round(post[intervene_node], 6),
        },
        "effect": {
            "absolute_delta": round(post[target] - base[target], 6),
            "relative_delta": round((post[target] - base[target]) / max(base[target], 1e-9), 6),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_OUT)
    ap.add_argument("--output-report", type=Path, default=DEFAULT_REPORT_OUT)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--seed-query", default="")
    ap.add_argument("--ancestor-depth", type=int, default=5)
    ap.add_argument("--intervene-node", default="")
    ap.add_argument("--intervene-value", type=float, default=1.0)
    ap.add_argument("--min-edge-prob", type=float, default=0.001)
    ap.add_argument("--full-graph", action="store_true", help="Build the model from the full graph instead of a target-centered ancestor subgraph.")
    args = ap.parse_args()

    graph = load_graph(args.graph)
    if args.full_graph:
        focused_graph = graph
    elif args.seed_query:
        focused_graph = extract_cluster_subgraph(graph, args.seed_query, max_depth=args.ancestor_depth)
    else:
        focused_graph = extract_ancestor_subgraph(graph, args.target, max_depth=args.ancestor_depth)
    model = build_model(focused_graph, min_edge_prob=args.min_edge_prob)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote model -> {args.output_model}")

    # If no intervene node provided, pick strongest direct parent of target.
    intervene_node = args.intervene_node
    if not args.seed_query:
        if not intervene_node:
            tnode = next((n for n in model["nodes"] if n["id"] == resolve_node_id(focused_graph, args.target)), None)
            if tnode and tnode.get("parents"):
                best = max(tnode["parents"], key=lambda p: float(p.get("edge_probability", 0.0)))
                intervene_node = best["id"]

        if intervene_node:
            report = build_intervention_report(
                model=model,
                target=args.target,
                intervene_node=intervene_node,
                intervene_value=args.intervene_value,
            )
            args.output_report.parent.mkdir(parents=True, exist_ok=True)
            args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                "[effect] "
                f"do({report['intervention']['do']}={report['intervention']['value']}) "
                f"=> P({report['target']}) {report['baseline']['P(target)']:.6f} "
                f"-> {report['post_intervention']['P(target)']:.6f} "
                f"(delta={report['effect']['absolute_delta']:.6f})"
            )
            print(f"[done] wrote intervention report -> {args.output_report}")
        else:
            print("[info] no intervention node selected; model only saved")
    else:
        print("[info] seed-query mode selected; intervention report skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
