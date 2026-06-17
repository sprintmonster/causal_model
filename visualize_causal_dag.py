#!/usr/bin/env python3
"""Visualize the causal DAG as a PNG.

Default behavior renders the ancestor subgraph around a target node so the
diagram stays readable. Use --all to render the full graph, though that can be
very dense for large models.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from textwrap import fill

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL = WORKSPACE / "final_graph" / "causal_model.json"
DEFAULT_OUT = WORKSPACE / "final_graph" / "causal_dag.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.unicode_minus": False,
    }
)

NODE_COLORS = {
    "Actor": "#4C78A8",
    "Event": "#F58518",
    "Claim": "#54A24B",
    "Topic": "#E45756",
    "Place": "#72B7B2",
    "Role": "#B279A2",
    "Quote": "#9D755D",
    "Mention": "#BAB0AC",
    "Entity": "#7F7F7F",
}

EDGE_COLORS = {
    "CAUSES": "#D62728",
    "ENABLES": "#2CA02C",
    "MEDIATES": "#1F77B4",
}


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(model: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in model.get("nodes", []):
        node_id = node.get("id")
        if not node_id:
            continue
        attrs = node.get("attributes") or {}
        graph.add_node(
            node_id,
            label=node.get("label", node_id),
            type=node.get("type", "Entity"),
            support_probability=float(attrs.get("support_probability", 0.0) or 0.0),
            support_count=int(attrs.get("support_count", 0) or 0),
        )

    for edge in model.get("edges", []):
        src = edge.get("from")
        dst = edge.get("to")
        rel = edge.get("rel", "CAUSES")
        if not src or not dst or src not in graph or dst not in graph:
            continue
        attrs = edge.get("attributes") or {}
        graph.add_edge(
            src,
            dst,
            rel=rel,
            support_probability=float(attrs.get("support_probability", 0.0) or 0.0),
            support_count=int(attrs.get("support_count", 0) or 0),
        )

    return graph


def resolve_focus(graph: nx.DiGraph, focus: str) -> str:
    focus_norm = focus.strip().lower()
    for node_id, data in graph.nodes(data=True):
        label = str(data.get("label", "")).lower()
        if node_id.lower() == focus_norm or label == focus_norm:
            return node_id
    for node_id, data in graph.nodes(data=True):
        label = str(data.get("label", "")).lower()
        if focus_norm in node_id.lower() or focus_norm in label:
            return node_id
    raise KeyError(f"focus node not found: {focus}")


def collect_subgraph(graph: nx.DiGraph, focus: str, depth: int, direction: str) -> nx.DiGraph:
    if depth < 0:
        return graph.copy()

    if focus not in graph:
        raise KeyError(focus)

    keep = {focus}
    frontier = {focus}

    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            if direction in {"ancestors", "both"}:
                next_frontier.update(graph.predecessors(node))
            if direction in {"descendants", "both"}:
                next_frontier.update(graph.successors(node))
        next_frontier -= keep
        if not next_frontier:
            break
        keep.update(next_frontier)
        frontier = next_frontier

    return graph.subgraph(keep).copy()


def topological_layers(subgraph: nx.DiGraph, focus: str) -> dict[str, int]:
    if not nx.is_directed_acyclic_graph(subgraph):
        # Fallback: use BFS distance from focus if the extracted window still has cycles.
        return {focus: 0}

    topo = list(nx.topological_sort(subgraph))
    layers = {nid: 0 for nid in topo}
    for nid in topo:
        preds = list(subgraph.predecessors(nid))
        if preds:
            layers[nid] = max(layers[p] + 1 for p in preds)
    if focus in layers:
        focus_layer = layers[focus]
        for nid in layers:
            layers[nid] -= focus_layer
    return layers


def layout_positions(subgraph: nx.DiGraph, focus: str) -> dict[str, tuple[float, float]]:
    layers = topological_layers(subgraph, focus)
    buckets: defaultdict[int, list[str]] = defaultdict(list)
    for node_id, layer in layers.items():
        buckets[layer].append(node_id)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in sorted(buckets.items()):
        nodes_sorted = sorted(nodes, key=lambda nid: (subgraph.nodes[nid].get("type", ""), subgraph.nodes[nid].get("label", "")))
        width = max(1, len(nodes_sorted) - 1)
        for idx, node_id in enumerate(nodes_sorted):
            x = 0.0 if width == 0 else idx - width / 2.0
            pos[node_id] = (x, -float(layer))

    if len(pos) > 1:
        try:
            focused = nx.spring_layout(subgraph, seed=42)
            # Blend hierarchy with force layout for better readability.
            for node_id, (sx, sy) in pos.items():
                fx, fy = focused.get(node_id, (sx, sy))
                pos[node_id] = (0.55 * sx + 0.45 * fx, 0.55 * sy + 0.45 * fy)
        except Exception:
            pass
    return pos


def node_color(node_type: str) -> str:
    return NODE_COLORS.get(node_type, NODE_COLORS["Entity"])


def edge_color(rel: str) -> str:
    return EDGE_COLORS.get(rel, "#7F7F7F")


def format_label(label: str, node_type: str, support: int, width: int = 18) -> str:
    base = fill(label, width=width)
    meta = f"[{node_type}, n={support}]"
    return f"{base}\n{meta}"


def render(subgraph: nx.DiGraph, focus: str, out_path: Path, title: str) -> None:
    plt.figure(figsize=(18, 12))
    ax = plt.gca()
    ax.set_title(title, fontsize=16, pad=20, fontweight="semibold")
    ax.axis("off")

    pos = layout_positions(subgraph, focus)

    node_types = [subgraph.nodes[n].get("type", "Entity") for n in subgraph.nodes]
    node_sizes = [600 + 120 * min(6, int(subgraph.nodes[n].get("support_count", 0))) for n in subgraph.nodes]
    colors = [node_color(nt) for nt in node_types]
    edge_colors = [edge_color(subgraph.edges[e].get("rel", "CAUSES")) for e in subgraph.edges]
    edge_widths = [0.5 + 2.0 * float(subgraph.edges[e].get("support_probability", 0.0)) for e in subgraph.edges]

    nx.draw_networkx_edges(
        subgraph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.75,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_color=colors,
        node_size=node_sizes,
        linewidths=1.2,
        edgecolors="#1f1f1f",
    )

    labels = {}
    for node_id, data in subgraph.nodes(data=True):
        label = str(data.get("label", node_id))
        node_type = str(data.get("type", "Entity"))
        support = int(data.get("support_count", 0) or 0)
        labels[node_id] = format_label(label, node_type, support)

    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=labels,
        font_size=8,
        font_color="#111111",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#DDDDDD", alpha=0.85),
    )

    # Highlight the focus node.
    if focus in pos:
        fx, fy = pos[focus]
        ax.scatter([fx], [fy], s=2200, facecolors="none", edgecolors="#000000", linewidths=2.0, zorder=5)

    # Legends.
    from matplotlib.patches import Patch

    node_legend = [Patch(facecolor=node_color(t), edgecolor="#1f1f1f", label=t) for t in ["Topic", "Event", "Claim", "Actor", "Place", "Role"]]
    edge_legend = [Patch(facecolor=edge_color(r), label=r) for r in ["CAUSES", "ENABLES", "MEDIATES"]]
    ax.legend(handles=node_legend + edge_legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="Legend")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target", default="topic:prohibition")
    ap.add_argument("--depth", type=int, default=2, help="Ancestor/descendant depth around the target.")
    ap.add_argument("--direction", choices=["ancestors", "descendants", "both"], default="ancestors")
    ap.add_argument("--all", action="store_true", help="Render the full graph instead of a focused subgraph.")
    ap.add_argument("--title", default="Causal DAG")
    args = ap.parse_args()

    model = load_model(args.model)
    graph = build_graph(model)

    if args.all:
        subgraph = graph
        focus = resolve_focus(graph, args.target) if graph else args.target
        title = f"{args.title} - full graph"
    else:
        focus = resolve_focus(graph, args.target)
        subgraph = collect_subgraph(graph, focus, args.depth, args.direction)
        title = f"{args.title} - {focus} ({args.direction}, depth={args.depth})"

    render(subgraph, focus, args.output, title)
    print(f"[done] wrote {args.output}")
    print(f"[info] nodes={subgraph.number_of_nodes()} edges={subgraph.number_of_edges()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())