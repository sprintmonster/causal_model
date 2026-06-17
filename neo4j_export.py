#!/usr/bin/env python3
"""Export the DAG causal model to Neo4j-friendly CSV and Cypher files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL = WORKSPACE / "final_graph" / "causal_model.json"
DEFAULT_OUT_DIR = WORKSPACE / "final_graph" / "neo4j"


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def flatten_batch_ids(attrs: dict[str, Any]) -> str:
    batch_ids = attrs.get("batch_ids")
    if isinstance(batch_ids, list):
        return "|".join(str(item) for item in batch_ids)
    return ""


def export_nodes(nodes: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "id:ID",
        "label",
        "type",
        "support_count:int",
        "support_probability:float",
        "base_rate:float",
        "batch_ids",
        "description",
        "date",
        "location",
        ":LABEL",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for node in nodes:
            attrs = node.get("attributes") or {}
            writer.writerow(
                {
                    "id:ID": node.get("id", ""),
                    "label": node.get("label", ""),
                    "type": node.get("type", ""),
                    "support_count:int": attrs.get("support_count", ""),
                    "support_probability:float": attrs.get("support_probability", ""),
                    "base_rate:float": attrs.get("base_rate", attrs.get("support_probability", "")),
                    "batch_ids": flatten_batch_ids(attrs),
                    "description": csv_value(attrs.get("description")),
                    "date": csv_value(attrs.get("date")),
                    "location": csv_value(attrs.get("location")),
                    ":LABEL": node.get("type", "Entity"),
                }
            )


def export_edges(edges: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        ":START_ID",
        ":END_ID",
        ":TYPE",
        "rel",
        "support_count:int",
        "support_probability:float",
        "log_support_odds:float",
        "batch_ids",
        "edge_probability:float",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for edge in edges:
            attrs = edge.get("attributes") or {}
            rel = edge.get("rel", "RELATED_TO")
            writer.writerow(
                {
                    ":START_ID": edge.get("from", ""),
                    ":END_ID": edge.get("to", ""),
                    ":TYPE": "CAUSAL_LINK",
                    "rel": rel,
                    "support_count:int": attrs.get("support_count", ""),
                    "support_probability:float": attrs.get("support_probability", ""),
                    "log_support_odds:float": attrs.get("log_support_odds", ""),
                    "batch_ids": flatten_batch_ids(attrs),
                    "edge_probability:float": attrs.get("edge_probability", attrs.get("support_probability", "")),
                }
            )


def export_cypher(out_path: Path, nodes_csv: str, edges_csv: str) -> None:
    cypher = f"""// Neo4j import script for the causal DAG
// Place the CSV files inside Neo4j's import directory, then execute this script.

LOAD CSV WITH HEADERS FROM 'file:///{nodes_csv}' AS row
MERGE (n:Entity {{id: row.`id:ID`}})
SET n.label = row.label,
    n.type = row.type,
    n.support_count = toInteger(row.`support_count:int`),
    n.support_probability = toFloat(row.`support_probability:float`),
    n.base_rate = toFloat(row.`base_rate:float`),
    n.batch_ids = CASE WHEN row.batch_ids = '' THEN [] ELSE split(row.batch_ids, '|') END,
    n.description = row.description,
    n.date = row.date,
    n.location = row.location;

LOAD CSV WITH HEADERS FROM 'file:///{edges_csv}' AS row
MATCH (s:Entity {{id: row.`:START_ID`}})
MATCH (t:Entity {{id: row.`:END_ID`}})
MERGE (s)-[r:CAUSAL_LINK]->(t)
SET r.rel = row.rel,
    r.support_count = toInteger(row.`support_count:int`),
    r.support_probability = toFloat(row.`support_probability:float`),
    r.log_support_odds = toFloat(row.`log_support_odds:float`),
    r.batch_ids = CASE WHEN row.batch_ids = '' THEN [] ELSE split(row.batch_ids, '|') END,
    r.edge_probability = toFloat(row.`edge_probability:float`);
"""
    out_path.write_text(cypher, encoding="utf-8")


def export_summary(model: dict[str, Any], out_path: Path, nodes_csv: str, edges_csv: str) -> None:
    summary = {
        "model_meta": model.get("meta", {}),
        "neo4j": {
            "nodes_csv": nodes_csv,
            "edges_csv": edges_csv,
            "node_label": "Entity",
            "relationship_type": "CAUSAL_LINK",
            "original_edge_relation": "stored in edges.csv rel column and r.rel property",
        },
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    model = load_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nodes_csv = args.output_dir / "nodes.csv"
    edges_csv = args.output_dir / "edges.csv"
    cypher_path = args.output_dir / "import.cypher"
    summary_path = args.output_dir / "neo4j_export_summary.json"

    export_nodes(model.get("nodes", []), nodes_csv)
    export_edges(model.get("edges", []), edges_csv)
    export_cypher(cypher_path, nodes_csv.name, edges_csv.name)
    export_summary(model, summary_path, nodes_csv.name, edges_csv.name)

    print(f"[done] wrote {nodes_csv}")
    print(f"[done] wrote {edges_csv}")
    print(f"[done] wrote {cypher_path}")
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())