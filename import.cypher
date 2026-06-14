// Neo4j import script for the causal DAG
// Place the CSV files inside Neo4j's import directory, then execute this script.

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {id: row.`id:ID`})
SET n.label = row.label,
    n.type = row.type,
    n.support_count = toInteger(row.`support_count:int`),
    n.support_probability = toFloat(row.`support_probability:float`),
    n.base_rate = toFloat(row.`base_rate:float`),
    n.batch_ids = CASE WHEN row.batch_ids = '' THEN [] ELSE split(row.batch_ids, '|') END,
    n.description = row.description,
    n.date = row.date,
    n.location = row.location;

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {id: row.`:START_ID`})
MATCH (t:Entity {id: row.`:END_ID`})
MERGE (s)-[r:CAUSAL_LINK]->(t)
SET r.rel = row.rel,
    r.support_count = toInteger(row.`support_count:int`),
    r.support_probability = toFloat(row.`support_probability:float`),
    r.log_support_odds = toFloat(row.`log_support_odds:float`),
    r.batch_ids = CASE WHEN row.batch_ids = '' THEN [] ELSE split(row.batch_ids, '|') END,
    r.edge_probability = toFloat(row.`edge_probability:float`);
