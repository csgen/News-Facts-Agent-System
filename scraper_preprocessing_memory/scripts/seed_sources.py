"""Seed SOURCE nodes into Neo4j from data/sources.json."""

import json
import sys

sys.path.insert(0, ".")

from src.config import settings
from src.memory.graph_store import GraphStore


def main():
    with open("data/sources.json") as f:
        sources = json.load(f)

    graph = GraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    for src in sources:
        graph.merge_source(
            source_id=src["source_id"],
            name=src["name"],
            domain=src["domain"],
            category=src["category"],
            base_credibility=src["base_credibility"],
        )
        print(f"  Seeded: {src['name']} ({src['source_id']})")

    graph.close()
    print(f"\nSeeded {len(sources)} sources into Neo4j.")


if __name__ == "__main__":
    main()
