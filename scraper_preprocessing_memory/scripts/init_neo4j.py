"""Initialize Neo4j schema (constraints + indexes). Idempotent — safe to run multiple times."""

import sys

sys.path.insert(0, ".")

from src.config import settings
from src.memory.graph_store import GraphStore


def main():
    graph = GraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    graph.init_schema()
    graph.close()
    print("Neo4j schema initialized successfully.")


if __name__ == "__main__":
    main()
