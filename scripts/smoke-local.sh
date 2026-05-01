#!/usr/bin/env bash
# Run the same smoke checks as `.github/workflows/release-image.yml` does in CI,
# but against a locally-built image and locally-spun-up Neo4j + Chroma sidecars.
#
# Usage:
#   bash scripts/smoke-local.sh
#
# Prerequisites:
#   - Docker Desktop (or Docker engine on Linux) running
#   - curl available on PATH (Git Bash on Windows has it; Mac/Linux do too)
#
# What it does:
#   1. Builds the image as `news_facts_system:smoke` from your local Dockerfile
#   2. Starts Neo4j 5-community + Chroma:latest on a dedicated `smoke-net`
#      Docker network (so they can talk to the test container by hostname)
#   3. Waits for both DBs to be ready (heartbeat / cypher-shell poll)
#   4. Runs the 4 smoke checks:
#        Smoke 1: imports
#        Smoke 2: MemoryAgent.init_schema()
#        Smoke 3: ensure_entity + get_entity_by_name round-trip
#        Smoke 4: Streamlit boots and answers /_stcore/health
#   5. Cleans up containers + network on exit (success or failure)
#
# Exit code: 0 on full pass, non-zero on any failure (matches CI semantics).

set -euo pipefail

IMAGE="news_facts_system:smoke"
NET="smoke-net"
NEO4J="smoke-neo4j"
CHROMA="smoke-chroma"
UI="smoke-ui"
NEO4J_PASS="smoketest123"

# Colours for human readability — disable with NO_COLOR=1 if redirecting to a file.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    G="\033[32m"; R="\033[31m"; B="\033[34m"; D="\033[2m"; N="\033[0m"
else
    G=""; R=""; B=""; D=""; N=""
fi

# ── Cleanup hook — runs even on failure / Ctrl-C ─────────────────────────────
cleanup() {
    echo -e "${D}=== cleanup ===${N}"
    docker rm -f "$NEO4J" "$CHROMA" "$UI" >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Ensure a clean starting state in case a previous run died mid-way.
cleanup

# ── 1. Build ─────────────────────────────────────────────────────────────────
echo -e "${B}=== Building $IMAGE ===${N}"
# `--load` ensures the image lands in the local Docker daemon (only needed if
# you've configured a custom buildx context; harmless otherwise).
docker build -t "$IMAGE" -f docker/Dockerfile .

# ── 2. Start sidecars ────────────────────────────────────────────────────────
echo -e "${B}=== Starting Neo4j + Chroma sidecars ===${N}"
docker network create "$NET" >/dev/null

docker run -d --name "$NEO4J" --network "$NET" \
    -e NEO4J_AUTH=neo4j/${NEO4J_PASS} \
    -p 7687:7687 -p 7474:7474 \
    neo4j:5-community >/dev/null

docker run -d --name "$CHROMA" --network "$NET" \
    -p 8000:8000 \
    chromadb/chroma:latest >/dev/null

# ── 3. Wait for readiness ────────────────────────────────────────────────────
echo -ne "${D}Waiting for Chroma..${N}"
for i in $(seq 1 60); do
    if curl -fsS http://localhost:8000/api/v2/heartbeat >/dev/null 2>&1 \
       || curl -fsS http://localhost:8000/api/v1/heartbeat >/dev/null 2>&1; then
        echo -e "  ${G}ready${N}"
        break
    fi
    [[ $i -eq 60 ]] && { echo -e "  ${R}TIMEOUT${N}"; exit 1; }
    echo -ne "."
    sleep 2
done

echo -ne "${D}Waiting for Neo4j...${N}"
for i in $(seq 1 60); do
    if docker exec "$NEO4J" cypher-shell -u neo4j -p ${NEO4J_PASS} "RETURN 1" >/dev/null 2>&1; then
        echo -e "   ${G}ready${N}"
        break
    fi
    [[ $i -eq 60 ]] && { echo -e "   ${R}TIMEOUT${N}"; exit 1; }
    echo -ne "."
    sleep 2
done

# Common env passed into every smoke step that needs the DBs.
COMMON_ENV=(
    -e NEO4J_URI=bolt://${NEO4J}:7687
    -e NEO4J_USER=neo4j
    -e NEO4J_PASSWORD=${NEO4J_PASS}
    -e CHROMA_HOST=${CHROMA}
    -e CHROMA_PORT=8000
    -e OPENAI_API_KEY=dummy-not-used
    -e GOOGLE_API_KEY=
)

# ── 4. Smoke 1 — imports ─────────────────────────────────────────────────────
echo -e "${B}=== Smoke 1: imports work in the container ===${N}"
docker run --rm "$IMAGE" python -c "
from src.memory.agent import MemoryAgent
from agents.fact_check_agent import fact_check_claim
from agents.memory_agent import get_memory
print('IMPORT OK')
"

# ── 5. Smoke 2 — schema init ─────────────────────────────────────────────────
echo -e "${B}=== Smoke 2: schema init against sidecar Neo4j ===${N}"
docker run --rm --network "$NET" "${COMMON_ENV[@]}" "$IMAGE" python -c "
from src.config import settings
from src.memory.agent import MemoryAgent
MemoryAgent(settings).init_schema()
print('SCHEMA OK')
"

# ── 6. Smoke 3 — round-trip ──────────────────────────────────────────────────
echo -e "${B}=== Smoke 3: round-trip write/read on Neo4j + Chroma ===${N}"
docker run --rm --network "$NET" "${COMMON_ENV[@]}" "$IMAGE" python -c "
from src.config import settings
from src.memory.agent import MemoryAgent
m = MemoryAgent(settings)
m._graph.ensure_entity_exists('SmokeTestEntity')
ent = m.get_entity_by_name('SmokeTestEntity')
assert ent is not None and ent['name'].lower() == 'smoketestentity', ent
print('ROUND-TRIP OK')
"

# ── 7. Smoke 4 — Streamlit ───────────────────────────────────────────────────
echo -e "${B}=== Smoke 4: Streamlit health endpoint ===${N}"
docker run -d --name "$UI" --network "$NET" "${COMMON_ENV[@]}" \
    -p 8501:8501 \
    "$IMAGE" sh -c "cd /app/PredictionAgent && streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true" \
    >/dev/null

echo -ne "${D}Waiting for Streamlit...${N}"
for i in $(seq 1 60); do
    if curl -fsS http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        echo -e "   ${G}STREAMLIT OK${N}"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo -e "   ${R}TIMEOUT${N}"
        echo -e "${R}Streamlit container logs:${N}"
        docker logs "$UI"
        exit 1
    fi
    echo -ne "."
    sleep 2
done

echo -e "${G}=== ALL SMOKE TESTS PASSED ===${N}"
