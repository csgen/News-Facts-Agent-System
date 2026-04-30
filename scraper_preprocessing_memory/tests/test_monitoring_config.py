"""Smoke tests for the monitoring stack's static config files.

Catches typos / structural breakage that would otherwise only surface at
`docker compose up` time. Includes a regression for the exact bug we hit
(panels referencing a Prometheus datasource UID that didn't match what
provisioning actually set).

If the test files are run from outside the repo (e.g. an installed package),
the docker/ tree may not exist — tests skip cleanly in that case.
"""

import json
import pathlib

import pytest

# Tests live at .../scraper_preprocessing_memory/tests/test_monitoring_config.py
# The docker/ tree is at the repo root (parents[2]).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DOCKER_DIR = _REPO_ROOT / "docker"

pytestmark = pytest.mark.skipif(
    not _DOCKER_DIR.exists(),
    reason="docker/ tree not present — running outside the monorepo layout",
)


# ──────────────────────────────────────────────────────────────────────────────
# Grafana dashboard JSON
# ──────────────────────────────────────────────────────────────────────────────


_DASHBOARD_PATH = _DOCKER_DIR / "grafana" / "provisioning" / "dashboards" / "news-facts-system.json"


def test_dashboard_json_parses():
    """The dashboard JSON must be valid JSON — Grafana auto-loads provisioned files
    only if they parse, otherwise it skips them silently."""
    if not _DASHBOARD_PATH.exists():
        pytest.skip(f"{_DASHBOARD_PATH} not present")
    with _DASHBOARD_PATH.open(encoding="utf-8") as f:
        d = json.load(f)
    # Sanity: it has panels and a uid
    assert "panels" in d
    assert d.get("uid"), "dashboard.uid is required for stable provisioning"


def test_dashboard_panels_use_pinned_prometheus_uid():
    """Every Prometheus-referencing panel must say uid='prometheus'.

    Regression: panels previously referenced uid='prometheus' but the provisioned
    datasource didn't pin a UID, so Grafana auto-generated a random one and
    every panel rendered 'datasource not found'. The fix was to pin
    `uid: prometheus` in datasources.yml. This test ensures both sides stay
    in sync."""
    if not _DASHBOARD_PATH.exists():
        pytest.skip(f"{_DASHBOARD_PATH} not present")
    with _DASHBOARD_PATH.open(encoding="utf-8") as f:
        d = json.load(f)

    bad = []
    for panel in d.get("panels", []):
        if panel.get("type") == "row":
            continue  # row markers don't have datasources
        ds = panel.get("datasource")
        if isinstance(ds, dict) and ds.get("type") == "prometheus":
            if ds.get("uid") != "prometheus":
                bad.append((panel.get("id"), panel.get("title"), ds.get("uid")))
    assert not bad, (
        "Panels referencing Prometheus must use uid='prometheus' to match the "
        "provisioned datasource. Offenders: " + repr(bad)
    )


def test_dashboard_panels_have_titles_and_unique_ids():
    """Every non-row panel needs a title (for the report) and a unique numeric id."""
    if not _DASHBOARD_PATH.exists():
        pytest.skip(f"{_DASHBOARD_PATH} not present")
    with _DASHBOARD_PATH.open(encoding="utf-8") as f:
        d = json.load(f)

    seen_ids = set()
    for panel in d.get("panels", []):
        if panel.get("type") == "row":
            continue
        assert panel.get("title"), f"Panel {panel.get('id')} missing title"
        assert panel.get("id") is not None, f"Panel {panel.get('title')!r} missing id"
        assert panel["id"] not in seen_ids, f"Duplicate panel id {panel['id']}"
        seen_ids.add(panel["id"])


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus scrape config
# ──────────────────────────────────────────────────────────────────────────────


_PROMETHEUS_YML_PATH = _DOCKER_DIR / "prometheus" / "prometheus.yml"


def test_prometheus_yml_has_expected_jobs():
    """The three scrape jobs we depend on must be present.

    `ui` → app counters/histograms, `metrics-collector` → DB gauges,
    `cadvisor` → per-container infra metrics. node-exporter is intentionally
    absent (Docker-Desktop incompatibility), so we don't expect it here."""
    if not _PROMETHEUS_YML_PATH.exists():
        pytest.skip(f"{_PROMETHEUS_YML_PATH} not present")
    yaml = pytest.importorskip("yaml")
    with _PROMETHEUS_YML_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    job_names = {job["job_name"] for job in cfg.get("scrape_configs", [])}
    expected = {"ui", "metrics-collector", "cadvisor"}
    missing = expected - job_names
    assert not missing, f"Missing scrape jobs: {missing}. Found: {job_names}"


def test_prometheus_metrics_collector_uses_5min_interval():
    """metrics-collector polls DBs every 5 min; Prometheus scrape interval must
    match so the gauge values aren't stale by the time they reach the dashboard."""
    if not _PROMETHEUS_YML_PATH.exists():
        pytest.skip(f"{_PROMETHEUS_YML_PATH} not present")
    yaml = pytest.importorskip("yaml")
    with _PROMETHEUS_YML_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mc_job = next(
        (j for j in cfg["scrape_configs"] if j["job_name"] == "metrics-collector"),
        None,
    )
    assert mc_job is not None, "metrics-collector job missing"
    assert mc_job.get("scrape_interval") == "300s", (
        f"metrics-collector scrape_interval should be 300s, got {mc_job.get('scrape_interval')}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Compose files parse cleanly
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "compose_filename",
    ["docker-compose.yml", "docker-compose.local.yml", "docker-compose.monitoring.yml"],
)
def test_compose_file_parses(compose_filename):
    path = _DOCKER_DIR / compose_filename
    if not path.exists():
        pytest.skip(f"{path} not present")
    yaml = pytest.importorskip("yaml")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "services" in cfg, f"{compose_filename} has no services key"


def test_monitoring_compose_has_expected_services():
    """The monitoring overlay must expose the four services we documented in the
    README (prometheus, grafana, cadvisor, metrics-collector). node-exporter is
    intentionally not in this list."""
    path = _DOCKER_DIR / "docker-compose.monitoring.yml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    yaml = pytest.importorskip("yaml")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    services = set(cfg.get("services", {}).keys())
    expected = {"prometheus", "grafana", "cadvisor", "metrics-collector"}
    missing = expected - services
    assert not missing, f"docker-compose.monitoring.yml missing services: {missing}"


# ──────────────────────────────────────────────────────────────────────────────
# Grafana datasource provisioning
# ──────────────────────────────────────────────────────────────────────────────


_DATASOURCES_PATH = _DOCKER_DIR / "grafana" / "provisioning" / "datasources" / "datasources.yml"


def test_datasources_pin_prometheus_uid():
    """The provisioned Prometheus datasource MUST have uid='prometheus' so the
    dashboard panels (which hard-code that uid) can resolve it."""
    if not _DATASOURCES_PATH.exists():
        pytest.skip(f"{_DATASOURCES_PATH} not present")
    yaml = pytest.importorskip("yaml")
    with _DATASOURCES_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prom = next(
        (d for d in cfg.get("datasources", []) if d.get("type") == "prometheus"),
        None,
    )
    assert prom is not None, "No Prometheus datasource provisioned"
    assert prom.get("uid") == "prometheus", (
        f"Prometheus datasource uid must be pinned to 'prometheus', got "
        f"{prom.get('uid')!r}. Without this pin, panels render "
        f"'datasource not found' (we hit this bug already once)."
    )
