"""Tests for app.py – API endpoint integration tests."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Must set env BEFORE importing app
os.environ.pop("API_MONITOR_TOKEN", None)


SAMPLE_TARGET = {
    "name": "Test Target",
    "base_url": "https://api.example.com",
    "api_key": "sk-test-key-123",
}


@pytest.fixture()
def client(tmp_path):
    """Create a test client with a temporary database."""
    import app as app_module
    from db import Database

    # Override database and monitor with temp paths
    original_db = app_module.db
    original_monitor = app_module.monitor

    test_db = Database(str(tmp_path / "test.db"))
    test_db.init_db()
    app_module.db = test_db

    from monitor import MonitorService
    test_monitor = MonitorService(
        db=test_db,
        log_dir=str(tmp_path / "logs"),
    )
    app_module.monitor = test_monitor

    from fastapi.testclient import TestClient
    with TestClient(app_module.app, raise_server_exceptions=False) as c:
        yield c

    # Restore originals
    app_module.db = original_db
    app_module.monitor = original_monitor


class TestHealthEndpoint:
    def test_health_ok(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True


class TestTargetsCRUD:
    def _create(self, client):
        """Helper to create a target and return its id."""
        res = client.post("/api/targets", json=SAMPLE_TARGET)
        assert res.status_code == 200, f"Create failed: {res.text}"
        data = res.json()
        item = data.get("item", data)
        return item["id"]

    def test_list_targets_empty(self, client):
        res = client.get("/api/targets")
        assert res.status_code == 200

    def test_create_target(self, client):
        res = client.post("/api/targets", json=SAMPLE_TARGET)
        assert res.status_code == 200, f"Create failed: {res.text}"
        data = res.json()
        item = data.get("item", data)
        assert item["name"] == "Test Target"
        assert "id" in item

    def test_get_target(self, client):
        target_id = self._create(client)
        res = client.get(f"/api/targets/{target_id}")
        assert res.status_code == 200

    def test_get_nonexistent_target(self, client):
        res = client.get("/api/targets/99999")
        assert res.status_code == 404

    def test_patch_target(self, client):
        target_id = self._create(client)
        res = client.patch(f"/api/targets/{target_id}", json={"name": "Updated"})
        assert res.status_code == 200

    def test_delete_target(self, client):
        target_id = self._create(client)
        res = client.delete(f"/api/targets/{target_id}")
        assert res.status_code == 200
        assert res.json()["ok"] is True
        # Verify deleted
        res = client.get(f"/api/targets/{target_id}")
        assert res.status_code == 404

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/targets/99999")
        assert res.status_code == 404


class TestAuth:
    """Test Bearer token authentication."""

    def test_no_token_configured_allows_access(self, client):
        """When API_MONITOR_TOKEN is not set, all endpoints should be accessible."""
        res = client.get("/api/targets")
        assert res.status_code == 200

    def test_health_always_accessible(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200


class TestDashboard:
    def test_dashboard_returns_stats(self, client):
        res = client.get("/api/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert "total_targets" in data
        assert "healthy" in data
