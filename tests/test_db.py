"""Tests for db.py – CRUD operations on targets and runs."""
import os
import tempfile
import time

import pytest

# Ensure we can import the project modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import Database


@pytest.fixture()
def db(tmp_path):
    """Create a fresh in-memory-like Database for each test."""
    db_path = str(tmp_path / "test.db")
    d = Database(db_path)
    d.init_db()
    return d


SAMPLE_TARGET = {
    "name": "Test Target",
    "base_url": "https://api.example.com",
    "api_key": "sk-test-key-123",
    "enabled": True,
    "interval_min": 30,
    "timeout_s": 30.0,
    "verify_ssl": False,
    "prompt": "hello",
    "anthropic_version": "2025-09-29",
    "max_models": 0,
    "source_url": "https://example.com",
}


class TestCreateTarget:
    def test_create_and_retrieve(self, db):
        target = db.create_target(SAMPLE_TARGET)
        assert target["name"] == "Test Target"
        assert target["base_url"] == "https://api.example.com"
        assert target["api_key"] == "sk-test-key-123"
        assert target["enabled"] in (True, 1)
        assert target["id"] is not None

    def test_create_returns_id(self, db):
        t = db.create_target(SAMPLE_TARGET)
        assert isinstance(t["id"], int) and t["id"] > 0

    def test_get_target_by_id(self, db):
        created = db.create_target(SAMPLE_TARGET)
        fetched = db.get_target(created["id"])
        assert fetched is not None
        assert fetched["name"] == created["name"]

    def test_get_nonexistent_target(self, db):
        assert db.get_target(9999) is None


class TestListTargets:
    def test_list_empty(self, db):
        assert db.list_targets() == []

    def test_list_multiple(self, db):
        db.create_target({**SAMPLE_TARGET, "name": "A"})
        db.create_target({**SAMPLE_TARGET, "name": "B"})
        targets = db.list_targets()
        assert len(targets) == 2
        names = {t["name"] for t in targets}
        assert names == {"A", "B"}


class TestUpdateTarget:
    def test_update_name(self, db):
        t = db.create_target(SAMPLE_TARGET)
        updated = db.update_target(t["id"], {"name": "New Name"})
        assert updated["name"] == "New Name"

    def test_update_enabled(self, db):
        t = db.create_target(SAMPLE_TARGET)
        updated = db.update_target(t["id"], {"enabled": False})
        assert updated["enabled"] in (False, 0)

    def test_update_nonexistent(self, db):
        result = db.update_target(9999, {"name": "X"})
        assert result is None

    def test_update_empty(self, db):
        t = db.create_target(SAMPLE_TARGET)
        result = db.update_target(t["id"], {})
        assert result["name"] == t["name"]


class TestDeleteTarget:
    def test_delete_existing(self, db):
        t = db.create_target(SAMPLE_TARGET)
        assert db.delete_target(t["id"]) is True
        assert db.get_target(t["id"]) is None

    def test_delete_nonexistent(self, db):
        assert db.delete_target(9999) is False

    def test_delete_removes_from_list(self, db):
        t = db.create_target(SAMPLE_TARGET)
        db.delete_target(t["id"])
        assert len(db.list_targets()) == 0


class TestRuns:
    def test_create_and_finish_run(self, db):
        t = db.create_target(SAMPLE_TARGET)
        now = time.time()
        run_id = db.create_run(
            target_id=t["id"],
            started_at=now,
            log_file="/tmp/test.jsonl",
        )
        assert isinstance(run_id, int) and run_id > 0

        db.finish_run(
            run_id,
            status="completed",
            finished_at=now + 10,
            total=5,
            success=4,
            fail=1,
            error=None,
        )

        runs = db.list_runs(t["id"], limit=10)
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        assert int(runs[0]["total"]) == 5

    def test_get_latest_run(self, db):
        t = db.create_target(SAMPLE_TARGET)
        now = time.time()
        db.create_run(target_id=t["id"], started_at=now, log_file="/tmp/a.jsonl")
        run2 = db.create_run(target_id=t["id"], started_at=now + 1, log_file="/tmp/b.jsonl")
        latest = db.get_latest_run(t["id"])
        assert latest is not None
        assert int(latest["id"]) == run2
