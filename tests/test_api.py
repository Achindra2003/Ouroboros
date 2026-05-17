from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("ouroboros.server.store") as mock_store:
        mock_store.list_sessions.return_value = []
        mock_store.get_session.return_value = None
        mock_store.delete_session.return_value = None
        from ouroboros.server import app
        with TestClient(app) as c:
            yield c


class TestModes:
    def test_get_modes(self, client):
        r = client.get("/api/modes")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5
        modes = [m["mode"] for m in data]
        assert "explore" in modes
        assert "analyze" in modes
        assert "create" in modes
        assert "solve" in modes
        assert "philosophize" in modes

    def test_modes_have_labels(self, client):
        r = client.get("/api/modes")
        for mode in r.json():
            assert "label" in mode
            assert "description" in mode
            assert "icon" in mode
            assert "config" in mode


class TestSessions:
    def test_list_sessions(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_nonexistent_session(self, client):
        r = client.get("/api/sessions/nonexistent123")
        assert r.status_code == 404

    def test_delete_session(self, client):
        r = client.delete("/api/sessions/nonexistent123")
        assert r.status_code == 200


class TestState:
    def test_get_state_idle(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data


class TestFrontend:
    def test_serves_index(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Ouroboros" in r.text
