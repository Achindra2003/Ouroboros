from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

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


class TestCorsOriginsParsing:
    def test_wildcard_default(self):
        from ouroboros.config import Settings

        assert Settings(allowed_origins="*").cors_origins == ["*"]

    def test_comma_separated(self):
        from ouroboros.config import Settings

        s = Settings(allowed_origins="https://a.com, https://b.com")
        assert s.cors_origins == ["https://a.com", "https://b.com"]

    def test_empty_falls_back_to_wildcard(self):
        from ouroboros.config import Settings

        assert Settings(allowed_origins="").cors_origins == ["*"]


class TestDemoSafety:
    def test_concurrent_session_limit_returns_429(self, client):
        import ouroboros.server as srv

        with patch.object(srv.settings, "max_concurrent_sessions", 1), \
             patch.object(srv, "_running", {"already-running"}):
            r = client.post("/api/start", params={"seed": "hi", "mode": "explore"})
        assert r.status_code == 429

    def test_demo_mode_clamps_cycles(self, client):
        import ouroboros.server as srv

        with patch.object(srv, "_run_graph", new=AsyncMock()), \
             patch.object(srv, "_running", set()), \
             patch.object(srv.settings, "demo_mode", True), \
             patch.object(srv.settings, "max_demo_cycles", 3):
            r = client.post(
                "/api/start",
                params={"seed": "hi", "mode": "explore"},
                json={"max_loop_guard": 50},
            )
            assert r.status_code == 200
            sid = r.json()["session_id"]
            assert srv._sessions[sid]["config"].max_loop_guard == 3
