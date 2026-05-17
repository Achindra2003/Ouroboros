from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ouroboros.cli import main, cmd_sessions, cmd_export, cmd_delete, _get_printer, RichPrinter, QuietPrinter, JSONPrinter


class TestPrinters:
    def test_rich_printer_thought(self):
        p = RichPrinter()
        result = p.thought("think", "A test thought", mood="curious", energy=80)
        assert "THINK" in result
        assert "curious" in result
        assert "A test thought" in result

    def test_rich_printer_insight(self):
        p = RichPrinter()
        result = p.insight("A surfaced insight")
        assert "A surfaced insight" in result

    def test_rich_printer_state_bar(self):
        p = RichPrinter()
        result = p.state_bar(mood="curious", energy=80, depth=2, cycle=3)
        assert "curious" in result
        assert "80" in result

    def test_quiet_printer_thought(self):
        p = QuietPrinter()
        assert p.thought("think", "text") is None

    def test_quiet_printer_insight(self):
        p = QuietPrinter()
        assert p.insight("insight text") == "insight text"

    def test_quiet_printer_state_bar(self):
        p = QuietPrinter()
        assert p.state_bar() is None

    def test_json_printer_event(self):
        p = JSONPrinter()
        result = p.event("think", "text", mood="curious")
        data = __import__("json").loads(result)
        assert data["node"] == "think"
        assert data["mood"] == "curious"

    def test_json_printer_insight(self):
        p = JSONPrinter()
        result = p.insight("test insight")
        data = __import__("json").loads(result)
        assert data["type"] == "insight"

    def test_json_printer_state_bar(self):
        p = JSONPrinter()
        assert p.state_bar() is None

    def test_get_printer(self):
        assert isinstance(_get_printer("rich"), RichPrinter)
        assert isinstance(_get_printer("quiet"), QuietPrinter)
        assert isinstance(_get_printer("json"), JSONPrinter)


class TestCLICommands:
    def test_sessions_empty(self, capsys, tmp_path):
        db = str(tmp_path / "test.db")
        args = MagicMock(format="rich", db=db)
        cmd_sessions(args)
        captured = capsys.readouterr()
        assert "No sessions found" in captured.out

    def test_sessions_json(self, capsys, tmp_path):
        db = str(tmp_path / "test.db")
        args = MagicMock(format="json", db=db)
        cmd_sessions(args)
        captured = capsys.readouterr()
        data = __import__("json").loads(captured.out)
        assert isinstance(data, list)

    def test_export_not_found(self, tmp_path):
        db = str(tmp_path / "test.db")
        args = MagicMock(session_id="nonexistent", format="json", db=db)
        with pytest.raises(SystemExit):
            cmd_export(args)

    def test_delete_session(self, capsys, tmp_path):
        db = str(tmp_path / "test.db")
        from ouroboros.cli import cmd_delete
        args = MagicMock(session_id="nonexistent", db=db)
        cmd_delete(args)
        captured = capsys.readouterr()
        assert "deleted" in captured.out.lower()


class TestCLIMain:
    def test_help(self, capsys):
        with patch("sys.argv", ["ouroboros", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_run_no_seed_no_stdin(self, capsys):
        with patch("sys.argv", ["ouroboros", "run"]):
            with patch("ouroboros.cli._read_stdin", return_value=None):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_run_with_seed(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_graph = MagicMock()
        mock_graph.astream = MagicMock(return_value=AsyncMock())
        mock_graph.get_state = MagicMock(return_value=MagicMock(next=[]))

        async def mock_astream(*a, **kw):
            return
            yield

        with patch("ouroboros.cli._get_llm") as mock_llm, \
             patch("ouroboros.cli.create_ouroboros_graph", return_value=mock_graph), \
             patch("ouroboros.cli.SessionStore") as mock_store, \
             patch("sys.argv", ["ouroboros", "run", "test seed", "--no-steer", "--format", "quiet", "--db", db]):
            mock_store_instance = MagicMock()
            mock_store.return_value = mock_store_instance
            mock_graph.astream = mock_astream
            mock_graph.get_state = MagicMock(return_value=MagicMock(next=[]))
            main()
