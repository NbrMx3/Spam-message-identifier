"""
Unit tests for db.py
Run with: python -m pytest test_db.py -q
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import db


def make_mock_connection(mock_cursor):
    """Create a mocked connection object that supports context manager usage."""
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None

    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = mock_cursor
    cursor_context.__exit__.return_value = None
    mock_conn.cursor.return_value = cursor_context

    return mock_conn


def test_get_database_url_strips_whitespace():
    with patch("db.os.getenv", return_value="  postgres://example  "):
        assert db.get_database_url() == "postgres://example"


def test_get_connection_raises_when_env_missing():
    with patch("db.os.getenv", return_value=""):
        with pytest.raises(ValueError, match="DATABASE_URL is not set"):
            db.get_connection()


@patch("db.psycopg2.connect")
def test_get_connection_calls_psycopg(mock_connect):
    with patch("db.os.getenv", return_value="postgres://example"):
        db.get_connection()

    mock_connect.assert_called_once_with("postgres://example")


def test_test_connection_returns_database_info():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "database": "neondb",
        "version": "PostgreSQL 16"
    }

    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        result = db.test_connection()

    assert result["database"] == "neondb"
    assert "PostgreSQL" in result["version"]


def test_init_db_executes_create_table_and_commit():
    mock_cursor = MagicMock()
    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        db.init_db()

    assert mock_cursor.execute.called
    execute_sql = mock_cursor.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS prediction_logs" in execute_sql
    mock_conn.commit.assert_called_once()


def test_save_prediction_inserts_and_returns_id():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [123]
    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        row_id = db.save_prediction(
            message="test message",
            predicted_label="SPAM",
            spam_probability=0.91,
            ham_probability=0.09,
            risk_level="HIGH",
        )

    assert row_id == 123
    assert mock_cursor.execute.called
    execute_sql, execute_params = mock_cursor.execute.call_args[0]
    assert "INSERT INTO prediction_logs" in execute_sql
    assert execute_params[0] == "test message"
    assert execute_params[1] == "SPAM"
    assert execute_params[2] == 0.91
    assert execute_params[3] == 0.09
    assert execute_params[4] == "HIGH"
    assert isinstance(execute_params[5], datetime)
    mock_conn.commit.assert_called_once()


def test_get_prediction_logs_returns_iso_datetimes_without_search():
    created_at = datetime(2026, 3, 24, 10, 30, tzinfo=timezone.utc)
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {
            "id": 1,
            "message": "hello",
            "predicted_label": "HAM",
            "spam_probability": 0.1,
            "ham_probability": 0.9,
            "risk_level": "LOW",
            "created_at": created_at,
        }
    ]
    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        logs = db.get_prediction_logs(page=2, page_size=5, search="")

    assert len(logs) == 1
    assert logs[0]["id"] == 1
    assert logs[0]["created_at"] == created_at.isoformat()

    execute_sql, execute_params = mock_cursor.execute.call_args[0]
    assert "ORDER BY created_at DESC" in execute_sql
    assert execute_params == (5, 5)


def test_get_prediction_logs_with_search_uses_ilike_filter():
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        logs = db.get_prediction_logs(page=1, page_size=10, search="spam")

    assert logs == []

    execute_sql, execute_params = mock_cursor.execute.call_args[0]
    assert "ILIKE" in execute_sql
    assert execute_params[0] == "%spam%"
    assert execute_params[1] == "%spam%"
    assert execute_params[2] == "%spam%"


def test_count_prediction_logs_without_search():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"total": 42}
    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        total = db.count_prediction_logs(search="")

    assert total == 42
    execute_sql = mock_cursor.execute.call_args[0][0]
    assert "COUNT(*)" in execute_sql


def test_count_prediction_logs_with_search():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"total": 3}
    mock_conn = make_mock_connection(mock_cursor)

    with patch("db.get_connection", return_value=mock_conn):
        total = db.count_prediction_logs(search="ham")

    assert total == 3
    execute_sql, execute_params = mock_cursor.execute.call_args[0]
    assert "ILIKE" in execute_sql
    assert execute_params == ("%ham%", "%ham%", "%ham%")
