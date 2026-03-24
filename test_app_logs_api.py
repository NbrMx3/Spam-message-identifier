"""
API tests for logs endpoints in app.py
Run with: python -m pytest test_app_logs_api.py -q
"""

from unittest.mock import patch

import app as app_module


def test_get_logs_returns_503_when_db_not_ready():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", False):
        response = client.get("/api/logs")

    assert response.status_code == 503
    data = response.get_json()
    assert data["success"] is False


def test_get_logs_validates_page_and_page_size():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", True):
        response = client.get("/api/logs?page=0&page_size=10")
        assert response.status_code == 400

        response = client.get("/api/logs?page=1&page_size=101")
        assert response.status_code == 400

        response = client.get("/api/logs?page=abc&page_size=10")
        assert response.status_code == 400


def test_get_logs_success_with_sort_and_search():
    client = app_module.app.test_client()
    fake_logs = [
        {
            "id": 1,
            "message": "free money",
            "predicted_label": "SPAM",
            "spam_probability": 0.95,
            "ham_probability": 0.05,
            "risk_level": "HIGH",
            "created_at": "2026-03-24T10:00:00+00:00",
        }
    ]

    with patch.object(app_module, "db_ready", True), patch.object(
        app_module, "get_prediction_logs", return_value=fake_logs
    ) as mock_get_logs, patch.object(
        app_module, "count_prediction_logs", return_value=1
    ) as mock_count:
        response = client.get("/api/logs?page=1&page_size=10&q=spam&sort=highest_spam")

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert len(data["logs"]) == 1
    assert data["pagination"]["total"] == 1
    assert data["sort"] == "highest_spam"

    mock_get_logs.assert_called_once_with(
        page=1,
        page_size=10,
        search="spam",
        sort="highest_spam",
    )
    mock_count.assert_called_once_with(search="spam")


def test_get_logs_handles_backend_exception():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", True), patch.object(
        app_module, "get_prediction_logs", side_effect=Exception("db failure")
    ):
        response = client.get("/api/logs")

    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert "db failure" in data["error"]


def test_get_logs_invalid_sort_returns_400():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", True), patch.object(
        app_module, "get_prediction_logs", side_effect=ValueError("Invalid sort value")
    ):
        response = client.get("/api/logs?sort=bad_sort")

    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Invalid sort value" in data["error"]


def test_export_logs_csv_returns_503_when_db_not_ready():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", False):
        response = client.get("/api/logs/export")

    assert response.status_code == 503


def test_export_logs_csv_success():
    client = app_module.app.test_client()
    fake_logs = [
        {
            "id": 1,
            "message": "limited offer",
            "predicted_label": "SPAM",
            "spam_probability": 0.88,
            "ham_probability": 0.12,
            "risk_level": "HIGH",
            "created_at": "2026-03-24T11:00:00+00:00",
        }
    ]

    with patch.object(app_module, "db_ready", True), patch.object(
        app_module, "get_prediction_logs_for_export", return_value=fake_logs
    ) as mock_export:
        response = client.get("/api/logs/export?q=offer&sort=newest&max_rows=50")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    disposition = response.headers.get("Content-Disposition", "")
    assert "attachment; filename=prediction_logs_" in disposition

    csv_text = response.get_data(as_text=True)
    assert "id,message,predicted_label,spam_probability,ham_probability,risk_level,created_at" in csv_text
    assert "limited offer" in csv_text

    mock_export.assert_called_once_with(search="offer", sort="newest", max_rows=50)


def test_export_logs_csv_validates_max_rows():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", True):
        response = client.get("/api/logs/export?max_rows=invalid")

    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False


def test_export_logs_csv_invalid_sort_returns_400():
    client = app_module.app.test_client()

    with patch.object(app_module, "db_ready", True), patch.object(
        app_module,
        "get_prediction_logs_for_export",
        side_effect=ValueError("Invalid sort value"),
    ):
        response = client.get("/api/logs/export?sort=bad_sort")

    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Invalid sort value" in data["error"]
