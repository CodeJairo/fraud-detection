from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ui.app import fetch_data


def test_fetch_data_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy"}

    with patch("src.ui.app.requests.get", return_value=mock_response):
        data = fetch_data("/health")
        assert data == {"status": "healthy"}


def test_fetch_data_404_error():
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Usuario no encontrado"}

    with patch("src.ui.app.requests.get", return_value=mock_response):
        data = fetch_data("/users/UNKNOWN/risk-profile")
        assert data["status_code"] == 404
        assert "Usuario no encontrado" in data["error"]


def test_fetch_data_connection_error():
    with patch("src.ui.app.requests.get", side_effect=requests.exceptions.ConnectionError):
        data = fetch_data("/health")
        assert data == {"connection_error": True}


def test_app_module_imports():
    import src.ui.app as ui_app
    assert hasattr(ui_app, "fetch_data")
    assert hasattr(ui_app, "show_connection_error")
