from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_analyze_tabular_rejects_unsupported_upload_type() -> None:
    response = client.post(
        "/analyze-tabular",
        files={"file": ("payload.json", b'{"foo":"bar"}', "application/json")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_tabular_rejects_empty_csv_upload() -> None:
    response = client.post(
        "/analyze-tabular",
        files={"file": ("empty.csv", b"user_id,merchant_id\n", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
