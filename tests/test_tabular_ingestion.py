from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.data_processing.tabular_ingestion import (
    TabularGraphConfig,
    TabularIngestionError,
    build_graph_from_tabular_file,
)


def test_csv_ingestion_builds_expected_graph(tmp_path) -> None:
    table = pd.DataFrame(
        [
            {
                "account_id": "u1",
                "merchant_id": "m1",
                "device_id": "d1",
                "ip": "10.0.0.1",
                "amount": 100.0,
                "timestamp": 1.0,
                "segment": "new",
            },
            {
                "account_id": "u2",
                "merchant_id": "m2",
                "device_id": "d1",
                "ip": "10.0.0.1",
                "amount": 40.0,
                "timestamp": 2.0,
                "segment": "returning",
            },
            {
                "account_id": "u3",
                "merchant_id": "m2",
                "device_id": "d2",
                "ip": "10.0.0.2",
                "amount": 10.0,
                "timestamp": 3.0,
                "segment": "new",
            },
        ]
    )
    file_path = tmp_path / "input.csv"
    table.to_csv(file_path, index=False)

    data, metadata = build_graph_from_tabular_file(
        file_path,
        config=TabularGraphConfig(
            user_id_column="account_id",
            merchant_id_column="merchant_id",
            device_id_column="device_id",
            shared_ip_column="ip",
            amount_column="amount",
            timestamp_column="timestamp",
        ),
    )

    assert data["user"].num_nodes == 3
    assert data["merchant"].num_nodes == 2
    assert data["device"].num_nodes == 2
    assert data[("user", "transacts", "merchant")].num_edges == 3
    assert data[("user", "uses", "device")].num_edges == 3
    assert data[("user", "shares_ip", "user")].num_edges == 2
    assert metadata["user_id_column"] == "account_id"
    assert metadata["num_transactions"] == 3


def test_excel_ingestion_builds_graph(tmp_path) -> None:
    pytest.importorskip("openpyxl")

    table = pd.DataFrame(
        [
            {"user_id": "u1", "merchant_id": "m1", "amount": 12.0, "timestamp": 1},
            {"user_id": "u2", "merchant_id": "m2", "amount": 15.0, "timestamp": 2},
        ]
    )
    file_path = tmp_path / "input.xlsx"
    table.to_excel(file_path, index=False)

    data, metadata = build_graph_from_tabular_file(
        file_path,
        config=TabularGraphConfig(
            merchant_id_column="merchant_id",
            amount_column="amount",
            timestamp_column="timestamp",
        ),
    )

    assert data["user"].num_nodes == 2
    assert data["merchant"].num_nodes == 2
    assert data[("user", "transacts", "merchant")].num_edges == 2
    assert metadata["source_rows"] == 2


def test_empty_table_is_rejected(tmp_path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("user_id,merchant_id\n", encoding="utf-8")

    with pytest.raises(TabularIngestionError, match="empty"):
        build_graph_from_tabular_file(empty_csv)


def test_unsupported_extension_is_rejected(tmp_path) -> None:
    invalid_file = tmp_path / "input.json"
    invalid_file.write_text('{"user_id":"u1"}', encoding="utf-8")

    with pytest.raises(TabularIngestionError, match="Unsupported file type"):
        build_graph_from_tabular_file(invalid_file)


def test_default_mapping_without_relationship_columns(tmp_path) -> None:
    table = pd.DataFrame(
        [
            {"customer": "u1", "segment": "new"},
            {"customer": "u2", "segment": "repeat"},
        ]
    )
    file_path = tmp_path / "users.csv"
    table.to_csv(file_path, index=False)

    data, metadata = build_graph_from_tabular_file(file_path)

    assert metadata["user_id_column"] == "customer"
    assert data["user"].num_nodes == 2
    assert data[("user", "transacts", "merchant")].num_edges == 0
    assert data[("user", "uses", "device")].num_edges == 0
    assert data[("user", "shares_ip", "user")].num_edges == 0
