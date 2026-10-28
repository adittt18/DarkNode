"""Tabular ingestion utilities that map CSV/Excel records into the canonical graph schema."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from torch_geometric.data import HeteroData

from src.schema import build_hetero_data, validate_schema

SUPPORTED_TABULAR_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx", ".xls")
_DEFAULT_USER_SIGNATURE_COLUMN = "__user_signature__"
_MERCHANT_FEATURE_COLUMN = "__merchant_key__"
_DEVICE_FEATURE_COLUMN = "__device_key__"
_MISSING_VALUE_TOKEN = "__missing__"


class TabularIngestionError(ValueError):
    """Raised when tabular ingestion or conversion fails validation."""


@dataclass(frozen=True)
class TabularGraphConfig:
    """Configuration for converting tabular records into the canonical graph schema."""

    user_id_column: str | None = None
    merchant_id_column: str | None = None
    device_id_column: str | None = None
    shared_ip_column: str | None = None
    amount_column: str | None = None
    timestamp_column: str | None = None


def _clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    if len(set(renamed.columns)) != len(renamed.columns):
        raise TabularIngestionError("Tabular input has duplicate column names after normalization.")
    return renamed


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_EXTENSIONS:
        supported = ", ".join(SUPPORTED_TABULAR_EXTENSIONS)
        raise TabularIngestionError(f"Unsupported file type {suffix!r}. Supported: {supported}.")

    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".xlsx":
            frame = pd.read_excel(path, engine="openpyxl")
        else:
            frame = pd.read_excel(path, engine="xlrd")
    except ImportError as exc:
        raise TabularIngestionError(
            f"Missing dependency for parsing {suffix!r} files: {exc}."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive parser error fallback
        raise TabularIngestionError(f"Failed to parse tabular file {path.name!r}: {exc}.") from exc

    frame = _clean_column_names(frame)
    frame = frame.dropna(axis=0, how="all")
    if frame.empty:
        raise TabularIngestionError("Tabular input is empty after removing blank rows.")
    if len(frame.columns) == 0:
        raise TabularIngestionError("Tabular input must contain at least one column.")
    return frame


def _tokenize(value: Any) -> str:
    if pd.isna(value):
        return _MISSING_VALUE_TOKEN
    return str(value)


def _require_column(columns: set[str], column: str | None, label: str) -> str | None:
    if column is None:
        return None
    if column not in columns:
        raise TabularIngestionError(f"Configured {label} column {column!r} is missing from input.")
    return column


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TabularIngestionError(f"Unable to convert value {value!r} to float.") from exc


def build_graph_from_tabular_frame(
    frame: pd.DataFrame,
    config: TabularGraphConfig | None = None,
) -> tuple[HeteroData, dict[str, Any]]:
    """Convert a loaded tabular dataset into the canonical heterogeneous graph schema."""

    cfg = config or TabularGraphConfig()
    normalized = _clean_column_names(frame).dropna(axis=0, how="all")
    if normalized.empty:
        raise TabularIngestionError("Tabular input is empty after removing blank rows.")

    columns = set(normalized.columns)
    user_id_column = cfg.user_id_column or ("user_id" if "user_id" in columns else normalized.columns[0])
    if user_id_column not in columns:
        raise TabularIngestionError(f"Configured user_id column {user_id_column!r} is missing from input.")

    merchant_id_column = _require_column(columns, cfg.merchant_id_column, "merchant_id")
    device_id_column = _require_column(columns, cfg.device_id_column, "device_id")
    shared_ip_column = _require_column(columns, cfg.shared_ip_column, "shared_ip")
    amount_column = _require_column(columns, cfg.amount_column, "amount")
    timestamp_column = _require_column(columns, cfg.timestamp_column, "timestamp")

    user_keys = list(dict.fromkeys(_tokenize(value) for value in normalized[user_id_column]))
    if not user_keys:
        raise TabularIngestionError("No user records were found in the selected user_id column.")

    user_index = {user_key: idx for idx, user_key in enumerate(user_keys)}
    relationship_columns = {
        user_id_column,
        merchant_id_column,
        device_id_column,
        shared_ip_column,
        amount_column,
        timestamp_column,
    } - {None}
    user_attribute_columns = [
        column for column in normalized.columns if column not in relationship_columns
    ]

    first_user_rows = normalized.drop_duplicates(subset=[user_id_column], keep="first")
    users: list[dict[str, Any]] = []
    for _, row in first_user_rows.iterrows():
        user_key = _tokenize(row[user_id_column])
        signature_parts = [f"{column}={_tokenize(row[column])}" for column in user_attribute_columns]
        signature = "|".join(signature_parts) if signature_parts else f"id={user_key}"
        users.append({_DEFAULT_USER_SIGNATURE_COLUMN: signature})

    merchants: list[dict[str, Any]] = []
    merchant_index: dict[str, int] = {}
    transactions: list[dict[str, Any]] = []
    if merchant_id_column is not None:
        merchant_keys = [
            _tokenize(value)
            for value in normalized[merchant_id_column]
            if not pd.isna(value)
        ]
        merchant_keys = list(dict.fromkeys(merchant_keys))
        merchant_index = {merchant_key: idx for idx, merchant_key in enumerate(merchant_keys)}
        merchants = [{_MERCHANT_FEATURE_COLUMN: merchant_key} for merchant_key in merchant_keys]

        for _, row in normalized.iterrows():
            merchant_value = row[merchant_id_column]
            if pd.isna(merchant_value):
                continue
            transaction = {
                "user_id": user_index[_tokenize(row[user_id_column])],
                "merchant_id": merchant_index[_tokenize(merchant_value)],
                "amount": _safe_float(row[amount_column]) if amount_column else 0.0,
                "timestamp": _safe_float(row[timestamp_column]) if timestamp_column else 0.0,
            }
            transactions.append(transaction)

    devices: list[dict[str, Any]] = []
    device_index: dict[str, int] = {}
    shared_devices: list[dict[str, Any]] = []
    if device_id_column is not None:
        device_keys = [
            _tokenize(value)
            for value in normalized[device_id_column]
            if not pd.isna(value)
        ]
        device_keys = list(dict.fromkeys(device_keys))
        device_index = {device_key: idx for idx, device_key in enumerate(device_keys)}
        devices = [{_DEVICE_FEATURE_COLUMN: device_key} for device_key in device_keys]
        for _, row in normalized.iterrows():
            device_value = row[device_id_column]
            if pd.isna(device_value):
                continue
            shared_devices.append(
                {
                    "user_id": user_index[_tokenize(row[user_id_column])],
                    "device_id": device_index[_tokenize(device_value)],
                }
            )

    shared_ips: list[dict[str, Any]] = []
    if shared_ip_column is not None:
        ip_to_users: dict[str, list[int]] = {}
        for _, row in normalized.iterrows():
            ip_value = row[shared_ip_column]
            if pd.isna(ip_value):
                continue
            ip_key = _tokenize(ip_value)
            ip_to_users.setdefault(ip_key, []).append(user_index[_tokenize(row[user_id_column])])

        for user_list in ip_to_users.values():
            unique_users = list(dict.fromkeys(user_list))
            for source_user, target_user in combinations(unique_users, 2):
                shared_ips.append(
                    {"source_user_id": source_user, "target_user_id": target_user}
                )
                shared_ips.append(
                    {"source_user_id": target_user, "target_user_id": source_user}
                )

    data = build_hetero_data(
        users=users,
        merchants=merchants,
        devices=devices,
        transactions=transactions,
        shared_devices=shared_devices,
        shared_ips=shared_ips,
        user_categorical=(_DEFAULT_USER_SIGNATURE_COLUMN,),
        merchant_categorical=(_MERCHANT_FEATURE_COLUMN,),
        device_categorical=(_DEVICE_FEATURE_COLUMN,),
    )
    validate_schema(data)
    metadata = {
        "source_rows": int(len(normalized)),
        "source_columns": list(normalized.columns),
        "user_id_column": user_id_column,
        "merchant_id_column": merchant_id_column,
        "device_id_column": device_id_column,
        "shared_ip_column": shared_ip_column,
        "amount_column": amount_column,
        "timestamp_column": timestamp_column,
        "num_users": int(data["user"].num_nodes),
        "num_merchants": int(data["merchant"].num_nodes),
        "num_devices": int(data["device"].num_nodes),
        "num_transactions": int(data[("user", "transacts", "merchant")].num_edges),
    }
    return data, metadata


def build_graph_from_tabular_file(
    file_path: str | Path,
    config: TabularGraphConfig | None = None,
) -> tuple[HeteroData, dict[str, Any]]:
    """Read CSV/Excel input and convert it into the canonical graph schema."""

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise TabularIngestionError(f"Input file does not exist: {path}.")
    frame = _read_table(path)
    return build_graph_from_tabular_frame(frame, config=config)
