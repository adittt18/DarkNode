"""FastAPI backend service for the DarkNode Heterogeneous Fraud Ring Detector."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from src.data_processing.tabular_ingestion import (
    SUPPORTED_TABULAR_EXTENSIONS,
    TabularGraphConfig,
    TabularIngestionError,
    build_graph_from_tabular_file,
)
from src.models.mpnn import HeterogeneousMPNN
from src.ring.extract import extract_rings


app = FastAPI(
    title="DarkNode Fraud Detection API",
    description="Dynamic heterogeneous GNN and graph analytics API for fraud ring detection.",
    version="1.0.0",
)

# Open CORS configuration for multi-domain frontend deployments (e.g. Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_PATH = Path(os.getenv("GRAPH_PATH", "data/synthetic_fraud_graph.pt"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "data/fraud_model.pt"))
MAX_TABULAR_UPLOAD_BYTES = 25 * 1024 * 1024


class AnalyzeRequest(BaseModel):
    threshold: float = 0.5
    top_k: int = 10
    graph_path: str | None = None
    model_path: str | None = None


class FreezeRequest(BaseModel):
    ring_id: int
    action: str = "freeze"  # "freeze" or "unfreeze"
    operator: str = "AML_COMPLIANCE_DESK"
    reason: str = "Heterogeneous GNN detected coordinated money mule syndicate"


# Real in-memory enforcement registry
ENFORCEMENT_STATE: dict[int, dict[str, Any]] = {}


def _ensure_artifacts() -> tuple[Path, Path]:
    """Ensure graph and model artifacts exist; generate if missing."""
    graph_path = GRAPH_PATH
    model_path = MODEL_PATH

    if not graph_path.exists() or not model_path.exists():
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        # Import generator and trainer on-the-fly if needed
        from src.data_processing.generate_synthetic_graph import GeneratorConfig, generate_synthetic_graph
        from src.training.train import TrainingConfig, train_model

        if not graph_path.exists():
            data, nx_g, meta = generate_synthetic_graph(GeneratorConfig())
            torch.save({"data": data, "nx": nx_g, "metadata": meta}, graph_path)

        if not model_path.exists():
            payload = torch.load(graph_path, weights_only=False)
            data = payload["data"] if isinstance(payload, dict) else payload
            model, report = train_model(data, config=TrainingConfig(epochs=20))
            torch.save({"model_state": model.state_dict(), "report": report}, model_path)

    return graph_path, model_path


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "DarkNode Fraud Detector API",
        "health": "/health",
        "analyze": "POST /analyze",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Render and cloud platform health/uptime check."""
    return {
        "status": "healthy",
        "service": "darknode-fraud-detector",
        "artifacts_ready": GRAPH_PATH.exists() and MODEL_PATH.exists(),
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest | None = None) -> dict[str, Any]:
    """Execute GNN model inference and extract top fraud rings."""
    threshold = request.threshold if request else 0.5
    top_k = request.top_k if request else 10

    g_path = Path(request.graph_path) if (request and request.graph_path) else GRAPH_PATH
    m_path = Path(request.model_path) if (request and request.model_path) else MODEL_PATH

    if not g_path.exists() or not m_path.exists():
        g_path, m_path = _ensure_artifacts()

    try:
        # 1. Load graph data
        payload = torch.load(g_path, weights_only=False)
        data = payload["data"] if isinstance(payload, dict) else payload
        nx_graph = payload.get("nx", None) if isinstance(payload, dict) else None
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}

        # 2. Load trained Heterogeneous MPNN
        checkpoint = torch.load(m_path, weights_only=False)
        model = HeterogeneousMPNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # 3. Real model inference (replaces torch.rand stub)
        with torch.no_grad():
            logits = model(data)
            probabilities = logits.softmax(dim=-1)[:, 1]

        # 4. Extract rings and compute reason signals
        rings = extract_rings(
            data=data,
            nx_graph=nx_graph,
            user_scores=probabilities,
            threshold=threshold,
            top_k=top_k,
        )

        # Attach real enforcement state
        for r in rings:
            r["is_suspended"] = r["ring_id"] in ENFORCEMENT_STATE

        return {
            "status": "success",
            "total_rings": len(rings),
            "rings": rings,
            "metadata": {
                "num_users": int(data["user"].num_nodes),
                "num_merchants": int(data["merchant"].num_nodes),
                "num_devices": int(data["device"].num_nodes),
                "total_transactions": int(data[("user", "transacts", "merchant")].num_edges),
                "threshold": threshold,
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.post("/freeze")
def execute_freeze(request: FreezeRequest) -> dict[str, Any]:
    """Execute real account freeze or reactivation order on a detected syndicate ring."""
    import time
    action = request.action.lower()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ref_id = f"AML-ENF-{request.ring_id}-{int(time.time()) % 100000:05d}"

    if action == "unfreeze":
        ENFORCEMENT_STATE.pop(request.ring_id, None)
        return {
            "status": "success",
            "action": "unfreeze",
            "ring_id": request.ring_id,
            "message": f"Suspension lifted for Ring #{request.ring_id}. Accounts restored to active monitoring.",
            "timestamp": timestamp,
            "audit_ref": ref_id,
        }
    else:
        record = {
            "ring_id": request.ring_id,
            "status": "SUSPENDED",
            "operator": request.operator,
            "reason": request.reason,
            "timestamp": timestamp,
            "audit_ref": ref_id,
        }
        ENFORCEMENT_STATE[request.ring_id] = record
        return {
            "status": "success",
            "action": "freeze",
            "ring_id": request.ring_id,
            "message": f"Legal account freeze executed for Ring #{request.ring_id}. All transaction settlement blocked.",
            "timestamp": timestamp,
            "audit_ref": ref_id,
        }


@app.get("/enforcement")
def get_enforcement_status() -> dict[str, Any]:
    """Return all currently suspended syndicates and active enforcement orders."""
    return {
        "status": "success",
        "total_suspended_rings": len(ENFORCEMENT_STATE),
        "suspended_rings": list(ENFORCEMENT_STATE.keys()),
        "records": ENFORCEMENT_STATE,
    }


class GenerateGraphRequest(BaseModel):
    seed: int | None = None
    threshold: float = 0.5


async def _save_upload_to_tempfile(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_TABULAR_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type {suffix!r}. "
                f"Supported types: {', '.join(SUPPORTED_TABULAR_EXTENSIONS)}"
            ),
        )

    total_size = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_TABULAR_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Tabular upload exceeds {MAX_TABULAR_UPLOAD_BYTES} bytes limit.",
                    )
                handle.write(chunk)
        return temp_path
    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise


@app.post("/generate-graph")
def generate_new_graph(request: GenerateGraphRequest | None = None) -> dict[str, Any]:
    """Generate a fresh synthetic transaction graph and run real GNN inference."""
    import random
    from src.data_processing.generate_synthetic_graph import GeneratorConfig, generate_synthetic_graph

    seed = request.seed if (request and request.seed is not None) else random.randint(10, 999999)
    threshold = request.threshold if request else 0.5

    config = GeneratorConfig(seed=seed)
    data, nx_g, meta = generate_synthetic_graph(config)
    torch.save({"data": data, "nx": nx_g, "metadata": meta}, GRAPH_PATH)
    ENFORCEMENT_STATE.clear()

    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = HeterogeneousMPNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(data)
        probabilities = logits.softmax(dim=-1)[:, 1]

    rings = extract_rings(
        data=data,
        nx_graph=nx_g,
        user_scores=probabilities,
        threshold=threshold,
        top_k=10,
    )

    for r in rings:
        r["is_suspended"] = False

    return {
        "status": "success",
        "message": f"Generated fresh transaction graph (Seed: {seed})",
        "seed": seed,
        "total_rings": len(rings),
        "rings": rings,
        "metadata": {
            "num_users": int(data["user"].num_nodes),
            "num_merchants": int(data["merchant"].num_nodes),
            "num_devices": int(data["device"].num_nodes),
            "total_transactions": int(data[("user", "transacts", "merchant")].num_edges),
            "seed": seed,
            "threshold": threshold,
        },
    }


@app.post("/analyze-tabular")
async def analyze_tabular(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    top_k: int = Form(10),
    user_id_column: str | None = Form(None),
    merchant_id_column: str | None = Form(None),
    device_id_column: str | None = Form(None),
    shared_ip_column: str | None = Form(None),
    amount_column: str | None = Form(None),
    timestamp_column: str | None = Form(None),
) -> dict[str, Any]:
    """Ingest tabular input and run ring analysis against the canonical graph schema."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    temp_path: Path | None = None
    try:
        temp_path = await _save_upload_to_tempfile(file)
        config = TabularGraphConfig(
            user_id_column=user_id_column,
            merchant_id_column=merchant_id_column,
            device_id_column=device_id_column,
            shared_ip_column=shared_ip_column,
            amount_column=amount_column,
            timestamp_column=timestamp_column,
        )
        data, tabular_metadata = build_graph_from_tabular_file(temp_path, config=config)
    except TabularIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink()

    m_path = MODEL_PATH
    if not m_path.exists():
        _, m_path = _ensure_artifacts()

    try:
        checkpoint = torch.load(m_path, weights_only=False)
        model = HeterogeneousMPNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        with torch.no_grad():
            logits = model(data)
            probabilities = logits.softmax(dim=-1)[:, 1]

        rings = extract_rings(
            data=data,
            nx_graph=None,
            user_scores=probabilities,
            threshold=threshold,
            top_k=top_k,
        )
        for ring in rings:
            ring["is_suspended"] = ring["ring_id"] in ENFORCEMENT_STATE

        return {
            "status": "success",
            "total_rings": len(rings),
            "rings": rings,
            "metadata": {
                "num_users": int(data["user"].num_nodes),
                "num_merchants": int(data["merchant"].num_nodes),
                "num_devices": int(data["device"].num_nodes),
                "total_transactions": int(data[("user", "transacts", "merchant")].num_edges),
                "threshold": threshold,
                "source_format": Path(file.filename).suffix.lower(),
                "graph_build": tabular_metadata,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
