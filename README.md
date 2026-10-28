# DarkNode: Dynamic Heterogeneous Graph Neural Network for Fraud-Ring Detection

**DarkNode** is an end-to-end graph intelligence platform engineered to detect coordinated fraud syndicates and multi-entity fraud rings across financial transaction networks. Built with PyTorch Geometric (PyG), NetworkX, and FastAPI.

---

## Problem Statement

Modern organized financial fraud rarely occurs through isolated rogue accounts. Fraudsters orchestrate distributed **fraud rings**—syndicates operating across synthetic identities that share physical devices, rotate proxy IP pools, and execute burst transaction velocities across targeted merchants.

Traditional rule-based engines (e.g., flagging accounts with burst activity or shared attributes) trigger massive false-positive rates, overwhelming compliance teams while missing subtle multi-hop structural collusion. **DarkNode** addresses this by modeling financial ecosystems as **dynamic heterogeneous graphs** and combining deep message-passing neural networks (Heterogeneous GraphSAGE MPNN) with topological community detection and multi-signal evidence scoring.

---

## Architecture & Pipeline Diagram

```
+-------------------------------------------------------------------------------------------------+
|                                        DARKNODE PIPELINE                                        |
+-------------------------------------------------------------------------------------------------+

  [ Ingestion / Generator ]
            |
            v
  +---------------------------------------------------------------------------------------------+
  | HETEROGENEOUS GRAPH (PyG HeteroData + NetworkX)                                             |
  |  - Node Types:   User (U), Merchant (M), Device (D)                                         |
  |  - Relations:    (U)-[transacts]->(M), (U)-[uses]->(D), (U)-[shares_ip]->(U)               |
  |  - Attributes:   Transaction Amounts, Microsecond Timestamps, Dynamic Velocity              |
  +---------------------------------------------------------------------------------------------+
            |
            +------------------------------------+
            |                                    |
            v                                    v
  +------------------------------------+   +------------------------------------+
  | HETEROGENEOUS MPNN (GraphSAGE)     |   | HEURISTIC RULE BASELINE            |
  |  - Multi-relation Conv Layers      |   |  - Device Sharing (>= 3 others)    |
  |  - Temporal Transaction Velocity   |   |  - Velocity (> 5 tx/hr)            |
  |  - Class-Imbalanced Focal Loss     |   +------------------------------------+
  +------------------------------------+                 |
            |                                            |
            v                                            v
  +-----------------------------------------------------------------------------+
  | USER-LEVEL FRAUD PROBABILITY INFERENCE & BENCHMARKING                       |
  +-----------------------------------------------------------------------------+
            |
            v
  +-----------------------------------------------------------------------------+
  | TOPOLOGICAL RING EXTRACTION (src/ring/extract.py)                           |
  |  - High-Risk Node Filtering (P(fraud) > threshold)                          |
  |  - Device & IP Subgraph Projection                                         |
  |  - Greedy Modularity Community Detection                                    |
  |  - Multi-Signal Evidence Scoring (Device %, IP pairs, Burst Rate, Model)   |
  +-----------------------------------------------------------------------------+
            |
            +------------------------------------+
            |                                    |
            v                                    v
  +------------------------------------+   +------------------------------------+
  | FASTAPI BACKEND (api/main.py)      |   | INTERACTIVE WEB UI (web/index.html)|
  |  - POST /analyze                   |   |  - Force-Directed Vis-Network      |
  |  - GET /health (Render Uptime)     |   |  - Ring Rank Sidebar & Evidence    |
  +------------------------------------+   |  - Account Freeze Simulation (Demo)|
                                           +------------------------------------+
```

---

## Baseline Comparison Note & Metric Analysis

To evaluate model performance, DarkNode includes a rule-based heuristic baseline (`src/ring/baseline.py`) evaluated against the same holdout validation split:

| Metric | Rule-Based Heuristic | DarkNode Heterogeneous GNN | Performance Analysis |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 84.0% | **99.5%** | **+15.5 percentage points gain** |
| **ROC-AUC** | 1.000 | **1.000** | Saturated on synthetic signals (see note below) |
| **Average Precision** | 1.000 | **1.000** | Saturated on synthetic signals (see note below) |
| **Precision** | 41.9% | **95.0%** | **+53.1 percentage points (false alarm share drops from 58.1% to 5.0%)** |
| **F1-Score** | 0.591 | **0.974** | **+0.383 F1 improvement (from 0.591 to 0.974)** |

### Why ROC-AUC and AP Don't Tell the Full Story
On synthetic benchmark datasets with clean demarcations, ranked discrimination metrics like **ROC-AUC** and **Average Precision (AP)** both saturate near `1.000` because extreme burst velocity pushes true fraudsters into the top ranking deciles for both approaches.

The **critical operational gap** is revealed in **Precision (41.9% vs. 95.0%)** and **F1-Score (0.591 vs. 0.974)**:
- **Heuristic Baseline Limitations**: Simple thresholds (such as sharing a device with 3+ users or exceeding 5 tx/hr) trigger on non-fraudulent accounts that happen to share subnets or devices in the graph topology. As a result, **58.1% of the heuristic flags are false alarms** (41.9% precision), burdening review queues.
- **GNN Disambiguation**: The Heterogeneous MPNN synthesizes multi-hop neighborhood context (e.g. *who* else connects to the device, *which* merchant cohort is targeted, and *how* the temporal network evolves), distinguishing benign overlap from coordinated syndicates and reducing the false-alarm share among flags to 5.0% (95.0% precision).

---

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/adittt18/DarkNode.git
cd DarkNode

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Graph Data
 
Create a synthetic heterogeneous graph with seeded fraud rings and microsecond transaction bursts:
 
```bash
python -m src.data_processing.generate_synthetic_graph --output data/synthetic_fraud_graph.pt
```
 
### 3. Run Self-Checks & Tests
 
```bash
python -m pytest
```
 
### 4. Train Model & Benchmark
 
Train the Heterogeneous GraphSAGE MPNN detector and benchmark against the heuristic baseline:
 
```bash
python -m src.training.train data/synthetic_fraud_graph.pt --epochs 20 --output data/fraud_model.pt
```
 
### 5. Launch Backend API
 
```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive API Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Run inference & extract rings: `POST http://localhost:8000/analyze`
- Ingest tabular files and analyze rings: `POST http://localhost:8000/analyze-tabular` (multipart upload)
 
### 6. Launch Web Dashboard
 
Simply open `web/index.html` directly in any web browser, or serve it locally:
```bash
python -m http.server 3000 --directory web
```
Navigate to `http://localhost:3000` to inspect the live force-directed fraud ring visualization.

---

## Tabular Ingestion (Backend API)

`POST /analyze-tabular` accepts `.csv`, `.xlsx`, and `.xls` uploads and converts rows into the existing heterogeneous graph schema before running the same ring extraction pipeline used by `/analyze`.

- **Default graph-building convention**
  - Rows represent user records.
  - `user_id` (or the first column when `user_id` is absent) is used as the user identifier.
  - Non-relationship columns are folded into user-node attributes/features.
- **Optional relationship columns** (multipart form fields):  
  `merchant_id_column`, `device_id_column`, `shared_ip_column`, `amount_column`, `timestamp_column`, `user_id_column`
  - `merchant_id_column` creates `user -> transacts -> merchant` edges.
  - `device_id_column` creates `user -> uses -> device` edges.
  - `shared_ip_column` creates pairwise `user -> shares_ip -> user` edges among users sharing the same IP value.
  - `amount_column` and `timestamp_column` are mapped to transaction edge attributes.
- **Validation and errors**
  - Unsupported file extensions return HTTP 400.
  - Malformed, empty, or invalid tabular inputs return HTTP 400 with descriptive error details.
  - Uploads are size-limited server-side (HTTP 413 when exceeded).

---

## Deployment & Production Links

| Service | Platform | Production URL | Status |
| :--- | :--- | :--- | :---: |
| **Interactive Dashboard (Frontend)** | **Vercel** | **[https://darknode-app.vercel.app](https://darknode-app.vercel.app)** <br> *(Mirror: [https://darknode-ai.vercel.app](https://darknode-ai.vercel.app))* | 🟢 Live |
| **GNN Inference Engine (Backend)** | **Render** | **[https://darknode.onrender.com](https://darknode.onrender.com)** <br> *(Health check: `/health`)* | 🟢 Configured |

---

## Deployment Instructions

### Backend (Render)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/adittt18/DarkNode)

DarkNode includes a root `render.yaml` blueprint for one-click deployment:
1. Click the button above or in [Render Dashboard](https://dashboard.render.com/) select **New +** > **Blueprint** linked to `https://github.com/adittt18/DarkNode`.
2. Configuration auto-loaded from `render.yaml`:
   - **Service Name**: `darknode`
   - **Environment**: Python 3 (3.11.9)
   - **Root Directory**: `.`
   - **Build Command**: `pip install -r api/requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
3. **Resource Optimization Note**:
   - `api/requirements.txt` is pre-configured with CPU-only PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`) to keep memory below 512 MB.
   - Low-memory environment variable `WEB_CONCURRENCY=1` is enabled by default.

### Frontend (Vercel)
The static intelligence console is deployed to Vercel:
- **Production Domains**: [https://darknode-app.vercel.app](https://darknode-app.vercel.app) & [https://darknode-ai.vercel.app](https://darknode-ai.vercel.app)
- Pre-configured to point securely (HTTPS) to `https://darknode.onrender.com`.
- CLI Deployment command:
  ```bash
  cd web
  vercel --prod --yes
  ```

---

## Dataset Scope & Real-World Adaptation

- **Default Benchmark (Synthetic)**: By default, `src/data_processing/generate_synthetic_graph.py` generates deterministic, scalable fraud networks containing user, device, merchant nodes, burst timestamps, and ground-truth ring memberships.
- **Real-World AML Data Limitations**: Public anti-money-laundering datasets like **Elliptic (Bitcoin)** or **IBM AML** primarily capture transaction flows (address-to-address or entity-to-entity) and **lack physical device or IP telemetry**. 
- **Adapting to Real Environments**: DarkNode is designed as an experimental graph prototype for multi-entity syndicate fraud detection where richer telemetry (device fingerprints, browser hashes, IP subnets, merchant gateways) can be collected. For pure transaction datasets without device data, the schema can be projected into bipartite user-merchant or account-account flow subgraphs, though graph neural networks demonstrate greater detection lift when multi-entity relationships are present.

---

## Interactive UI Features

- **Force-Directed Graph**: Powered by `vis-network` with smooth physics stabilization and node clustering.
- **Evidence Inspector**: Breakdowns of shared device %, shared IP pairs, burst velocity, GNN fraud confidence, and new accounts.
- **Node-Level Focus**: Click individual user accounts to zoom and isolate their neighborhood graph.
- **Account Freeze Simulation (Demo)**: A simulated mitigation action designed for hackathon walkthroughs to demonstrate operational compliance workflows.
- **JSON Export**: One-click download of structured ring payloads for compliance auditing.

---

## Acknowledgements & Attribution

This project builds upon foundational research and graph schema designs from [Apate](https://github.com/roboticol/Apate), expanded here into the **DarkNode** prototype with community ring extraction, heuristic benchmarking, a lightweight FastAPI backend, and an interactive intelligence interface.

---

## License

This project is licensed under the [MIT License](LICENSE).
