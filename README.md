<div align="center">

# 🛰️ OceanSentinel

### AI-Powered Maritime Oil Spill Detection & Vessel Attribution System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SIH 2026 — Problem Statement #26143 (NTRO)**

*Detect oil spills from SAR satellite imagery → Trace spill origin via ocean drift modeling → Identify responsible vessels using AIS correlation*

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Demo](#-demo) · [API Reference](#-api-reference) · [Tech Stack](#-tech-stack)

</div>

---

## 📋 Overview

OceanSentinel is a full-stack maritime intelligence platform that automates the detection, tracking, and attribution of marine oil spills. The system processes Sentinel-1 SAR (Synthetic Aperture Radar) imagery through a deep learning segmentation model, simulates oil drift trajectories using Lagrangian physics, and correlates results with vessel AIS (Automatic Identification System) data to rank suspect vessels.

### The Problem

Marine oil spills cause catastrophic environmental damage to ocean ecosystems. Current detection relies heavily on manual analysis, delayed reporting, and limited vessel accountability. By the time a spill is attributed to a vessel, critical evidence windows have closed.

### Our Solution

OceanSentinel closes this gap with an automated 3-stage pipeline that runs in **under 30 seconds**:

| Stage | Input | Output | Time |
|-------|-------|--------|------|
| 🔍 **Detection** | SAR GeoTIFF | Spill polygon + area + age estimate | ~5s |
| 🌊 **Drift Modeling** | Spill location + ocean data | Origin point + forecast trajectory | ~10s |
| 🚢 **Vessel Attribution** | Origin + AIS records | Ranked suspect list with scores | ~5s |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Deep Learning Detection
- **U-Net with ResNet-50** encoder for SAR oil spill segmentation
- 3-class output: background ocean · oil spill · lookalike (biogenic slick)
- Tile-based inference with overlap merging for arbitrarily large scenes
- Automatic **spill age estimation** using Fay's spreading equations

</td>
<td width="50%">

### 🌊 Lagrangian Drift Modeling
- **OpenDrift OpenOil** integration for physics-based particle simulation
- Fallback synthetic Lagrangian engine (zero dependencies)
- **Hindcast**: trace spill backward to probable release origin
- **Forecast**: predict spill spread 48–72 hours ahead

</td>
</tr>
<tr>
<td width="50%">

### 🚢 Vessel Attribution Engine
- Spatial + temporal AIS filtering around estimated origin
- Multi-criteria vessel scoring: proximity · trajectory alignment · AIS darkness · vessel type · speed anomaly
- **DBSCAN clustering** for origin point estimation from particle endpoints

</td>
<td width="50%">

### 🗺️ Real-Time Dashboard
- **React + Leaflet** interactive map with layer toggles
- Real-time pipeline progress via **WebSocket** streaming
- Vessel ranking panel with score breakdowns
- One-click demo with pre-loaded incident scenarios

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
                    ┌──────────────────────────┐
                    │   Sentinel-1 SAR Scene   │
                    │      (GeoTIFF)           │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   Stage 1: DETECTION     │
                    │   ├─ SAR Preprocessing   │
                    │   │  (σ⁰ calibration,    │
                    │   │   speckle filter)     │
                    │   ├─ U-Net Segmentation   │
                    │   │  (ResNet-50 encoder)  │
                    │   ├─ Polygon Extraction   │
                    │   └─ Age Estimation       │
                    │      (Fay's equations)    │
                    └───────────┬──────────────┘
                                │ spill polygon, centroid, area
                                │
          ┌─────────────────────▼────────────────────┐
          │          Stage 2: DRIFT MODELING          │
          │   ├─ CMEMS Ocean Currents + ERA5 Wind    │
          │   ├─ Lagrangian Particle Simulation      │
          │   │  (OpenDrift or synthetic fallback)    │
          │   ├─ Hindcast → Release Origin           │
          │   │  (DBSCAN clustering)                 │
          │   └─ Forecast → 72h Spread Prediction    │
          └─────────────────────┬────────────────────┘
                                │ origin (lon, lat, time)
                                │
          ┌─────────────────────▼────────────────────┐
          │       Stage 3: VESSEL ATTRIBUTION        │
          │   ├─ AIS Spatial + Temporal Filter       │
          │   ├─ Anomaly Detection                   │
          │   │  (darkness, speed deviations)         │
          │   ├─ Multi-Criteria Scoring              │
          │   │  ┌──────────────────────────┐        │
          │   │  │ Proximity ........... 35%│        │
          │   │  │ Trajectory align .... 25%│        │
          │   │  │ AIS darkness ........ 20%│        │
          │   │  │ Vessel type ......... 10%│        │
          │   │  │ Speed anomaly ....... 10%│        │
          │   │  └──────────────────────────┘        │
          │   └─ Ranked Suspect Vessel List          │
          └─────────────────────┬────────────────────┘
                                │
          ┌─────────────────────▼────────────────────┐
          │          INTERACTIVE DASHBOARD           │
          │   React 18 · Leaflet · WebSocket · TW    │
          │   ├─ Live pipeline progress bar           │
          │   ├─ Map: spill polygon + drift tracks   │
          │   ├─ Vessel ranking with score breakdown  │
          │   └─ Layer toggles + incident selector   │
          └──────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend + ML pipeline |
| Node.js | 20+ | Frontend build tooling |
| Docker | 24+ | *(Optional)* Containerized deployment |

### Option 1: Local Development

```bash
# 1. Clone the repository
git clone https://github.com/praveenhargari09/sih2026-ps26143-ocean-sentinel.git
cd sih2026-ps26143-ocean-sentinel

# 2. Set up environment
cp .env.example .env
# Edit .env with your CMEMS/ERA5 credentials (optional — demo works without them)

# 3. Install Python dependencies
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 4. Start the backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — the dashboard is ready.

### Option 2: Docker Compose

```bash
docker-compose up --build
# Backend:  http://localhost:8000
# Frontend: http://localhost:5173
```

---

## 🎮 Demo

**No satellite data required** — the system ships with synthetic scenarios for instant demo.

### From the Dashboard
Click **"Demo"** in the top bar to run the full pipeline on the Arabian Sea synthetic scenario.

### From the API
```bash
curl http://localhost:8000/api/demo/generate
```

### Pre-loaded Incidents
Select real incident reconstructions from the **Mission Selector** dropdown:

| Incident | Year | Location | Data |
|----------|------|----------|------|
| Ennore Oil Spill | 2017 | Chennai, India | SAR + AIS + Ocean |
| MV X-Press Pearl | 2021 | Sri Lanka | SAR + AIS + Ocean |
| Arabian Sea Demo | — | Arabian Sea | Fully synthetic |

---

## 📁 Project Structure

```
sih2026-ps26143-ocean-sentinel/
│
├── backend/                          # Python backend
│   ├── api/                          # FastAPI application
│   │   ├── main.py                   # App entry point, CORS, router setup
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── pipeline.py               # 3-stage orchestration engine
│   │   ├── pipeline_bus.py           # WebSocket event broadcaster
│   │   ├── data_loader.py            # GeoJSON/CSV data loading
│   │   ├── incident_resolver.py      # Filesystem resolver for incidents
│   │   ├── db.py                     # SQLAlchemy async database setup
│   │   └── routes/                   # API route modules
│   │       ├── spills.py             #   /api/spills — spill CRUD
│   │       ├── vessels.py            #   /api/vessels — vessel ranking
│   │       ├── ais.py                #   /api/ais — AIS track queries
│   │       ├── ocean.py              #   /api/ocean — oceanographic data
│   │       ├── demo.py               #   /api/demo — synthetic demo runner
│   │       ├── incidents.py          #   /api/incidents — incident loader
│   │       └── pipeline_ws.py        #   /ws/pipeline — WebSocket progress
│   │
│   ├── detection/                    # SAR oil spill detection (ML)
│   │   ├── model.py                  # U-Net architecture + loss functions
│   │   ├── inference.py              # Production inference engine
│   │   ├── train.py                  # Training loop with metrics logging
│   │   ├── dataset.py                # PyTorch Dataset for SAR tiles
│   │   ├── preprocess.py             # SAR preprocessing pipeline
│   │   ├── evaluate.py               # Evaluation metrics (IoU, Dice)
│   │   └── age_estimator.py          # Fay's spreading equation solver
│   │
│   ├── drift/                        # Ocean drift simulation
│   │   ├── drift_engine.py           # Lagrangian particle drift engine
│   │   ├── hindcast.py               # Backward trajectory simulation
│   │   ├── forecast.py               # Forward trajectory prediction
│   │   ├── origin_estimator.py       # DBSCAN-based origin clustering
│   │   └── data_fetcher.py           # CMEMS/ERA5 data downloader
│   │
│   ├── db/                           # Database layer
│   │   └── models.py                 # SQLAlchemy ORM models
│   │
│   └── synthetic_data/               # Synthetic data generators
│       ├── dataset.py                # Synthetic SAR scene generator
│       ├── ais_tracks.csv            # Pre-generated AIS tracks
│       └── ocean_wind_currents.csv   # Pre-generated forcing data
│
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── App.tsx                   # Main application component
│   │   ├── main.tsx                  # React entry point
│   │   ├── types.ts                  # TypeScript interfaces
│   │   ├── index.css                 # Tailwind CSS + custom styles
│   │   ├── api/                      # Axios API client
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useSpillData.ts       #   Spill data management
│   │   │   └── useWebSocket.ts       #   WebSocket connection
│   │   ├── context/                  # React context providers
│   │   │   └── LayerContext.tsx       #   Map layer visibility
│   │   └── components/               # UI components
│   │       ├── MapView.tsx           #   Leaflet map container
│   │       ├── SpillLayer.tsx        #   Spill polygon overlay
│   │       ├── DriftLayer.tsx        #   Drift trajectory overlay
│   │       ├── VesselLayer.tsx       #   Vessel track overlay
│   │       ├── VesselRankingPanel.tsx #   Suspect vessel list
│   │       ├── SpillInfoPanel.tsx    #   Spill detail panel
│   │       ├── PipelineStatusBar.tsx #   Progress indicator
│   │       ├── UploadPanel.tsx       #   SAR upload modal
│   │       ├── MissionSelector.tsx   #   Incident dropdown
│   │       └── DataSourceBadge.tsx   #   Data provenance badge
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── nginx.conf                    # Production Nginx config
│
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Raw SAR/AIS/ocean data
│   ├── processed/                    # Pipeline outputs
│   ├── uploads/                      # User-uploaded SAR scenes
│   ├── synthetic/                    # Synthetic data generators
│   │   ├── scenario_config.yaml      #   Demo scenario definition
│   │   ├── generate_sar_scene.py     #   SAR scene synthesizer
│   │   └── generate_ais_tracks.py    #   AIS track synthesizer
│   └── incidents/                    # Pre-loaded incident packs
│       ├── ennore_2017/              #   Ennore oil spill (2017)
│       ├── xpress_pearl_2021/        #   MV X-Press Pearl (2021)
│       └── arabian_sea_demo/         #   Synthetic demo scenario
│
├── models/
│   └── checkpoints/                  # Trained model weights (gitignored)
│
├── scripts/                          # Utility scripts
│   ├── download_zenodo.py            # Download training data from Zenodo
│   ├── pilot_setup.py               # One-command environment setup
│   └── zip_project.py               # Package project for submission
│
├── docker-compose.yml                # Multi-container deployment
├── Dockerfile.backend                # Backend container
├── Dockerfile.frontend               # Frontend container (Nginx)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔌 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/demo/generate` | Run full synthetic demo pipeline |
| `GET` | `/api/incidents` | List available incident scenarios |
| `POST` | `/api/incidents/{id}/load` | Load and run a specific incident |
| `POST` | `/api/spills/detect` | Upload SAR scene → detect oil spill |
| `POST` | `/api/spills/{id}/drift` | Run hindcast + forecast drift model |
| `GET` | `/api/spills` | List all analyzed spills |
| `GET` | `/api/vessels/{spill_id}` | Get ranked suspect vessel list |
| `GET` | `/api/ais/track/{mmsi}` | Get AIS track for a vessel |
| `GET` | `/api/ocean/grid` | Get ocean current grid data |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/ws/pipeline/{spill_id}` | Real-time pipeline progress events |

**Event payload:**
```json
{
  "stage": "detection | drift | vessels",
  "percent": 45,
  "message": "Running hindcast simulation...",
  "spill_id": "abc123"
}
```

---

## 🧪 Model Details

### U-Net Segmentation

| Parameter | Value |
|-----------|-------|
| Architecture | U-Net with skip connections |
| Encoder | ResNet-50 (ImageNet pretrained) |
| Input | 2-channel SAR (VV + VH polarization) |
| Output | 3-class segmentation map |
| Tile Size | 512 × 512 px |
| Loss Function | Combined CrossEntropy + Dice (0.5 + 0.5) |
| Classes | `0` Background · `1` Oil Spill · `2` Lookalike |

### Spill Age Estimation

Uses **Fay's gravity-viscous spreading equation** to estimate time since release:

$$A(t) = k_2 \cdot V^{5/6} \cdot t^{3/4}$$

Inverted to solve for time given observed area.

### Drift Physics

**Lagrangian particle advection** with Stokes drift:

$$\frac{dx}{dt} = u_{current} + \alpha \cdot u_{wind} + \sigma \cdot \mathcal{N}(0,1)$$

where $\alpha \approx 0.03$ (Stokes factor) and $\sigma$ represents turbulent diffusion.

---

## 🗃️ Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| Sentinel-1 SAR Oil Spill Dataset | [Zenodo](https://zenodo.org) | U-Net training & validation |
| CMEMS Ocean Currents | [Copernicus Marine](https://marine.copernicus.eu) | Drift model forcing |
| ERA5 Wind Fields | [CDS](https://cds.climate.copernicus.eu) | Wind-driven drift component |
| AIS Ship Traffic | [MarineCadastre](https://marinecadastre.gov) | Vessel attribution |

> **Note**: The demo mode works entirely with synthetic data — no external datasets required.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML / Detection** | PyTorch · segmentation-models-pytorch · albumentations · rasterio |
| **Drift Modeling** | OpenDrift (OpenOil) · NumPy · SciPy · netCDF4 · xarray |
| **AIS Analysis** | pandas · scikit-learn (DBSCAN) · GeoPandas · Shapely |
| **Backend API** | FastAPI · Uvicorn · SQLAlchemy · WebSockets · Pydantic |
| **Frontend** | React 18 · TypeScript · Vite · react-leaflet · Tailwind CSS · Recharts |
| **Infrastructure** | Docker · Docker Compose · Nginx |

---

## 👥 Team

Built for **Smart India Hackathon 2026** — Problem Statement #26143 (NTRO)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🛢️ by Team OceanSentinel**

*Protecting our oceans, one pixel at a time.*

</div>
