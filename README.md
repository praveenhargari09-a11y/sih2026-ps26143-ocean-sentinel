<div align="center">

# Ocean Sentinel

### AI-Powered Maritime Oil Spill Detection & Vessel Attribution System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SIH 2026 — Problem Statement #26143 (NTRO)**
**Team Ocean Sentinel (Team ID: 181109)**

*Detect oil spills from SAR satellite imagery → Trace spill origin via ocean drift modeling → Rank suspect vessels using AIS correlation*

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Data Provenance](#-data-provenance) · [Results](#-results)

</div>

---

## Overview

Ocean Sentinel is a full-stack maritime intelligence platform designed to automate the detection, tracking, and attribution of marine oil spills. The system operates via a **3-stage, 6-step pipeline**:

1. **Detection:** SAR GeoTIFF → Spill polygon (Semantic Segmentation)
2. **Drift Modeling:** Ocean current/wind forcing → Hindcast (48h) / Forecast (72h)
3. **Vessel Attribution:** Spatial filtering → AIS anomaly correlation → Ranked suspect list

> **Note on Demo Mode:** The live demo endpoints utilize synthetic/reconstructed placeholder data and a mock detector bypass to ensure the pipeline runs reliably without heavy GPU requirements or live API dependencies. The full PyTorch U-Net models are available in the codebase (`backend/detection/`).

---

## ⚠️ Limitations & Disclaimer
* **Relative Scoring:** The vessel attribution scores are heuristic guidelines designed to rank suspects for human investigators. They do not represent a mathematical probability of guilt.
* **Collision Incidents:** In cases of ship-to-ship collisions (e.g., Ennore 2017), multiple vessels may overlap in time and space, complicating algorithmic blame.
* **Age Estimation:** The Fay spreading equations yield approximate release times based on assumed oil volumes.
* **Demo Data:** The included incidents rely on reconstructed or synthetic AIS/SAR placeholders, not live raw data feeds.

---

## Features

<table>
<tr>
<td width="50%">

### Deep Learning Detection
- **U-Net with ResNet-50** encoder for SAR oil spill segmentation (2-channel VV+VH).
- 3-class output: background ocean, oil spill, lookalike.
- Automatic **spill age estimation** using Fay's spreading equations.

</td>
<td width="50%">

### Lagrangian Drift Modeling
- **OpenDrift** integration for physics-based particle simulation.
- Fallback synthetic Lagrangian engine (no OpenDrift dependency).
- **Hindcast**: trace spill backward to probable release origin.
- **Forecast**: predict spill spread 72 hours ahead.

</td>
</tr>
<tr>
<td width="50%">

### Vessel Attribution Engine
- Spatial + temporal AIS filtering around the estimated origin (DBSCAN clustered).
- Multi-criteria vessel scoring based on 5 heuristic factors.

</td>
<td width="50%">

### Real-Time Dashboard
- **React + Leaflet** interactive map with layer toggles.
- Real-time pipeline progress via **WebSocket** streaming.
- Vessel ranking panel with score breakdowns.

</td>
</tr>
</table>

---

## Architecture

```text
                    ┌──────────────────────────┐
                    │   Sentinel-1 SAR Scene   │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   Stage 1: DETECTION     │
                    │   ├─ 1. SAR Preprocess   │
                    │   └─ 2. U-Net Segment    │
                    └───────────┬──────────────┘
                                │ spill polygon, age estimate
                                │
          ┌─────────────────────▼────────────────────┐
          │          Stage 2: DRIFT MODELING         │
          │   ├─ 3. Hindcast Simulation (48h)        │
          │   └─ 4. Forecast Simulation (72h)        │
          └─────────────────────┬────────────────────┘
                                │ origin (lon, lat, time)
                                │
          ┌─────────────────────▼────────────────────┐
          │       Stage 3: VESSEL ATTRIBUTION        │
          │   ├─ 5. AIS Correlation                  │
          │   └─ 6. Heuristic Scoring                │
          └─────────────────────┬────────────────────┘
                                │
          ┌─────────────────────▼────────────────────┐
          │          INTERACTIVE DASHBOARD           │
          └──────────────────────────────────────────┘
```

---

## Screenshots

*(TODO: Add UI screenshots to `docs/screenshots/`)*

* `![Dashboard View](docs/screenshots/dashboard.png)`
* `![Vessel Rankings](docs/screenshots/rankings.png)`

---

## Quick Start

### 1. Local Development

```bash
# 1. Clone the repository
git clone https://github.com/praveenhargari09-a11y/sih2026-ps26143-ocean-sentinel.git
cd sih2026-ps26143-ocean-sentinel

# 2. Set up environment
cp .env.example .env
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 3. Start the backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** to view the dashboard.

### 2. Docker Compose

```bash
docker-compose up --build
```
- **Backend:** http://localhost:8000
- **Frontend:** http://localhost:5173 (Maps to internal port 80 via Nginx)

---

## Data Provenance

The system includes three pre-loaded demonstration scenarios. Because historical raw SAR/AIS data is often proprietary or inaccessible, we rely on reconstructions:

| Incident | SAR Source | Ocean Forcing | AIS Source | Type |
|----------|------------|---------------|------------|------|
| **Ennore 2017** | Synthetic Placeholder | Reconstructed | MarineTraffic (Reconstructed) | `real_with_placeholders` |
| **X-Press Pearl 2021** | Synthetic Placeholder | Reconstructed | MarineTraffic (Reconstructed) | `reconstructed` |
| **Arabian Sea Demo** | Synthetic | Synthetic | Synthetic | `synthetic` |

---

## Model Details

### Detection Engine
- **Architecture:** U-Net (ResNet-50) via `segmentation-models-pytorch`.
- **Loss Function:** Combined CrossEntropy + Dice.

### Spill Age Estimation
Uses Fay's gravity-viscous spreading equation to estimate time since release, inverted as:

$$t = \left( \frac{A}{k_2 \cdot V^{5/6}} \right)^{4/3}$$

### Drift Physics
Lagrangian particle advection incorporating wind drift:

$$\frac{dx}{dt} = u_{current} + \alpha \cdot u_{wind} + \sigma \cdot \mathcal{N}(0,1)$$

*(Where $\alpha \approx 0.03$ is implemented as a wind drift factor, internally referred to as `STOKES_FACTOR` in code).*

### Vessel Scoring Heuristics
Vessel suspicion is ranked using a 5-factor hand-tuned heuristic formula (not machine-learned):
1. **Proximity:** (35%) Distance from origin point in the origin time window.
2. **Trajectory Alignment:** (25%) Vessel course vs. slick elongation axis.
3. **AIS Darkness:** (20%) Transponder gaps near origin time/location.
4. **Vessel Type:** (10%) Tankers scored higher risk.
5. **Speed Anomaly:** (10%) Unexplained slow-down or drift in the origin zone.

---

## Results

**Detection model:** Prototype stage. 
**Validation metrics:** TODO. 

*Currently, the live system bypasses GPU inference to ensure smooth demo execution using pre-generated incident topologies.*

---

## Team Ocean Sentinel

**SIH 2026 | Team ID: 181109**

* [Member 1 Name] - [Role]
* [Member 2 Name] - [Role]
* [Member 3 Name] - [Role]
* [Member 4 Name] - [Role]
* [Member 5 Name] - [Role]
* [Member 6 Name] - [Role]

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
