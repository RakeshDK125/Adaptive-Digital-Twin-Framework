<div align="center">
  <h1>AIDA-Twin: Adaptive Intelligent Digital Twin Framework</h1>
  
  <p><b>A self-healing, highly autonomous Digital Twin architecture bridging the gap between passive monitoring and active control via Reinforcement Learning and Agentic AI.</b></p>

  <!-- Badges -->
  <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.12-blue.svg?style=for-the-badge" alt="Python">
    <img src="https://img.shields.io/badge/React-18.x-61DAFB.svg?style=for-the-badge&logo=react" alt="React">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker" alt="Docker">
  </p>
</div>

<br />

## 📖 Overview

Traditional Digital Twins are purely observational—they monitor state but require human intervention to resolve faults. **AIDA-Twin** introduces a paradigm shift by functioning as an **Active Decision Intelligence Engine**. 

It achieves true autonomy by tightly orchestrating three advanced AI paradigms:
1. **Model-Agnostic Meta-Learning (MAML) over PPO**: For continuous, rapid-adaptation control policies.
2. **Multi-Agent Systems (MAS)**: For discrete, logical reasoning and localized fault handling.
3. **Semantic Knowledge Graphs (Neo4j)**: Providing the topology and spatial-temporal context required for the swarm to coordinate.

When high-velocity telemetry streams exhibit concept drift (detected via **ADWIN**), the system automatically triggers a self-healing pipeline, adapting its control policies to unseen operational regimes without human intervention.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Quickstart & Installation](#-quickstart--installation)
- [Reproducing Benchmarks](#-reproducing-benchmarks)
- [Documentation](#-documentation)
- [Citation](#-citation)

---

## 🌟 Key Features

*   🎯 **Active Decision Intelligence**: Transforms passive digital twins into proactive, self-correcting autonomous agents capable of taking direct physical action via simulation.
*   📉 **Concept Drift Detection (ADWIN)**: Actively monitors continuous high-velocity telemetry streams, instantly detecting distribution anomalies and concept drift using the ADWIN algorithm.
*   ⚡ **Rapid Adaptation via Meta-RL**: Utilizes MAML principles integrated with `Stable-Baselines3` (PPO/SAC/A2C) to adapt to novel hardware faults and unseen operating environments with minimal adaptation steps.
*   🤖 **Swarm Architecture**: Employs an intelligent multi-agent coordination layer that reasons across time and network topology using semantic graph insights.
*   📊 **Real-time XAI Dashboard**: A dynamic React/TypeScript frontend offering complete visibility into the Twin's health metrics, drift alerts, active control policies, and SHAP-based Explainable AI (XAI) insights.

---

## 🏗️ System Architecture

AIDA-Twin operates on a robust microservice architecture:

### 1. The Reasoning Engine (`backend/`)
The Python/FastAPI backend acts as the brain of the Digital Twin. It orchestrates the flow of telemetry data into the **Virtual Representation Engine**. 
- **RL Backend**: Powered by `Stable-Baselines3`.
- **Graph Intelligence**: `Neo4j` handles complex topological queries to route agents to anomalous nodes.
- **Agentic Coordination**: A custom multi-agent logic controller handling dispatch, failure queuing, and cross-agent communication.

### 2. The Visualization Interface (`frontend/`)
A responsive web application built for operational visibility.
- **Framework**: React 18 + TypeScript.
- **Styling**: TailwindCSS for sleek, dynamic UI components.
- **Data Visualization**: Recharts for live telemetry plotting, fault visualizations, and RL convergence tracking.

---

## 📂 Repository Structure

```text
📦 Adaptive-Digital-Twin-Framework
 ┣ 📂 backend/
 ┃ ┣ 📂 app/              # FastAPI application & core domain logic
 ┃ ┣ 📂 data/             # AI4I, Gas Turbine, and Hydraulic datasets
 ┃ ┣ 📂 outputs/          # Generated results, CSVs, JSONs, and Figures
 ┃ ┣ 📂 scripts/          # Benchmark and evaluation harnesses (Five-seed, LODO)
 ┃ ┗ 📜 main.py           # Backend entry point
 ┣ 📂 frontend/           # React + TypeScript Dashboard
 ┣ 📂 docs/               # In-depth architectural and API documentation
 ┣ 📜 docker-compose.yml  # Container orchestration
 ┗ 📜 README.md           # This file
```

---

## 🚀 Quickstart & Installation

### Prerequisites
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Compose (v2+)
*   Git

### Deployment via Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RakeshDK125/Adaptive-Digital-Twin-Framework.git
   cd Adaptive-Digital-Twin-Framework
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env if you need to override the default Neo4j credentials or API configurations
   ```

3. **Build and Spin Up the Infrastructure**:
   ```bash
   docker compose up -d --build
   ```

4. **Access the System**:
   - **Frontend UI Dashboard**: [http://localhost](http://localhost)
   - **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Neo4j Browser**: [http://localhost:7474](http://localhost:7474)

---

## 🔬 Reproducing Benchmarks

For researchers looking to reproduce the findings or run the evaluation harnesses (Five-Seed Ablations, Leave-One-Domain-Out cross-validation, and Equal-Budget Baseline Tuning), the pipeline has been strictly upgraded to use **organic, real sklearn metrics** directly computed on real datasets. No data is mocked.

```bash
# Navigate to the scripts directory
cd backend/scripts

# 1. Ensure you have downloaded the real datasets
python download_datasets.py

# 2. Run the master evaluation suite (Runs 135 models over 5 seeds + LODO + Output generation)
python run_all.py
```
*Note: Resulting CSVs, `REPORT.md`, and publication-ready graphs are automatically zipped into `AIDA_Twin_Honest_Results.zip` in your Downloads folder.*

---

## 📚 Documentation

Deep dives into the core architectural components and design decisions can be found in the `/docs` directory:
- 📖 [**Architecture & Theoretical Foundation**](docs/ARCHITECTURE.md)
- ⚙️ [**Installation & Docker Deployment**](docs/INSTALLATION.md)
- 🔌 [**API Reference (REST & WebSockets)**](docs/API.md)
- 🔬 [**Experimental Reproducibility & Benchmarks**](docs/EXPERIMENTS.md)

---

## 🤝 Contributing

We welcome community contributions! Whether it is expanding the supported Reinforcement Learning algorithms (e.g., adding native Ray/RLlib support), improving the multi-agent orchestration efficiency, or fixing bugs:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you utilize the AIDA-Twin framework or our benchmark suites in your research, please cite our upcoming paper:

```bibtex
@article{aidatwin2026,
  title={Adaptive Intelligent Digital Twin Framework utilizing Meta-RL and Agentic AI},
  author={Rakesh et al.},
  journal={[Insert Target Journal Name]},
  year={2026}
}
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
