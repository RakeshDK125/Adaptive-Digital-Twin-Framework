# AIDA-Twin: Adaptive Intelligent Digital Twin Framework

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?style=for-the-badge)
![React](https://img.shields.io/badge/React-18.x-61DAFB.svg?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker)

A publication-quality research prototype demonstrating a highly autonomous, self-healing **Digital Twin** architecture for IoT and Industrial systems. 

AIDA-Twin bridges the critical gap between passive monitoring and active control. It achieves this by fusing **Reinforcement Learning (Stable-Baselines3)** for continuous control optimization with a **Multi-Agent System (MAS)** for discrete, logical reasoning. These components are tightly unified and orchestrated by a semantic **Knowledge Graph (Neo4j)**.

---

## 🌟 Key Features

*   **Active Decision Intelligence**: Transforms traditional passive digital twins into proactive, self-correcting agents capable of taking direct action.
*   **Concept Drift Detection (ADWIN)**: Actively monitors continuous high-velocity telemetry streams, instantly detecting distribution anomalies and concept drift using the ADWIN algorithm.
*   **Rapid Adaptation via Meta-RL**: Utilizes Model-Agnostic Meta-Learning (MAML) principles over PPO to adapt to novel hardware faults and unseen operating environments with minimal adaptation steps.
*   **Swarm Architecture**: Employs an intelligent multi-agent coordination layer that reasons across time and network topology using topological insights from Neo4j.
*   **Real-time Dashboard**: A dynamic React/TypeScript frontend offering complete visibility into the Twin's health metrics, agent activities, drift alerts, and active policies.

---

## 🏗️ System Architecture & Stack

The AIDA-Twin repository is segmented into two primary components driven by modern microservices:

### 1. The Reasoning Engine (`backend/`)
A robust Python and FastAPI service that drives the core intelligence:
* **Reinforcement Learning Engine**: Powered by `Stable-Baselines3` (PPO, SAC, A2C).
* **Graph Intelligence**: Managed via `Neo4j` for complex topological queries.
* **Agentic Coordination Layer**: Custom multi-agent logic handling dispatch and cross-agent communication.
* **Data Processing**: Leverages `Pandas` and `NumPy` for high-throughput sensor telemetry formatting.

### 2. The Visualization Interface (`frontend/`)
A modern, responsive web application built for operational visibility:
* **Framework**: React 18 + TypeScript.
* **Styling**: TailwindCSS for sleek, dynamic UI components.
* **Data Visualization**: Recharts for live telemetry plotting and RL convergence tracking.

---

## 🚀 Quickstart & Reproducibility

### Prerequisites
*   Docker & Docker Compose (v2+)
*   Git

### Deployment via Docker

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

## 🔬 Running Benchmarks & Experiments

For researchers looking to reproduce the findings or run the evaluation harnesses (Five-Seed Ablations, Leave-One-Dataset-Out cross-validation, and Baseline Tuning):

```bash
# Navigate to the scripts directory
cd backend/scripts

# Run the primary robust evaluation suite
python five_seed_runs.py

# Run the Cross-Domain generalization evaluations (LODO)
python lodo_experiments.py
```
*Note: Resulting CSVs and raw metrics are automatically dumped into `backend/outputs/experiments_run/` for subsequent statistical analysis.*

---

## 📚 Documentation Navigation

Deep dives into the core architectural components and design decisions can be found in the `/docs` directory:
- 📖 [**Architecture & Theoretical Foundation**](docs/ARCHITECTURE.md)
- ⚙️ [**Installation & Docker Deployment**](docs/INSTALLATION.md)
- 🔌 [**API Reference (REST & WebSockets)**](docs/API.md)
- 🔬 [**Experimental Reproducibility & Benchmarks**](docs/EXPERIMENTS.md)

---

## 🤝 Contributing

We welcome community contributions, particularly for expanding the supported Reinforcement Learning algorithms or improving the multi-agent orchestration efficiency. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Cite Us

If you utilize the AIDA-Twin framework or our benchmark suites in your research, please cite our upcoming paper in Elsevier.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
