# Experimental Reproducibility & Benchmarks

To validate the theoretical contributions of this framework for journal submission, we provide an automated benchmarking suite.

## The Industrial Event Simulator
Located at `backend/scripts/simulator.py`, this module deterministically simulates thousands of hours of machine run-time. It injects progressive physical degradation and sudden catastrophic concept drifts, acting as the testbed for our algorithms.

## Running the Benchmark
To generate the core performance metrics (comparing Baseline PPO against our Meta-RL Engine):
```bash
cd backend
python scripts/benchmark.py
```
**Output**: Generates `benchmark_results.csv` detailing MSE, Convergence Steps, and Drift Detection Latency.

## Running the Ablation Study
To mathematically prove the contribution of the KG and the Agent Swarm, we run a structural ablation study:
```bash
cd backend
python scripts/ablation.py
```
**Output**: Generates `ablation_results.csv` detailing the Fault Mitigation Accuracy and Decision Latency as modules are systematically disabled.

## Data Visualization
The generated `.csv` files are pre-formatted for direct ingestion into `matplotlib` or `plotly` to generate the exact figures required for LaTeX IEEE/Elsevier manuscripts.
