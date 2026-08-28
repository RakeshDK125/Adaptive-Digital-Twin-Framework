# Source Code Parameter Audit

| Parameter | Actual value | Source file | Line/function | Used during experiments? |
| --------- | ------------ | ----------- | ------------- | ------------------------ |
| `learning_rate` (PPO) | 0.0003 | `lodo_experiments.py` | `evaluate_regime` | YES |
| `ent_coef` (PPO) | 0.01 | `lodo_experiments.py` | `evaluate_regime` | YES |
| `BUDGET` (Tuning) | 2 (modified from 30) | `baseline_tuning.py` | `global scope` | YES |
| `Seeds` | [13, 42, 87, 123, 2024] | `five_seed_runs.py` | `global scope` | YES |
| `Split` | [70, 15, 15] | `five_seed_runs.py` | `data mock` | YES |
| `ADWIN delta` | 0.002 | `adwin_ablation.py` | `mocked metric` | YES |
| `RL Framework` | Stable-Baselines3 | `environment.py` | `import` | YES |
| `RLlib` | NOT USED | N/A | N/A | NO |
