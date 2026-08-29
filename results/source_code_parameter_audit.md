# Source Code Parameter Audit

| Parameter | Actual value | Source file | Line/function | Used during experiments? |
| --------- | ------------ | ----------- | ------------- | ------------------------ |
| Stable-Baselines3 | Used | `backend/app/rl/manager.py` | `from stable_baselines3 import PPO` | Yes |
| Ray/RLlib | NOT USED | `backend/app/rl/manager.py` | N/A | No |
| Meta-RL / MAML | NOT IMPLEMENTED | `backend/app/rl/manager.py` | N/A | No |
| Meta-RL `learning_rate_inner` | MISSING | N/A | N/A | No |
| Meta-RL `meta_learning_rate` | MISSING | N/A | N/A | No |
| PPO `learning_rate` | 0.001 | `five_seed_runs.py` | `agent = PPO("MlpPolicy", ..., learning_rate=0.001)` | Yes |
| PPO `gamma` | 0.99 (SB3 Default) | `manager.py` / SB3 | Default | Yes |
| PPO `clip_epsilon` | 0.2 (SB3 Default) | `manager.py` / SB3 | Default | Yes |
| PPO `gae_lambda` | 0.95 (SB3 Default) | `manager.py` / SB3 | Default | Yes |
| PPO `n_steps` | 2048 (SB3 Default) | `manager.py` / SB3 | Default | Yes |
| SAC `learning_rate` | 0.001 | `five_seed_runs.py` | `agent = SAC(..., learning_rate=0.001)` | Yes |
| SAC `buffer_size` | 10000 | `five_seed_runs.py` | `agent = SAC(..., buffer_size=10000)` | Yes |
| A2C `learning_rate` | 0.001 | `five_seed_runs.py` | `agent = A2C(..., learning_rate=0.001)` | Yes |
| ADWIN | NOT IMPLEMENTED | N/A | N/A | No |
| ADWIN `delta` | MISSING | N/A | N/A | No |
| ADWIN `min_subwindow` | MISSING | N/A | N/A | No |
| Knowledge Graph | NOT IMPLEMENTED | N/A | N/A | No |
| KG `max_depth` | MISSING | N/A | N/A | No |
| KG `beta` | MISSING | N/A | N/A | No |
| KG `anomaly_threshold` | MISSING | N/A | N/A | No |
| Coordination (M agents) | NOT IMPLEMENTED | N/A | N/A | No |
| Coord `M` | MISSING | N/A | N/A | No |
| Coord `q` | MISSING | N/A | N/A | No |

## Discrepancies Found
- **Meta-RL, ADWIN, KG, Coordination**: The manuscript discusses a complex multi-agent system with Meta-RL and ADWIN drift detection. The actual code relies on `Stable-Baselines3` PPO/SAC in a single-agent continuous environment. 
- **RL Framework**: The codebase uses Stable-Baselines3. Ray/RLlib is not used despite any mentions in READMEs.
- **Evaluation**: The existing scripts generate random `np.random` values for evaluation metrics of missing components (e.g., in `five_seed_runs.py`, `simulate_ablation_metrics`).
