# Actual Hyperparameters from Implementation

## Meta-PPO
learning_rate_inner = MISSING (Not implemented)
meta_learning_rate = MISSING (Not implemented)
gamma = MISSING (Not implemented)
clip_epsilon = MISSING (Not implemented)
gae_lambda = MISSING (Not implemented)
n_steps = MISSING (Not implemented)
epochs = MISSING (Not implemented)
batch_size = MISSING (Not implemented)
entropy_coef = MISSING (Not implemented)
value_coef = MISSING (Not implemented)
max_grad_norm = MISSING (Not implemented)
K = MISSING (Not implemented)
N_ft = MISSING (Not implemented)
B = MISSING (Not implemented)
N_iter = MISSING (Not implemented)

## SAC
learning_rate = 0.001
batch_size = 256 (SB3 Default)
gamma = 0.99 (SB3 Default)
tau = 0.005 (SB3 Default)
buffer_size = 10000
entropy_coef = "auto" (SB3 Default)

## A2C
learning_rate = 0.001
n_steps = 5 (SB3 Default)
gamma = 0.99 (SB3 Default)
gae_lambda = 1.0 (SB3 Default)
entropy_coef = 0.0 (SB3 Default)
value_coef = 0.5 (SB3 Default)

## ADWIN
delta = MISSING (Not implemented)
min_subwindow = MISSING (Not implemented)

## Knowledge Graph
max_depth = MISSING (Not implemented)
beta = MISSING (Not implemented)
anomaly_threshold = MISSING (Not implemented)

## Coordination
M = MISSING (Not implemented)
q = MISSING (Not implemented)
dispatch_policy = MISSING (Not implemented)

## Environment
seeds = [13, 42, 87, 123, 2024]
split = [70, 15, 15]
CPU = ACTUAL 
RAM = ACTUAL
Python = ACTUAL
RL framework = Stable-Baselines3
Stable-Baselines3 = v2.2.1
Ray/RLlib = NOT USED
Neo4j = NOT USED
