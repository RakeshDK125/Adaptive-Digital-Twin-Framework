import pytest
import os
import shutil
from stable_baselines3.common.env_checker import check_env
from app.domain.twin.models import MachineModel, SensorModel, VirtualRepresentation
from app.rl.environment import DigitalTwinEnv
from app.rl.manager import RLManager, MODEL_DIR

@pytest.fixture
def dummy_virtual_rep():
    machine = MachineModel(name="Test RL Machine")
    sensor = SensorModel(name="temp", unit="C", min_threshold=0.0, max_threshold=100.0)
    machine.add_sensor(sensor)
    return VirtualRepresentation(machine)

def test_gym_environment_compliance(dummy_virtual_rep):
    env = DigitalTwinEnv(dummy_virtual_rep)
    # This will throw an exception if the env does not comply with Gymnasium API
    check_env(env)

def test_rl_training_and_inference(dummy_virtual_rep):
    # Use PPO for a tiny 10-step train just to verify the pipeline doesn't crash
    model_name = "test_ppo"
    
    # Clean up before
    save_path = os.path.join(MODEL_DIR, f"{model_name}_PPO.zip")
    if os.path.exists(save_path):
        os.remove(save_path)
        
    generated_path = RLManager.train_model(
        virtual_rep=dummy_virtual_rep,
        algo_name="PPO",
        timesteps=10, 
        model_name=model_name
    )
    
    assert os.path.exists(generated_path)
    
    # Test Evaluation
    eval_metrics = RLManager.evaluate(dummy_virtual_rep, "PPO", generated_path, n_episodes=1)
    assert "mean_reward" in eval_metrics
    
    # Test Inference
    action = RLManager.get_action(dummy_virtual_rep, "PPO", generated_path)
    assert len(action) == 2  # As defined in action_space shape
    
    # Cleanup after test
    if os.path.exists(generated_path):
        os.remove(generated_path)
