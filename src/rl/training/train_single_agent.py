# Quick test from project root

from src.rl.environments.single_agent_env import DTNDroneEnvironment
from src.config.simulation_config import SimulationConfig

print('🧪 Testing environment...')
config = SimulationConfig()
env = DTNDroneEnvironment(config)

obs, info = env.reset()
print(f'✅ Reset successful - Obs shape: {obs.shape}')
print(f'✅ Action space: {env.action_space}')

action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
print(f'✅ Step successful - Reward: {reward:.3f}')
print('🎉 Environment test passed!')
