import gymnasium as gym
from Minigrid.minigrid.minigrid_env import MiniGridEnv

env: MiniGridEnv = gym.make(
   "MiniGrid-Empty-8x8-v0",
   render_mode="human")
observation, info = env.reset(seed=42)
print(observation)
for _ in range(1000):
   # action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()