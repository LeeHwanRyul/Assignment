import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

env = gym.make(
   "MiniGrid-Empty-5x5-v0",
   render_mode="human")
observation, info = env.reset(seed=42)
print(env.action_space)
print(observation["image"])
print(FlatObsWrapper(env).observation_space)

input()

while True:
   pass
"""
for _ in range(1000):
   # action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
"""