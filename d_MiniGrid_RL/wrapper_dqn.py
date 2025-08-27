from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Wall
from typing import Optional
from minigrid.manual_control import ManualControl
from collections import deque

import numpy as np

map = [[0, 0, 1, 0, 0, 0],
       [1, 0, 1, 0, 1, 0],
       [0, 0, 1, 0, 1, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 1, 1, 0, 1, 0],
       [0, 0, 0, 0, 1, 0]] 

class Wrapper(MiniGridEnv):
    def __init__(self,
                 args = None,
                 size=8,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0,
                 max_steps: Optional[int] = None,
                 stack_size=4, 
                 **kwargs
                 ):
        self.args = args
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if args.render_mode == "human":
            super().__init__(
                    mission_space=mission_space,
                    grid_size=size,
                    max_steps=100,
                    **kwargs,
            )
        else:
            super().__init__(
                    mission_space=mission_space,
                    grid_size=size,
                    max_steps=255,
                    **kwargs,
            )

    @staticmethod
    def _gen_mission():
        return "grand mission"
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        first_frame = obs['image'][...,0]   # type 채널만 (7,7)

        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)

        stacked_obs = np.array(self.frames)
        return stacked_obs, info

    def step(self, action_controlled):
        if self.args.is_evaluate:
            self.render()

        self.p_pos = self.agent_pos

        obs, reward, terminated, truncated, info = super().step(action_controlled)

        front_pos = obs['image'][3, 5]

        # reward *= 100

        distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.agent_start_pos))

        if self.agent_pos != self.p_pos:
            reward += distance * 0.001

        frame = obs['image'][...,0]  # type 채널만
        self.frames.append(frame)
        stacked_obs = np.array(self.frames)

        return stacked_obs, reward, terminated, truncated, info
    
    def _observation_wrapping(self, obs):
        x = obs['image']
        x = x.astype(np.float32)
        x[x == 2] = 0
        x[2:6, :] *= 1.5
        return x

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for j in range(len(map)):
            for i in range(len(map[0])):
                if map[j][i]:
                    self.grid.set(i+1, j+1, Wall())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


def main():
    env = Wrapper(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=555)
    manual_control.start()

if __name__ == "__main__":
    main()