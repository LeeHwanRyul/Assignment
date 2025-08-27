import gymnasium as gym
import argparse
from DQN import DQN
from Q_Learning import Q_Learning
from SARSA import SARSA
from wrapper import Wrapper
# from wrapper_dqn import Wrapper
from config import args_q_learning, args_sarsa, args_dqn

import random
import numpy as np
import torch

class GlobalConfig:
    def __init__(self):
        self.seed = 555
        self.path2save_train_history = "train_history"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

config = GlobalConfig()
seed_everything(config.seed)

def make_env(args, render=None):
    env = Wrapper(args=args, render_mode=render)
    return env

# Object 기준 왼쪽부터
# 2, 5, 0: wall
# 0, 0, 0: 암흑시야
# 1, 0, 0: 자유 공간

def main(args, eval=False):
    if not eval:
        #agent = Q_Learning(make_env, args)
        agent = SARSA(make_env, args)
        #agent = DQN(make_env, args)
        agent.train()
    else:
        #agent = Q_Learning(make_env, args)
        agent = SARSA(make_env, args)
        #agent = DQN(make_env, args)
        #agent.load_model("q_table_q_learning.pkl")
        agent.load_model("q_table_sarsa.pkl")
        #agent.load_model("dqn_fin.pth")
        agent.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="MiniGrid-Custom")

    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    #args = args_q_learning.get_args(rest_args)
    args = args_sarsa.get_args(rest_args)
    # args = args_dqn.get_args(rest_args)

    args.path2save_train_history = config.path2save_train_history

    main(args, args.is_evaluate)