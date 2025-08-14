import gymnasium as gym
import argparse
from DQN import DQN
from wrapper import Wrapper

from config import args_dqn

seed = 555

def make_env(env_name, render=None):
    env = gym.make(env_name, render_mode=render)
    env = Wrapper(env)
    env.seed(seed)
    return env

def main(args, eval=False):
    if not eval:
        agent = DQN(make_env, args)
        agent.train()
    else:
        agent = DQN(make_env, args)
        agent.load_model()
        agent.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="")

    args, rest_args = parser.parse_known_args()

    main(args, args.is_evaluate)