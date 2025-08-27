import argparse
from utils import boolean_argument

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='MiniGrid-Custom')

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--validation_num_episodes', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.3)

    parser.add_argument('--is_evaluate', type=boolean_argument, default=True, help='for evaluation')
    parser.add_argument('--render_mode', type=str, default="None", help='rendering mode')

    return parser.parse_args(rest_args)