import argparse
from utils import boolean_argument

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='MiniGrid-Custom')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--validation_num_episodes', type=int, default=1)
    parser.add_argument('--batch_size', type=float, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.95)
    parser.add_argument('--replay_buffer_size', type=float, default=50000)
    parser.add_argument('--steps_between_train', type=float, default=1)
    parser.add_argument('--episode_reward_avg_solved', type=float, default=5)
    parser.add_argument('--train_num_episodes_before_next_test', type=float, default=100)
    parser.add_argument('--target_sync_step_interval', type=float, default=500)

    parser.add_argument('--is_evaluate', type=boolean_argument, default=True, help='for evaluation')
    parser.add_argument('--render_mode', type=str, default="None", help='rendering mode')

    return parser.parse_args(rest_args)