import numpy as np
from minigrid.manual_control import ManualControl
import random
import pickle
import os

from gymnasium.wrappers import RecordVideo

class SARSA:
    def __init__(self, make_env, args):
        self.args = args
        env = make_env(args, args.render_mode)
        self.test_env = make_env(args, "rgb_array")

        self.start_epsilon = args.epsilon

        self.device = args.device
        print("device:", self.device)
        self.env = env
        self.num_episodes = args.num_episodes
        self.validation_num_episodes = args.validation_num_episodes
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.epsilon = args.epsilon

        self.q_table = {}

    def save_model(self, filename="q_table_sarsa.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[INFO] Q-table saved to {filename}")

    def load_model(self, filename="q_table_sarsa.pkl"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"[INFO] Q-table loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No saved Q-table found at {filepath}")

    def greedy_action(self, observation):
        if observation not in self.q_table:
            self.q_table[observation] = np.zeros(self.env.action_space.n)

        action_values = self.q_table[observation]
        action_values = action_values[:3]

        print(action_values)

        max_value = np.max(action_values)
        action = np.random.choice([i for i, v in enumerate(action_values) if v == max_value])

        return action

    def epsilon_greedy_action(self, observation):
        if observation not in self.q_table:
            self.q_table[observation] = np.zeros(self.env.action_space.n)

        action_values = self.q_table[observation]
        action_values = action_values[:3]

        if np.random.rand() < self.epsilon:
            action = random.choice(range(len(action_values)))
        else:
            max_value = np.max(action_values)
            action = np.random.choice([i for i, v in enumerate(action_values) if v == max_value])
        return action

    def encode_state(self, obs):
        forward_view = obs['image'][1:6, 4:, 0]
        return tuple(forward_view.flatten())

    def train(self):
        episode_reward_list = []
        episode_td_error_list = []

        training_time_steps = 0
        is_train_success = False

        for episode in range(self.num_episodes):
            episode_reward = 0.0
            episode_td_error = 0.0

            observation, _ = self.env.reset()
            observation = self.encode_state(observation)
            visited_states = [observation]

            episode_step = 0
            done = False

            if self.args.epsilon >= 0.1:
                epsilon_decay = (self.start_epsilon - 0.1) / (self.num_episodes / 2)
                self.args.epsilon = self.start_epsilon - epsilon_decay * episode

            while not done:
                action = self.epsilon_greedy_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_observation = self.encode_state(next_observation)
                episode_step += 1
                episode_reward += reward

                next_action = self.epsilon_greedy_action(next_observation)

                if next_observation not in self.q_table:
                    self.q_table[next_observation] = np.zeros(self.env.action_space.n)

                td_error = reward + self.gamma * self.q_table[next_observation][next_action] - self.q_table[observation][
                    action]
                self.q_table[observation][action] += self.alpha * td_error
                episode_td_error += td_error

                training_time_steps += 1  # Q-table 업데이트 횟수

                visited_states.append(next_observation)
                observation = next_observation

                done = terminated or truncated

            print(
                "[EPISODE: {0:>2}]".format(episode + 1, observation),
                "Episode Steps: {0:>2}, Visited States Length: {1:>2}, Episode Reward: {2}".format(
                    episode_step, len(visited_states), episode_reward
                ),
                "GOAL" if done and observation == 15 else ""
            )
            episode_reward_list.append(episode_reward)
            episode_td_error_list.append(episode_td_error / episode_step)

            if (episode + 1) % 1000 == 0:
                episode_reward_list_test, avg_episode_reward_test = self.validate()
                print("[VALIDATION RESULTS: {0} Episodes, Episode Reward List: {1}] Episode Reward Mean: {2:.3f}".format(
                    self.validation_num_episodes, episode_reward_list_test, avg_episode_reward_test
                ))
                if avg_episode_reward_test == 1:
                    print("***** TRAINING DONE!!! *****")
                    is_train_success = True
                    break

        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "q_table_sarsa.pkl")
        self.save_model(save_path)

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for episode in range(self.validation_num_episodes):
            episode_reward = 0  # cumulative_reward
            episode_step = 1

            observation, _ = self.test_env.reset()

            observation = self.encode_state(observation)

            done = truncated = False
            while not done and not truncated:
                action = self.greedy_action(observation)
                next_observation, reward, done, truncated, _ = self.test_env.step(action)
                next_observation = self.encode_state(next_observation)
                episode_reward += reward
                observation = next_observation
                episode_step += 1

            episode_reward_lst[episode] = episode_reward

        return episode_reward_lst, np.mean(episode_reward_lst)
    
    def evaluate(self, num_ep = 10):
        episode_rewards = []

        self.test_env = RecordVideo(self.test_env, video_folder="./videos", name_prefix="minigrid")

        for episode in range(num_ep):
            observation, _ = self.test_env.reset()
            observation = self.encode_state(observation)
            total_reward = 0
            done = False

            while not done:
                # 탐험 없이 greedy action 사용
                if observation in self.q_table:
                    action = np.argmax(self.q_table[observation])
                else:
                    action = self.test_env.action_space.sample()  # unseen state fallback

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)
                next_observation = self.encode_state(next_observation)

                total_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards)
        print("[EVALUATE] Episodes: {0}, Rewards: {1}, Average: {2:.3f}".format(
            num_ep, episode_rewards, avg_reward
        ))
        return episode_rewards, avg_reward