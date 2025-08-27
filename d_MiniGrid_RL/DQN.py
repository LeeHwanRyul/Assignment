import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
import time
import os
from shutil import copyfile

from gymnasium.wrappers import RecordVideo

import wandb

import matplotlib.pyplot as plt

wandb.login(key="dd9058e8c2d7ddf2080f6c1af5c5af4c09940373")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, n_features, n_actions, in_channels=4):
        super(QNet, self).__init__()
        self.n_actions = n_actions

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2),  # (1,7,7) -> (32,3,3)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # (32,3,3) -> (64,1,1)
            nn.ReLU(),
            nn.Flatten()
        )

        # dummy input으로 feature 크기 자동 계산
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 7, 7)
            n_features = self.conv(dummy).shape[1]

        # Fully connected Q-value head
        self.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        # np.ndarray -> torch.tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        # CNN + FC
        features = self.conv(x)
        q_values = self.fc(features)
        return q_values
    
    def get_action(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            action = random.randrange(0, self.n_actions)
        else:
            q_values = self.forward(obs)
            action = torch.argmax(q_values, dim=-1)
            action = action.item()
        return action

Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done']
)

# Agent는 매 step마다 action을 취하는데 n, n+1 step은 시계열적으로 연관되어 있어 학습에
# 방해가 될 수 있다. 그러므로 Agent의 state, action을 버퍼에 저장한 뒤에 무작위로 샘플링한다.
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        observations = np.array(observations)
        next_observations = np.array(next_observations)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)

        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
    
class DQN:
    def __init__(self, make_env, args):
        self.args = args
        env = make_env(args, args.render_mode)
        self.test_env = make_env(args, "rgb_array")

        self.epsilon = args.epsilon
        self.start_epsilon = args.epsilon

        self.device = args.device
        print("device:", self.device)
        self.env = env
        self.num_episodes = args.num_episodes
        self.validation_num_episodes = args.validation_num_episodes
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.is_evaluate = args.is_evaluate
        self.replay_buffer_size = args.replay_buffer_size
        self.steps_between_train = args.steps_between_train
        self.episode_reward_avg_solved = args.episode_reward_avg_solved
        self.train_num_episodes_before_next_test = args.train_num_episodes_before_next_test
        self.target_sync_step_interval = args.target_sync_step_interval

        self.wandb = wandb.init(
            project="DQN_minigrid",
            config=vars(args)
        )

        # n_features : (7, 7, 3)
        # n_actions  : 3       
        self.q = QNet(
            env.observation_space['image'].shape, 3).to(self.device)
        self.target_q = QNet(
            env.observation_space['image'].shape, 3).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.time_steps = 0
        self.train_time_step = 0
        self.total_time_steps = 0

    def model_save(self, validation_episode_reward_avg):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "dqn_{0}.pth".format(
            validation_episode_reward_avg
        )
        torch.save(self.q.state_dict(), os.path.join(current_dir, filename))

        copyfile(
            src=os.path.join(current_dir, filename),
            dst=os.path.join(current_dir, "dqn.pth")
        )

    def load_model(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_params = torch.load(os.path.join(current_dir, filename))
        self.q.load_state_dict(model_params)

    def train(self):
        episode_rewards = []
        total_train_start_time = time.time()

        epsilon = self.start_epsilon

        for n_episode in range(1, self.num_episodes + 1):
            if epsilon >= 0.1:
                epsilon_decay = (self.start_epsilon - 0.1) / self.num_episodes
                epsilon = self.start_epsilon - epsilon_decay * n_episode

            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                action = self.q.get_action(observation, epsilon)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, done)

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.total_time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    loss = self.model_train()

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % 10 == 0:
                print(
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>5},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:6.3f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Elapsed Time: {}".format(total_training_time)
                )

            self.wandb.log({
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                "[TRAIN] Epsilon": epsilon,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "Training Episode": n_episode,
            })

            episode_rewards.append(episode_reward)

            if n_episode % self.train_num_episodes_before_next_test == 0:
                validation_episode_reward_lst, validation_episode_reward_avg = self.validate()
                print("[VALIDATION RESULTS: {0}] Episode Reward Average: {1:.3f}".format(
                    validation_episode_reward_lst, validation_episode_reward_avg
                ))
                if validation_episode_reward_avg > self.episode_reward_avg_solved:
                    print("***** TRAINING DONE!!! *****")

        self.model_save(validation_episode_reward_avg)

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.show()

    def model_train(self):
        self.train_time_step += 1

        batch = self.replay_buffer.sample(self.batch_size)

        observation, action, next_observation, reward, done = batch

        # q 추출
        q_out = self.q(observation)
        # print(q_out)
        q_value = q_out.gather(dim=-1, index=action)

        with torch.no_grad():
            q_any = self.target_q(next_observation)
            # 각 배치상태에서 가장 큰 Q (maxQ(s,a)) 선택
            max_q_any = q_any.max(dim=1, keepdim=True).values
            max_q_any[done] = 0.0

            # target = r + gamma*maxQ(s,a)
            target = reward + self.gamma * max_q_any

        # loss = 1/N * Sigma(Q(s,a) - target)^2
        # q_value = Q(s,a)
        # target.detach() = target = r + gamma*maxQ(s,a)
        loss = F.mse_loss(target.detach(), q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target Q는 특정 간격으로 느리게 업데이트 되어야 함.
        # 이유: Q(s,a)와 maxQ(s',a')이 동일 네트워크 상에서 업데이트 되면서
        # maxQ를 Q가 따라잡지 못해 학습이 불안정해진다.
        if self.time_steps % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()
    
    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                action = self.q.get_action(observation, epsilon=0.0)
                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)

    def evaluate(self):
        env = self.test_env
        q = self.q
        num_episodes = 5

        env = RecordVideo(env, video_folder="./videos", name_prefix="minigrid")

        for i in range(num_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation, _ = env.reset()

            episode_steps = 0

            done = False

            while not done:
                episode_steps += 1
                action = q.get_action(observation, epsilon=0.0)

                next_observation, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
                i, episode_steps, episode_reward
            ))