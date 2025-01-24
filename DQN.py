import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义经验元组，用于存储经验
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.to(self.fc1.weight.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, input_size, output_size, batch_size=256, gamma=0.9, epsilon=1.0, epsilon_decay=0.998,
                 epsilon_min=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.losses = []

        # 定义深度 Q 网络和目标 Q 网络
        self.q_network = DQNNetwork(input_size, output_size).to(device)
        self.target_q_network = DQNNetwork(input_size, output_size).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        # self.target_q_network.eval()
        # 定义优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()  # 误差函数
        # 定义经验回放缓存
        self.replay_buffer = deque(maxlen=1000)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.995)

    def choose_action(self, state):
        # epsilon-greedy策略选择动作
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.output_size, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def store_experience(self, experience):
        # 存储经验到经验回放缓存
        self.replay_buffer.append(experience)

    def sample_batch(self):
        # 从经验回放缓存中随机采样一个批次
        batch = random.sample(self.replay_buffer, self.batch_size)

        # 正确地解包 Experience 元组
        state_batch = torch.stack([torch.tensor(experience[0], dtype=torch.float32).to(device) for experience in batch])
        action_batch = torch.tensor([experience[1] for experience in batch], dtype=torch.long).to(device)
        next_state_batch = torch.stack([torch.tensor(experience[2], dtype=torch.float32).to(device) for experience in batch])
        reward_batch = torch.tensor([experience[3] for experience in batch], dtype=torch.float32).to(device)

        return state_batch, action_batch, next_state_batch, reward_batch

    def update_q_network(self, batch):
        # 计算 Q 值的损失
        state_batch, action_batch, next_state_batch, reward_batch = batch

        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_q_network(next_state_batch).max(1).values.detach()
        target_q_values = reward_batch + self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))

        # next_q_values = self.q_network(next_state_batch).detach()
        #
        # best_next_action = torch.max(next_q_values.cpu(), 1)[1].data.numpy()[0]
        # next_q_target = self.target_q_network(next_state_batch).detach()
        # q_next_target_eval = next_q_target[0][best_next_action]
        #
        # target = reward_batch + self.gamma * q_next_target_eval
        #
        # q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        #
        # loss = self.loss_func(q_values, target)
        # 更新 Q 网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)  # 梯度裁剪
        self.optimizer.step()
        # print(f"---------------------{loss.item()}-----------------------")
        self.losses.append(loss.item())

    def update_target_q_network(self):
        # 更新目标 Q 网络的权重
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, episode):
        # 在每个训练步骤中执行一次训练
        if len(self.replay_buffer) >= self.batch_size:
            batch = self.sample_batch()
            self.update_q_network(batch)
            # if episode % 300 == 0:
            self.update_target_q_network()
            # 调整 epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save_model(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)
