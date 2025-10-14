import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network import Qnet
from game.wrapped_flappy_bird import GameState
from utils import *
from utils import ReplayBuffer


class DQN:
    '''DQN 智能体'''
    learning_rate = 1e-5
    def __init__(self, frames, hidden_dim, action_dim, learning_rate, gamma, target_update, device):
        self.device = device 
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update = target_update
        self.count = 0

        # 创建网络
        self.q_net = Qnet(frames, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device)
        self.target_q_net = Qnet(frames, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device)

        # 权重初始化 (N(0, 0.01))
        self._init_weights(self.q_net)
        self._init_weights(self.target_q_net)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def take_action(self, state, epsilon):
        '''ε-贪婪选动作'''
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                return self.q_net(state).argmax().item()

    def update(self, transition_dict):
        '''从经验池采样一批数据并更新Q网络'''
        states = torch.tensor(transition_dict['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        #print(f"loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期同步目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# ====================== 训练函数 ============================
def train(num_episodes=num_episodes):
    env = GameState()
    replay_buffer = ReplayBuffer(buffer_size)
    dqn_agent = DQN(frames, hidden_dim, action_dim, learning_rate, gamma, target_update, device)

    os.makedirs("save", exist_ok=True)
    step = 0
    epsilon = INITIAL_EPSILON
    return_list = []

    for episode in range(num_episodes):
        done = False
        episode_return = 0

        # 初始状态（执行一次 do_nothing）
        do_nothing = np.zeros(action_dim)
        do_nothing[0] = 1
        image, _, _ = env.frame_step(do_nothing)
        image = preprocess(image)
        frames_deque = deque([image] * 4, maxlen=4)
        state = np.stack(frames_deque, axis=0)
        last_action_index = 0

        '''开始当前episode'''
        step_in_episode = 0
        max_steps_per_episode = 10000  # 比如 10000 帧

        while not done and step_in_episode < max_steps_per_episode:
            step += 1
            step_in_episode += 1

            # --- 动作频率控制 ---
            if step % FRAME_PER_ACTION == 0:
                if step <= OBSERVE:
                    # 预热阶段前 10000 步纯随机：50% flap, 50% 不动
                    action_index = 1 if np.random.rand() < 0.35 else 0
                else:
                    action_index = dqn_agent.take_action(state, epsilon)
            # 否则重复上一个动作（保持）
            else:
                action_index = last_action_index

            # 转为 one-hot 动作向量
            action = np.zeros(action_dim, dtype=int)
            action[action_index] = 1

            # 环境交互
            next_image, reward, done = env.frame_step(action)
            reward = np.clip(reward, -1, 1) # 奖励裁剪
            next_image = preprocess(next_image)
            frames_deque.append(next_image)
            next_state = np.stack(frames_deque, axis=0)
            last_action_index = action_index
            
            # 存入replay
            replay_buffer.add(state, action_index, reward, next_state, done)

            state = next_state
            episode_return += reward

            # 预热阶段之后开始，网络更新
            if step > OBSERVE:
                batch = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'rewards': batch[2],
                    'next_states': batch[3],
                    'dones': batch[4]
                }
                dqn_agent.update(transition_dict)

                # ε线性退火
                if epsilon > FINAL_EPSILON and step < EXPLORE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                else:
                    epsilon = 0.001 # 退火完成后，ε = 0.001

        # ================== Episode End ==================
        return_list.append(episode_return)
        print(f"Episode {episode+1}/{num_episodes} | Return: {episode_return:.2f} | Steps: {step_in_episode} | Epsilon: {epsilon:.4f} | Buffer: {replay_buffer.size()} | Total Steps: {step}")

        # 定期保存
        if (episode + 1) % 1000 == 0:
            torch.save(dqn_agent.q_net.state_dict(), f"save/dqn_flappy_{episode+1}.pth")
        
        if episode < 20:
            print("Action Index:", action_index, "Action Vector:", action)

    # 训练结束保存最终模型
    torch.save(dqn_agent.q_net.state_dict(), f"save/dqn_flappy_final.pth")
    return return_list


# ====================== 主程序入口 ============================
if __name__ == "__main__":
    train()
    


    