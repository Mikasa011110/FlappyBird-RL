import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import random

from network import Qnet
from game.wrapped_flappy_bird import GameState
from utils import *
from utils import ReplayBuffer

# ====================== 超参数设置 ============================
learning_rate = 1e-4
num_episodes = 30000
hidden_dim = 256
gamma = 0.99
target_update = 500
buffer_size = 50000      # 重放内存大小
minimal_size = 20000
batch_size = 32
frames = 4
action_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_PER_ACTION = 1 # 动作采样间隔

# ε 贪婪策略参数（作者推荐设定）
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
OBSERVE = 10000             # 前 10000 步纯随机
EXPLORE = 300000           # ε 从 0.1 -> 0.0001 的退火步数
epsilon = INITIAL_EPSILON
FIXED_EPSILON = 0.001       # annealing 完成后固定值



def preprocess(image):
    '''预处理图像,将图像转换为灰度图像，调整大小为80x80，并归一化到0-1之间'''
    image = Image.fromarray(image)
    image = image.convert('L')  # 转为灰度图
    image = image.resize((80, 80))  # 调整大小
    image = np.array(image)
    image = image / 255.0  # 归一化
    return image


class ReplayBuffer:
    '''经验回放池'''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 双端队列，先进先出, 超出maxlen时，自动删除最早的元素
    
    def add(self, state, action, reward, next_state, done): # 添加经验
        self.buffer.append((state, action, reward, next_state, done)) # done是一个布尔值，True表示到达episode终点
    
    def sample(self, batch_size): # 随机抽取旧的经验供训练（uniform distribution)
        transitions = random.sample(self.buffer, batch_size) # 从buffer中随机抽取batch_size个经验
        state, action, reward, next_state, done = zip(*transitions) # 解压
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self): # 目前buffer中数据的数量
        return len(self.buffer)
    

class DQN:
    '''DQN 智能体'''
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

        loss = F.smooth_l1_loss(q_values, target_q_values)
        #print(f"loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ✅ 打印 Q 值统计信息（每隔若干步）
        if self.count % 500 == 0:
            with torch.no_grad():
                all_q = self.q_net(states)
                print(f"[Q Stats] step={self.count:<6} "
                      f"q_min={all_q.min().item():.3f} "
                      f"q_max={all_q.max().item():.3f} "
                      f"q_mean={all_q.mean().item():.3f} "
                      f"loss={loss.item():.5f}")

        # 定期同步目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# ====================== 训练函数 ============================
def train(num_episodes=num_episodes):
    dqn_agent = None
    try:
        env = GameState(render = False)
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
                        action_index = 1 if np.random.rand() < 0.4 else 0
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
                next_image = preprocess(next_image)
                frames_deque.append(next_image)
                next_state = np.stack(frames_deque, axis=0)
                last_action_index = action_index

                # 存入replay
                replay_buffer.add(state, action_index, reward, next_state, done)

                state = next_state
                episode_return += reward

                # 预热阶段之后开始更新qnet
                if step > OBSERVE and replay_buffer.size() > minimal_size:
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
            if (episode + 1) % 100 == 0:
                np.save("save/return_list.npy", np.array(return_list))
            print(f"Episode {episode+1}/{num_episodes} | Return: {episode_return:.2f} | Steps: {step_in_episode} | Epsilon: {epsilon:.4f} | Buffer: {replay_buffer.size()} | Total Steps: {step}")

            # 定期保存
            if (episode + 1) % 200 == 0:
                torch.save(dqn_agent.q_net.state_dict(), f"save/dqn_flappy_{episode+1}.pth")

            if episode < 20:
                print("Action Index:", action_index, "Action Vector:", action)

        # 训练结束保存最终模型
        torch.save(dqn_agent.q_net.state_dict(), f"save/dqn_flappy_final.pth")
        return return_list
    
    except KeyboardInterrupt:
        if dqn_agent is not None:
            print("\n Training interrupted by user. Saving current model...")
            torch.save(dqn_agent.q_net.state_dict(), "save/dqn_flappy_interrupt.pth")
            np.save("save/return_list.npy", np.array(return_list))
            print(" Model saved successfully after interruption.")
        else:
            print("\n Interrupted before model initialization — nothing to save.")
        return return_list if 'return_list' in locals() else []

# ====================== 主程序入口 ============================
if __name__ == "__main__":
    train()

    


    