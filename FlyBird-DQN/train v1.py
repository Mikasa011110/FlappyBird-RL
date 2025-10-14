from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network import Qnet
from game.wrapped_flappy_bird import GameState
from utils import *
from utils import INITIAL_EPSILON


class DQN:
    '''DQN算法'''
    def __init__(self, frames, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.device = device 
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim # 动作维度
        self.frames = frames # 每次输入为4帧
        self.q_net = Qnet(frames, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device) # 训练网络
        self.target_q_net = Qnet(frames, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device) # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate) # 优化器，Adam算法

        self.gamma = gamma # 折扣因子
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 记录步数，记录与环境交互的总步数


    def take_action(self, state, epsilon):
        '''根据训练网络近似Q(s,a)，使用ε-贪婪算法选择动作a'''
        if np.random.random() < epsilon:  # 所以ε越小，探索越少，利用越多
            action = np.random.randint(self.action_dim) # 探索：随机选择动作
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device) # 转换成张量，并添加一个维度用于存放batch_size
            with torch.no_grad():
                action = self.q_net(state).argmax().item() # 将state张量输入训练网络，得到Q值最大的动作
        return action

    def update(self, transition_dict):
        '''使用经验回放池中的一批经验，更新训练网络'''
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device) # view(-1, 1)将actions转换成列向量
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) # view(-1, 1)转换成列向量
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device) # view(-1, 1)转换成列向量

        # 使用训练网络计算当前Q值
        q_values = self.q_net(states).gather(1, actions) # gather()根据actions索引，选出对应的Q值

        # 使用目标网络计算目标Q值（y值）
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) #使用目标网络计算的下一个状态的最大Q值
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones) # 如果done是True，说明到达终点，不需要加上折扣后的下一个状态的Q值

        # 计算损失函数
        loss = torch.mean(F.mse_loss(q_values, target_q_values)) # 均方误差损失函数（当前Q值和目标Q值的差距）

        # 优化训练网络
        self.optimizer.zero_grad() # PyTorch默认梯度累积，这里梯度清零
        loss.backward() # 反向传播
        self.optimizer.step() # 更新参数

        # 每隔target_update步，将训练网络的参数复制到目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # load_state_dict()将一个网络的参数加载到另一个网络中
        self.count += 1
    
def train(num_episodes=500):
    '''初始化'''
    env = GameState()
    replay_buffer = ReplayBuffer(buffer_size)
    epsilon = INITIAL_EPSILON
    dqn_agent = DQN(
        frames=frames,
        hidden_dim=128,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        target_update=target_update,
        device=device
    )
    return_list = []

    step = 0  # 全局交互步数计数，用于计算epsilon衰减

    for episode in range(num_episodes):
        '''每局初始化环境和状态'''
        env = GameState()
        done = False
        episode_return = 0
        
        '''先执行一次action=不动，获取一帧图像'''
        do_nothing = np.zeros(action_dim)
        do_nothing[0] = 1
        image, _, _ = env.frame_step(do_nothing)
        image = preprocess(image)

        frames_deque = deque([image] * 4, maxlen=4) # 使用一个长度为4的队列，保存最近4帧图像
        state = np.stack(frames_deque, axis=0) # 4帧图像为一个state

        '''开始当前episode'''
        step_in_episode = 0
        max_steps_per_episode = 10000  # 比如 10000 帧
        while not done and step_in_episode < max_steps_per_episode:
            # ε-贪婪选择动作
            action_index = dqn_agent.take_action(state, epsilon)

            # 转成 one-hot 动作向量(由于环境返回的是0或者1)
            action = np.zeros(action_dim)
            action[action_index] = 1

            # 与环境交互
            next_image, reward, done = env.frame_step(action)
            reward = np.clip(reward, -1, 1) # 奖励裁剪
            next_image = preprocess(next_image)
            step += 1

            # 将新图像存入队列deque，获得next state
            frames_deque.append(next_image)
            next_state = np.stack(frames_deque, axis=0)

            # 存入经验池
            replay_buffer.add(state, action_index, reward, next_state, done)

            # 更新状态与回报
            state = next_state
            episode_return += reward

            # 当经验足够时更新网络
            if replay_buffer.size() > minimal_size:
                batch = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'rewards': batch[2],
                    'next_states': batch[3],
                    'dones': batch[4]
                }
                dqn_agent.update(transition_dict)


            # 7. 衰减 ε
            epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - step * 1e-6)
            
            # 超过上限强行结束
            step_in_episode += 1
            if step_in_episode >= max_steps_per_episode:
                done = True


        # 回合结束，记录结果
        return_list.append(episode_return)
        print(f"Episode {episode+1}/{num_episodes} | Return: {episode_return:.2f} | Steps: {step_in_episode} | Epsilon: {epsilon:.4f} | Buffer: {replay_buffer.size()} | Total Steps: {step}")
    
    torch.save(dqn_agent.q_net.state_dict(), f"save/dqn_flappy.pth")    
    return return_list

if __name__ == "__main__":
    train()


    