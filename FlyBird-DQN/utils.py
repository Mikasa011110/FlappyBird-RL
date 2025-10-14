from PIL import Image
import numpy as np
import collections
import random
import torch

# ====================== 超参数设置 ============================
learning_rate = 1e-6
num_episodes = 30000
hidden_dim = 256
gamma = 0.99
target_update = 1000
buffer_size = 50000      # 重放内存大小
minimal_size = 50000
batch_size = 32
frames = 4
action_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_PER_ACTION = 2 # 动作采样间隔

# ε 贪婪策略参数（作者推荐设定）
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
OBSERVE = 10000             # 前 10000 步纯随机
EXPLORE = 3000000           # ε 从 0.1 -> 0.0001 的退火步数
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
    '''
    输出类似：
    state = ([1,2], [2,3], [3,4])
    action = (0, 1, 0)
    reward = (1, 1, -1)
    next_state = ([1,3], [2,4], [3,5])
    done = (False, False, True)
    '''
    def size(self): # 目前buffer中数据的数量
        return len(self.buffer)