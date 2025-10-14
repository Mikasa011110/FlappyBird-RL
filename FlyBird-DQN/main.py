from train import train
from utils import *
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # 解决plt.show 报错问题

if __name__ == "__main__":
    num_episodes = 30000
    return_list = train(num_episodes)

    # 绘图
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    returns_smooth = moving_average(return_list, window_size=10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(returns_smooth)), returns_smooth, color='blue', label='Smoothed Returns (10-episode MA)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Return vs Episodes (Smoothed)')
    plt.grid(True)
    plt.legend()
    plt.savefig('return_curve.png', dpi=300, bbox_inches='tight')  # 高分辨率 & 边缘紧凑
    plt.show()
