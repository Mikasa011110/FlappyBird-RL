import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # 解决plt.show 报错问题

if __name__ == "__main__":
    returns = np.load("save/return_list.npy")

    # 计算滑动平均 (窗口大小=50)
    window = 50
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(8, 5))
    plt.plot(returns, color='lightgray', label="Raw Return")
    plt.plot(range(window-1, len(returns)), smoothed, color='blue', label=f"Smoothed ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Flappy Bird DQN Training Curve (Smoothed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("save/training_curve.png", dpi=300) 
    plt.show()