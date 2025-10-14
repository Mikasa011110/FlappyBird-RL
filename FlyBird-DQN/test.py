import torch
import numpy as np
from collections import deque
from game.wrapped_flappy_bird import GameState
from utils import preprocess
from network import Qnet

# ================= 超参数（和训练时一致） =================
frames = 4
hidden_dim = 256
action_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "save/dqn_flappy_1800.pth"  # 你的模型路径
RENDER = True  # 如果想看游戏画面，设为 True

# ================= 加载模型 =================
q_net = Qnet(frames, hidden_dim, action_dim).to(device)
q_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
q_net.eval()

print(f"✅ Loaded model from {MODEL_PATH}")

# ================= 初始化游戏 =================
env = GameState(render = True)
do_nothing = np.zeros(action_dim)
do_nothing[0] = 1
image, _, _ = env.frame_step(do_nothing)
image = preprocess(image)

frames_deque = deque([image] * 4, maxlen=4)
state = np.stack(frames_deque, axis=0)

episode_return = 0
step = 0

# ================= 游戏循环 =================
while True:
    step += 1
    # 选动作（不探索，纯策略）
    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
    with torch.no_grad():
        action_index = q_net(state_tensor).argmax().item()

    action = np.zeros(action_dim, dtype=int)
    action[action_index] = 1

    # 与环境交互
    next_image, reward, done = env.frame_step(action)
    next_image = preprocess(next_image)
    frames_deque.append(next_image)
    next_state = np.stack(frames_deque, axis=0)

    state = next_state
    episode_return += reward

    # 输出调试信息
    if step % 10 == 0:
        print(f"Step: {step}, Action: {action_index}, Reward: {reward:.2f}, Total Return: {episode_return:.2f}")

    if done:
        print(f"\n🏁 Episode finished! Total Reward: {episode_return:.2f}, Steps: {step}")
        break

print("✅ Test finished.")