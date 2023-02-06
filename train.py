"""
使用ddqn训练神折纸2
作者：laidage
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from game_logic import Env
from itertools import count
import os
import logging
from net import Net
logging.basicConfig(filename='logger.log', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# 将ddqn训练需要的数据进行标准化处理，包括[当前游戏状态，动作，奖励，执行动作后的游戏状态，游戏是否已结束]
Transition = namedtuple("Transition", ['state', 'action', 'reward', 'next_state', 'done'])

# 参数
BATCH_SIZE = 128 # 单次训练样本个数
GAMMA = 0.9 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
EPS = 0.9 # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
STEPS_DONE = 0
COLORS = 5

def choose_action(state, env):
    # 选择动作策略
    rand = random.random()
    if rand <= EPS:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(290 * (COLORS - 1))]], device=device, dtype=torch.long)


class ReplayMemory():
    # 使用经验回放来提高训练效率
    def __init__(self, capacity=20000):
        self.memory = deque([], capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# ddqn需要使用两个神经网络
policy_net = Net().to(device) # 策略网络
target_net = Net().to(device) # 目标网络
if os.path.exists("model/net.pth"):
    policy_net.load_state_dict(torch.load("model/net.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# 使用adam优化器，稳定出效果
optimizer = optim.Adam(policy_net.parameters(), lr=0.0001) 
replayMemory = ReplayMemory()

def optimize_model():
    # 游戏过程中，不断优化模型
    if len(replayMemory) < BATCH_SIZE:
        return
    batch = replayMemory.sample(BATCH_SIZE)
    batch = Transition(*zip(*batch))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    dones = torch.cat(batch.done)
    # 本来考虑用添加mask的方法剪除非法动作，最终使用了另一种方法
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # next_state_values = policy_net(state_batch).gather(1, action_batch).detach()

    # 避免过度估计，使用了dqn的改进算法ddqn
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    q_next = target_net(next_state_batch).detach()
    q_next_eval = policy_net(next_state_batch).detach()
    q_a = q_next_eval.argmax(dim=1)
    q_a = torch.reshape(q_a,(-1,1))
    expected_state_action_values =  GAMMA * q_next.gather(1, q_a) * (1 - dones).unsqueeze(1) + reward_batch.unsqueeze(1)
    loss_func = nn.SmoothL1Loss()
    loss = loss_func(state_action_values, expected_state_action_values) # 计算误差
    logging.info(loss)

    optimizer.zero_grad()
    loss.backward() # 反向传播
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train():
    # ddqn训练的整体流程，不断与环境交互todo
    episode = 200001
    env = Env()
    global COLORS, STEPS_DONE, EPS
    for i in range(1, episode):
        if STEPS_DONE < 1000:
            # 随着轮次增加，减小探索率
            EPS += 0.00009
        k = random.randint(1, 42) # 选取42个关卡进行训练
        file_path = "config/"+ str(k)
        state, COLORS = env.load_config_file(file_path)
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        for t in count():
            logging.info("i: {} t: {}".format(i, t))
            action = choose_action(state, env)
            state_, reward, done = env.step(action.item())
            state_ = torch.tensor(state_, dtype=torch.float64).unsqueeze(0)
            done_int = 0
            if done:
                done_int = 1
            reward = torch.tensor([reward], dtype=torch.float64, device=device)
            done_int = torch.tensor([done_int], dtype=torch.int, device=device)
            replayMemory.push(state, action, reward, state_, done_int)
            
            state = state_
            optimize_model()
            if done:
                break
        if i % 10 == 0:
            # 一定轮次后对目标网络进行更新
            target_net.load_state_dict(policy_net.state_dict())
        if i % 100 == 0:
            # 一定轮次后保存模型
            if not os.path.isdir("model"):
                os.makedirs("model")
            torch.save(policy_net.state_dict(), "model/net.pth")

if __name__ == "__main__":
    train()