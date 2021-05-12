import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math

from model import DQN
from replay_memory import ReplayMemory

# Hyperparameters
REPLAY_CAPACITY = 2500
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10000
USE_CUDA = torch.cuda.is_available()

class MarioAgent:
    def __init__(self, n_inputs, n_actions, lr=0.02, net_type='dqn'):
        self.replay_memory = ReplayMemory(REPLAY_CAPACITY, USE_CUDA)

        self.step = 0
        self.onlineNet = DQN(n_inputs, n_actions).float()
        if USE_CUDA: 
            self.onlineNet = self.onlineNet.to(device='cuda')
        if net_type == 'ddqn':
            self.targetNet = DQN(n_inputs, n_actions).float()
            if USE_CUDA: 
                self.targetNet = self.targetNet.to(device='cuda')
        
        self.optimizer = optim.Adam(self.onlineNet.parameters, lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.step / EPS_DECAY)
        self.step += 1

        if sample > eps_threshold:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.onlineNet(state)
            action_idx = torch.argmax(action_values, axis=1).item()
        else:
            action_idx = np.random.randint(self.action_dim)

        return action_idx

    def learn(self, state, next_state, action, reward, done):

        if self.step % TARGET_UPDATE == 0 and self.net_type == 'ddqn':
            self.targetNet.load_state_dict(self.OnlineNet.state_dict())

        # Push to replay memory
        self.replay_memory.push(state, next_state, action, reward, done)

        # Sample from the memory
        state, next_state, action, reward, done = self.replay_memory.sample(BATCH_SIZE)

        Q = self.onlineNet(state)[np.arange(0, BATCH_SIZE), action]
        Q_next = self.onlineNet(next_state)

        if self.net_type == 'ddqn':
            Q_next = self.targetNet(next_state)[np.arange(0, BATCH_SIZE), torch.argmax(Q_next, axis=1)]
        
        Q_target = (reward + (1 - done.float()) * GAMMA * Q_next).float()

        loss = self.loss_fn(Q, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return Q.mean().item(), loss.item()

    def save(self, filepath):
        torch.save(dict(online=self.onlineNet.state_dict(), 
                        target=self.targetNet.state_dict(),
                        step=self.step),
                    filepath)
        print(f"Model saved to {filepath}.")

    def load(self, filepath):
        if not filepath.exists():
            raise ValueError(f"{filepath} does not exist")

        ckp = torch.load(filepath, map_location=('cuda' if USE_CUDA else 'cpu'))
        self.step = ckp.get('step')
        self.onlineNet.load_state_dict(ckp.get('online'))
        self.targetNet.load_state_dict(ckp.get('target'))

        print(f"Model loaded from {filepath}.")




    



