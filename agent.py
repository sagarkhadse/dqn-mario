# ------------------------------------------------------------------------------------------------ #
# agent.py - Mario Agent that uses DDQN/DQN to train on a super mario bros environment
# ------------------------------------------------------------------------------------------------ #
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math

from model import DQN
from replay_memory import ReplayMemory

# Hyperparameters
REPLAY_CAPACITY = 25000
BATCH_SIZE = 16
GAMMA = 0.9
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 0.9999975
TARGET_UPDATE = 10000
BURNIN = 1280
USE_CUDA = torch.cuda.is_available()

class MarioAgent:
    def __init__(self, n_inputs, n_actions, lr=0.02, net_type='dqn'):
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.net_type = net_type
        self.replay_memory = ReplayMemory(REPLAY_CAPACITY, USE_CUDA)

        self.step = 0
        self.eps_threshold = EPS_START
        self.onlineNet = DQN(n_inputs, n_actions).float()
        self.targetNet = DQN(n_inputs, n_actions).float()
        if USE_CUDA: 
            self.onlineNet = self.onlineNet.to(device='cuda')
            self.targetNet = self.targetNet.to(device='cuda')
        
        self.optimizer = optim.Adam(self.onlineNet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """Select the best action

        Args:
            state: current state

        Returns:
            action: best action
        """
        sample = np.random.rand()
        self.eps_threshold *= EPS_DECAY
        self.eps_threshold = max(EPS_END, self.eps_threshold)
        self.step += 1

        if sample > self.eps_threshold:
            state = torch.FloatTensor(state).cuda() if USE_CUDA else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.onlineNet(state)
            action_idx = torch.argmax(action_values, axis=1).item()
        else:
            action_idx = np.random.randint(self.n_actions)
        
        return action_idx

    def learn(self, state, next_state, action, reward, done):
        """Learn based on the given parameters

        Args:
            state: Current state
            next_state: Next state
            action: Action selected
            reward: Reward for the state and action
            done: Completion function

        Returns:
            q, loss: q value, loss value
        """
        # Push to replay memory
        self.replay_memory.push(state, next_state, action, reward, done)

        if self.step % TARGET_UPDATE == 0 and self.net_type == 'ddqn':
            self.targetNet.load_state_dict(self.OnlineNet.state_dict())

        if self.step < BURNIN:
            return None, None

        # Sample from the memory
        state, next_state, action, reward, done = self.replay_memory.sample(BATCH_SIZE)

        Q = self.onlineNet(state)[np.arange(0, BATCH_SIZE), action]

        with torch.no_grad():
            Q_next = self.onlineNet(next_state)
            Q_next = torch.argmax(Q_next, axis=1)

            if self.net_type == 'ddqn':
                Q_next = self.targetNet(next_state)[np.arange(0, BATCH_SIZE), Q_next]
        
            Q_target = (reward + (1 - done.float()) * GAMMA * Q_next).float()

        loss = self.loss_fn(Q, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return Q.mean().item(), loss.item()

    def save(self, filepath):
        """Saves the model

        Args:
            filepath (string): file path to store the model in
        """
        torch.save(dict(online=self.onlineNet.state_dict(), 
                        target=self.targetNet.state_dict(),
                        step=self.step,
                        epsilon=self.eps_threshold),
                    filepath)
        print(f"Model saved to {filepath}.")

    def load(self, filepath):
        """Load a model from a file

        Args:
            filepath (string ): file path to load the model from

        Raises:
            ValueError: error if file not found
        """
        if not filepath.exists():
            raise ValueError(f"{filepath} does not exist")

        ckp = torch.load(filepath, map_location=('cuda' if USE_CUDA else 'cpu'))
        self.step = ckp.get('step')
        self.eps_threshold = ckp.get('epsilon')
        self.onlineNet.load_state_dict(ckp.get('online'))
        self.targetNet.load_state_dict(ckp.get('target'))

        print(f"Model loaded from {filepath}.")




    



