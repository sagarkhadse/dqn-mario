import random
import torch
from collections import deque


class ReplayMemory(object):

    def __init__(self, capacity, use_cuda = False):
        self.memory = deque(maxlen=capacity)
        self.use_cuda = use_cuda

    def push(self, state, next_state, action, reward, done):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])
        self.memory.append((state, next_state, action, reward, done,))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def __len__(self):
        return len(self.memory)