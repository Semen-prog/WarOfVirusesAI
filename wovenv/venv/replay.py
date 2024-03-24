import torch
from collections import deque
from .utils import write_error
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1)
        self.data = [None for _ in range(capacity)]
        self.n_entries = 0
        self.write = 0

    def add(self, p, data):
        tree_index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_index, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def len(self):
        return self.tree.n_entries

    def update_beta(self):
        self.beta += (1 - self.beta) / 10

    def add_experience(self, state, action, next_state, reward, done):
        max_priority = torch.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.max_priority
        self.tree.add(max_priority, (state, torch.tensor(action), torch.tensor(reward), next_state, torch.tensor(done)))

    def sample(self, batch_size):
        batch, priorities, index = [], [], []
        segment = self.tree.total() / batch_size
        self.update_beta()
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            index.append(idx)
            priorities.append(p)
            batch.append(data)
        sampling_probabilities = torch.tensor(priorities) / self.tree.total()
        is_weight = torch.pow(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return states.to(device), actions.to(device), next_states.to(device), rewards.to(device), dones.to(device), torch.tensor(index).to(device), is_weight.to(device)

    def update_priorities(self, idx: torch.Tensor, errors: torch.Tensor):
        for i, e in zip(idx, errors):
            p = self._calculate_priority(e.item())
            self.tree.update(i.item(), p)

    def _calculate_priority(self, error):
        return (error + 1e-5) ** self.alpha
