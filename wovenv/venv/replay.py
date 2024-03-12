import random
from wovenv.venv.snapshot import SnapShot, Action

class Replay:
    def __init__(self, size: int) -> None:
        self.buffer, self.priority = [], []
        self.size = size

    def len(self) -> int:
        return len(self.buffer)

    def pop(self):
        index = min(range(self.len()), key=lambda x: self.priority[x])
        self.buffer.pop(index)
        self.priority.pop(index)

    def add(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool, p: float) -> None:
        self.buffer.append((s, a, ns, r, d))
        self.priority.append(p)
        if self.len() > self.size:
            self.pop()

    def sample(self, batch_size: int):
        values = [random.uniform(0, 1) for _ in range(batch_size)]
        sls, idx = [], []
        values.sort()
        ind, tmp, sm = 0, 0, sum(self.priority)
        for i in range(self.len()):
            tmp += self.priority[i] / sm
            while ind < batch_size and values[ind] <= tmp + 0.0001:
                idx.append(i)
                sls.append(self.buffer[i])
                ind += 1
        return sls, idx
    
    def update_priority(self, prs, idx):
        for i, p in zip(idx, prs):
            self.priority[i] = p