import torch
from wovenv.venv.snapshot import SnapShot, Action
from torchrl.data.replay_buffers import PrioritizedReplayBuffer, ListStorage

class Replay:
    def __init__(self, size: int) -> None:
        self.data = dict()
        self.buffer = PrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=ListStorage(size))

    def len(self) -> int:
        return len(self.buffer)
    
    def add(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool, p: float) -> None:
        index = self.buffer.add(1)
        self.data[index] = (s, a, ns, r, d)
        self.buffer.update_priority(index, p)

    def sample(self, sample_size: int) -> tuple[list[tuple], torch.Tensor]:
        result, info = self.buffer.sample(sample_size, return_info=True)
        return ([self.data[r.item()] for r in info['index']], info['index'])
    
    def update_priority(self, prs, idx) -> None:
        self.buffer.update_priority(idx, prs)