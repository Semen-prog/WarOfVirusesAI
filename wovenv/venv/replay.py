import torch
from wovenv.venv.snapshot import SnapShot, Action
from torchrl.data.replay_buffers import PrioritizedReplayBuffer, ListStorage

class Replay:
    def __init__(self, size: int) -> None:
        self.data = dict()
        self.buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4, storage=ListStorage(size))

    def len(self) -> int:
        return len(self.buffer)
    
    def add(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool) -> None:
        index = self.buffer.add(hash((s, a)))
        self.data[index] = (s, a, ns, r, d)

    def sample(self, sample_size: int) -> tuple[list[tuple], torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        result, info = self.buffer.sample(sample_size, return_info=True)
        return ([self.data[r.item()] for r in info['index']], info['index'].to(device), info['_weight'].to(device))
    
    def update_priority(self, idx, prs) -> None:
        self.buffer.update_priority(idx, prs)