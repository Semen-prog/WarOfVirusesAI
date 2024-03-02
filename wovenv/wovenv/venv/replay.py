import torch, numpy, random
from wovenv.venv.snapshot import SnapShot, Action
from wovenv import MAX_TURN
from collections import deque

class Replay(object):
    def __init__(self, size: int) -> None:
        self.rl = deque(maxlen=size)
        self.size = size

    def sample(self, sample_size):
        return random.sample(list(self.rl), min(len(self.rl), sample_size))

    def add(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool) -> None:
        self.rl.append((s, a, ns, r, d))

    