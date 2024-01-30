import torch
from wovenv.venv.snapshot import SnapShot, Action
from wovenv import MAX_TURN
from collections import deque

class Replay:

    def __init__(self, size: int) -> None:
        self.rl = deque()
        self.size = size

    def add(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool) -> None:
        self.rl.append((s, a, ns, r, d))
        if len(self.rl) > self.size:
            self.rl.popleft()

    