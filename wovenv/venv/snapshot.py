import torch, math
from wovenv.venv.state import *
from wovenv import N, M, MAX_TURN, coefs

class Action:

    def __init__(self, i: int, j: int) -> None:
        
        self.i = i
        self.j = j

    def to_index(self) -> int:
        return (self.i * M + self.j)

    def __hash__(self) -> int:
        return hash((self.i, self.j))
    
    def __eq__(self, __value: object) -> bool:
        return hasattr(__value, "i") and hasattr(__value, "j") and \
                self.i == __value.i and self.j == __value.j

class SnapShot:

    def __init__(self, table: list[list[State]], cached: list[Action]) -> None:

        self.table = tuple(map(tuple, table))
        self.cached = tuple(cached)
    
    def turn(self):
        return len(self.cached)

    def _clear(self):

        n, m = self.shape()
        self._used = [[False for _ in range(m)] for _ in range(n)]

    def _find_access(self, i, j, state, state_p):

        self._used[i][j] = True
        if self.table[i][j] != state and self.table[i][j] != state_p: return

        n, m = self.shape()

        for di in range(-1, 2):
            for dj in range(-1, 2):
                ip, jp = i + di, j + dj
                if 0 <= ip < n and 0 <= jp < m and not self._used[ip][jp]:
                    self._find_access(ip, jp, state, state_p)

    def get_legal_actions(self) -> list[Action]:

        self._clear()
        n, m = self.shape()

        for i in range(n):
            for j in range(m):
                if self.table[i][j] == State.BLUE_CROSS and not self._used[i][j]:
                    self._find_access(i, j, State.BLUE_CROSS, State.BLUE_TOWER)

        answer = []

        for i in range(n):
            for j in range(m):
                if self._used[i][j] and (self.table[i][j] == State.RED_CROSS or self.table[i][j] == State.EMPTY):
                    answer.append(Action(i, j))
        
        return answer
    
    def finished(self) -> bool:

        return len(self.get_legal_actions()) == 0

    def shape(self) -> tuple[int, int]:

        return (len(self.table), len(self.table[0]))
    
    def score(self) -> float:

        cnt = [0, 0, 0, 0]
        self._clear()
        n, m = self.shape()

        for i in range(n):
            for j in range(m):
                if self.table[i][j] == State.BLUE_CROSS and not self._used[i][j]:
                    self._find_access(i, j, State.BLUE_CROSS, State.BLUE_TOWER)

        for i in range(n):
            for j in range(m):
                if self.table[i][j] == State.BLUE_CROSS: cnt[0] += 1
                if self.table[i][j] == State.BLUE_TOWER: cnt[1] += 1
                if self.table[i][j] == State.RED_CROSS: cnt[2] += 1
                if self.table[i][j] == State.RED_TOWER: cnt[3] += 1

        ans = cnt[0] * coefs[0] + coefs[1] * cnt[1] - cnt[2] * coefs[2] - coefs[3] * cnt[3]

        return ans
    
    def to_tensor(self) -> torch.Tensor:
        res = torch.zeros(6, N, M)
        for i in range(N):
            for j in range(M):
                res[self.table[i][j].value][i][j] = 1.
        for a in self.cached:
            res[5][a.i][a.j] = 1
        return res
    
    def __eq__(self, __value: object) -> bool:
        
        return hasattr(__value, "table") and hasattr(__value, "cached") and self.table == __value.table and self.cached == __value.cached
    
    def __hash__(self) -> int:
        
        return hash((self.table, self.cached))
    
def form_action(index: int) -> Action:
    return Action(index // M, index % M)

def form_states(data: list[SnapShot]) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res    = torch.autograd.Variable(torch.stack([s.to_tensor().to(device) for s in data])).to(device)
    lines  = torch.diag(torch.ones(MAX_TURN))
    resp   = torch.autograd.Variable(torch.stack([lines[s.turn()].to(device) for s in data])).to(device)
    return res, resp
