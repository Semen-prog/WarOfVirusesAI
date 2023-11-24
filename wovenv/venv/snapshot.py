from wovenv.venv.state import *
from wovenv import N, M, MAX_TURN, coefs

class Action:

    def __init__(self, i: int, j: int, c: bool) -> None:
        
        self.i = i
        self.j = j
        self.change = c

    def __hash__(self) -> int:
        return hash((self.i, self.j,  self.change))
    
    def __eq__(self, __value: object) -> bool:
        return hasattr(__value, "i") and hasattr(__value, "j") and hasattr(__value, "change") and \
                self.i == __value.i and self.j == __value.j and self.change == __value.change

class SnapShot:

    def __init__(self, table: list[list[State]], turn: int) -> None:

        self.table = tuple(map(tuple, table))
        self.turn = turn

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
                if self.table[i][j] == State.RED_CROSS and not self._used[i][j]:
                    self._find_access(i, j, State.RED_CROSS, State.RED_TOWER)

        answer = []

        for i in range(n):
            for j in range(m):
                if self._used[i][j] and (self.table[i][j] == State.BLUE_CROSS or self.table[i][j] == State.EMPTY):
                    if self.turn != MAX_TURN: answer.append(Action(i, j, False))
                    answer.append(Action(i, j, True))
        
        return answer
    
    def get_opponents_actions(self) -> list[Action]:

        self._clear()
        n, m = self.shape()

        for i in range(n):
            for j in range(m):
                if self.table[i][j] == State.BLUE_CROSS and not self._used[i][j]:
                    self._find_access(i, j, State.BLUE_CROSS, State.BLUE_TOWER)

        answer = []

        for i in range(n):
            for j in range(m):
                if self._used[i][j] and (self.table[i][j] == State.RED_CROSS):
                    if self.turn != MAX_TURN: answer.append(Action(i, j, False))
                    answer.append(Action(i, j, True))
        
        return answer
    
    def finished(self) -> bool:

        return len(self.get_legal_actions()) == 0

    def shape(self) -> tuple[int, int]:

        return (len(self.table), len(self.table[0]))
    
    def score(self) -> int:

        cnt = [0, 0, 0, 0, len(self.get_legal_actions()), len(self.get_opponents_actions())]
        n, m = self.shape()

        for i in range(n):
            for j in range(m):
                if self.table[i][j].value < 4: cnt[self.table[i][j].value] += 1

        ans = 0
        for s in State:
            ans += cnt[s.value] * coefs[s.value]

        return ans

    def unroll(self) -> list[int]:

        res = []

        n, m = self.shape()
        for i in range(n):
            for j in range(m):
                res.append(self.table[i][j].value)

        res.append(self.turn)

        return res
    
    def __eq__(self, __value: object) -> bool:
        
        return hasattr(__value, "table") and hasattr(__value, "turn") and self.table == __value.table and self.turn == __value.turn
    
    def __hash__(self) -> int:
        
        return hash((self.table, self.turn))