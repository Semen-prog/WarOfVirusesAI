import os, torch
from wovenv.venv.state import *
from wovenv import N, M, MAX_TURN, coefs, DPATH, IN_PATH, OUT_PATH
from wovenv.venv.utils import write_error, write_access, clear_access

def score(snap: torch.Tensor) -> float:
    return (snap.sum(1).sum(1) @ coefs).item()

class Env:
    def __init__(self, write_log=True) -> None:
        self.write_log=write_log
        self.reset()

    def form_index(self, i: int, j: int) -> int:
        return i * M + j
    
    def form_action(self, index: int) -> tuple[int, int]:
        return index // M, index % M

    def _clear(self) -> None:
        n, m = self.shape()
        self._used = [[False for _ in range(m)] for _ in range(n)]

    def reset(self) -> torch.Tensor:
        self.cached = []
        self.table = [[ State.EMPTY for _ in range(M) ] for _ in range(N) ]
        self.table[0][0] = State.BLUE_CROSS
        self.table[N - 1][M - 1] = State.RED_CROSS
        if self.write_log:
            clear_access()
            write_access(f"{N} {M} 2\n")
            for i in range(N):
                for j in range(M):
                    if i == 0 and j == 0: write_access("1 ")
                    elif i == N - 1 and j == M - 1: write_access("2 ")
                    else: write_access("0 ")
                write_access("\n")

        return self.get_snapshot()

    def reinit(self, snap: torch.Tensor):
        tab = snap[:-1].argmax(dim=0)
        self.cached = []
        self.table = [[ State.EMPTY for _ in range(M) ] for _ in range(N) ]
        for i in range(N):
            for j in range(M):
                self.table[i][j] = State(tab[i][j])
                if snap[-1][i][j] > 0:
                    self.cached.append((i, j))
    
    def turn(self):
        return len(self.cached)
    
    def _write_action(self, i: int, j: int, player: int):
        if not self.write_log: return
        if self.table[i][j] == State.EMPTY:
            write_access(f"{i} {j} {player}\n")
        else:
            write_access(f"{i} {j} {-player}\n")

    def finish_log(self):
        if self.write_log: write_access("-1 -1 -1")

    def get_snapshot(self) -> torch.Tensor:
        snap = torch.zeros(7, N, M)
        for i in range(N):
            for j in range(M):
                snap[self.table[i][j].value][i][j] = 1.
        for i, j in self.cached:
            snap[5][i][j] = 1.
        snap[6] = self.get_legal_actions()
        return snap

    def shape(self) -> tuple[int, int]:
        return (len(self.table), len(self.table[0]))
    
    def _change_player(self):
        self.cached = []
        n, m = self.shape()
        for i in range(n):
            for j in range(m):
                if self.table[i][j] == State.RED_CROSS: self.table[i][j] = State.BLUE_CROSS
                elif self.table[i][j] == State.RED_TOWER: self.table[i][j] = State.BLUE_TOWER
                elif self.table[i][j] == State.BLUE_CROSS: self.table[i][j] = State.RED_CROSS
                elif self.table[i][j] == State.BLUE_TOWER: self.table[i][j] = State.RED_TOWER

    def _find_access(self, i, j, state, state_p):
        self._used[i][j] = True
        if self.table[i][j] != state and self.table[i][j] != state_p: return
        n, m = self.shape()
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ip, jp = i + di, j + dj
                if 0 <= ip < n and 0 <= jp < m and not self._used[ip][jp]:
                    self._find_access(ip, jp, state, state_p)

    def _fill_access(self, state, statep):
        n, m = self.shape()
        self._clear()
        for i in range(n):
            for j in range(m):
                if self.table[i][j] == state and not self._used[i][j]:
                    self._find_access(i, j, state, statep)

    def get_legal_actions(self) -> torch.Tensor:
        n, m = self.shape()
        self._fill_access(State.BLUE_CROSS, State.BLUE_TOWER)
        result = torch.zeros(n, m)
        for i in range(n):
            for j in range(m):
                if self._used[i][j] and (self.table[i][j] == State.EMPTY or self.table[i][j] == State.RED_CROSS):
                    result[i][j] = 1
        return result
    
    def finished(self) -> bool:
        return self.get_legal_actions().sum() == 0
    
    def _get_string_table(self) -> str:
        n, m = self.shape()
        tab = ""
        for i in range(n):
            for j in range(m):
                tab += str(self.table[i][j].value)
                tab += " "
            tab += "\n"
        return tab

    def _make_turn(self, i: int, j: int, player: int) -> None:
        self._clear()
        n, m = self.shape()
        if not (0 <= i < n and 0 <= j < m and (self.table[i][j] == State.RED_CROSS or self.table[i][j] == State.EMPTY)):
            tab = self._get_string_table()
            write_error(f"Error: incorrect cell was provided: i = {i}, j = {j}, table:\n{tab}")
            return
        self._fill_access(State.BLUE_CROSS, State.BLUE_TOWER)
        if not self._used[i][j]:
            tab = self._get_string_table()
            write_error(f"Error: cannot make turn: i = {i}, j = {j}, table:\n{tab}")
            return
        self._write_action(i, j, player)
        
        self.table[i][j] = State.BLUE_CROSS if self.table[i][j] == State.EMPTY else State.BLUE_TOWER
        self.cached.append((i, j))

    def _get_actions(self) -> list[tuple[int, int]]:
        if DPATH == "":
            write_error(f"Error: DPATH == ''")
        n, m = self.shape()
        inf = open(IN_PATH, "w")
        inf.write(f"{n} {m}\n")
        for i in range(n):
            for j in range(m):
                inf.write(f"{self.table[i][j].value} ")
            inf.write("\n")
        inf.close()
        os.system(f"{DPATH} < {IN_PATH} > {OUT_PATH}")
        ouf = open(OUT_PATH, "r")
        turns = []
        k = int(ouf.readline().split()[0])
        for _ in range(k):
            i, j = map(int, ouf.readline().split())
            i -= 1
            j -= 1
            turns.append((i, j))
        ouf.close()
        return turns

    def _skip(self) -> None:
        self._change_player()
        while not self.finished():
            self.cached = []
            action_list = self._get_actions()
            for i, j in action_list:
                self._make_turn(i, j, 2)
        self._change_player()
    
    def step(self, a: int) -> tuple[torch.Tensor, int, bool]:
        st = self.get_snapshot()
        _i, _j = self.form_action(a)
        self._make_turn(_i, _j, 1)
        if self.turn() == MAX_TURN:
            self._change_player()
            action_list = self._get_actions()
            for i, j in action_list:
                self._make_turn(i, j, 2)
            self._change_player()
        if self.finished():
            self._skip()
        sn = self.get_snapshot()
        return (sn, score(sn) - score(st), self.finished())