import os
from sys import exit
from wovenv.venv.state import *
from wovenv import N, M, MAX_TURN, coefs, DPATH, IN_PATH, OUT_PATH
from wovenv.venv.snapshot import SnapShot, Action
from wovenv.venv.utils import write_error, write_access, clear_access

class Env:

    def __init__(self, write_log=True) -> None:
        
        self.write_log=write_log
        self.reset()

    def _clear(self) -> None:

        n, m = self.shape()
        self._used = [[False for _ in range(m)] for _ in range(n)]

    def reset(self) -> SnapShot:

        self.turn = 1
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
    
    def _write_action(self, a: Action, player: int):
        if self.table[a.i][a.j] == State.EMPTY:
            write_access(f"{a.i} {a.j} {player}\n")
        else:
            write_access(f"{a.i} {a.j} {-player}\n")

    def finish_log(self):
        write_access("-1 -1 -1")
    
    def _reinit(self, s: SnapShot) -> None:

        self.table = list(map(list, s.table))
        self.turn = s.turn

    def get_snapshot(self) -> SnapShot:

        return SnapShot(self.table, self.turn)

    def shape(self) -> tuple[int, int]:

        return (len(self.table), len(self.table[0]))
    
    def reverse(self):

        self.turn = 1
        n, m = self.shape()

        for i in range(n):
            for j in range(m):
                if self.table[i][j] == State.RED_CROSS: self.table[i][j] = State.BLUE_CROSS
                elif self.table[i][j] == State.RED_TOWER: self.table[i][j] = State.BLUE_TOWER
                elif self.table[i][j] == State.BLUE_CROSS: self.table[i][j] = State.RED_CROSS
                elif self.table[i][j] == State.BLUE_TOWER: self.table[i][j] = State.RED_TOWER

    def _check_access(self, i, j) -> bool:

        self._used[i][j] = True
        if self.table[i][j] == State.BLUE_CROSS: return True

        n, m = self.shape()

        for di in range(-1, 2):
            for dj in range(-1, 2):
                ip, jp = i + di, j + dj
                if 0 <= ip < n and 0 <= jp < m and not self._used[ip][jp] \
                    and (self.table[ip][jp] == State.BLUE_CROSS or self.table[ip][jp] == State.BLUE_TOWER) \
                    and self._check_access(ip, jp):
                    return True
                
        return False
    
    def _get_string_table(self) -> str:

        n, m = self.shape()
        tab = ""
        for i in range(n):
            for j in range(m):
                tab += str(self.table[i][j].value)
                tab += " "
            tab += "\n"

        return tab


    def _make_turn(self, a: Action, player: int) -> None:

        self._clear()

        n, m = self.shape()
        if not (0 <= a.i < n and 0 <= a.j < m and (self.table[a.i][a.j] == State.RED_CROSS or self.table[a.i][a.j] == State.EMPTY)):
            
            tab = self._get_string_table()
            write_error(f"Error: incorrect cell was provided: i = {a.i}, j = {a.j}, table:\n{tab}")

            return
        
        if not a.change and self.turn == MAX_TURN:

            tab = self._get_string_table()
            write_error(f"Error: violation of MAX_TURN rule: table:\n{tab}")

            return
        
        if not self._check_access(a.i, a.j):

            tab = self._get_string_table()
            write_error(f"Error: cannot make turn: i = {a.i}, j = {a.j}, table:\n{tab}")

            return
        
        if self.write_log: self._write_action(a, player)
        
        self.table[a.i][a.j] = State.BLUE_CROSS if self.table[a.i][a.j] == State.EMPTY else State.BLUE_TOWER
        self.turn += 1

    def _get_actions(self) -> list[Action]:

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
        for e in range(k):
            i, j = map(int, ouf.readline().split())
            i -= 1
            j -= 1
            turns.append(Action(i, j, e == k - 1))

        ouf.close()
        return turns

    def _skip(self) -> None:
        self.reverse()
        self.turn = -float('inf')
        while not self.get_snapshot().finished():
            action_list = self._get_actions()
            for ar in action_list:
                ar.change = False
                self._make_turn(ar, 2)
        self.reverse()
    
    def step(self, a: Action) -> tuple[SnapShot, int, bool]:

        s = self.get_snapshot()

        start = s.score()
        self._make_turn(a, 1)

        if not a.change:
            if self.get_snapshot().finished():
                self._skip()
            sn = self.get_snapshot()
            return (sn, sn.score() - start, sn.finished())
        
        self.reverse()

        action_list = self._get_actions()
        for ar in action_list:
            self._make_turn(ar, 2)

        self.reverse()
        
        if self.get_snapshot().finished():
            self._skip()

        sn = self.get_snapshot()
        return (sn, sn.score() - start, sn.finished())
    
    def get_result(self, s: SnapShot, a: Action) -> tuple[SnapShot, int, bool]:

        ts = self.get_snapshot()
        self._reinit(s)

        next_s, r, done = self.step(a)
        self._reinit(ts)

        return (next_s, r, done)
