#!venv/bin/python3
from subprocess import Popen, PIPE
from time import time
from click import command, argument, option
from click.types import Path, File
from collections import deque
import os


class TLPopen(Popen):
    cur_time = 0


class Field:
    def __init__(self, test):
        self.n, self.m, self.k = map(int, test.readline().strip().split())
        self.A = [list(map(int, line.strip().split())) for line in test]

    def post(self, popen, player_id=None):
        print(self.n, self.m, self.k, file=popen.stdin, flush=True)
        if player_id is not None:
            print(player_id, file=popen.stdin, flush=True)
        for line in self.A:
            print(*line, file=popen.stdin, flush=True)

    def make_move(self, move):
        for i in range(move[1]):
            y, x = move.pop() - 1, move.pop() - 1
            self.A[x][y] = move[0] if self.A[x][y] == 0 else -move[0]


@command(help='Gorilla interactor.')
@argument('player', type=Path(dir_okay=False, exists=True), nargs=-1)
@argument('val', type=Path(dir_okay=False, exists=True))
@argument('test', type=File())
@option('--time-limit', '-t', type=int, help='Time limit for each player (seconds).')
@option('--verbose', '-v', is_flag=True, help='Verbose for validator.')
class Gorill:
    def __init__(self, player, val, test, time_limit, verbose):
        self.field = Field(test)
        self.k = self.field.k
        self.check_folded = [False for _ in range(self.k)]
        self.time_limit = time_limit
        popen_pref = [] if time_limit is None else ['timeout', str(time_limit * (self.k + 1))]
        val_suf = ['-v'] if verbose else []
        assert (self.k == len(player))
        self.popen = [TLPopen(popen_pref + [os.path.join('.', p)], universal_newlines=True, stdin=PIPE, stdout=PIPE)
                      for p in player]
        self.val = Popen([os.path.join('.', val)] + val_suf, universal_newlines=True, stdin=PIPE, stdout=PIPE)
        self.mainloop()

    def endgame(self, player_id):
        print(f'Player {player_id} wins')
        for p in range(self.k):
            if not self.check_folded[p]:
                self.popen[p].terminate()
        self.val.terminate()
        exit(0)

    def make_moves(self, popen, player_id):
        if not self.check_folded[player_id - 1]:
            g = time()
            arr = popen.stdout.readline().strip().split()
            popen.cur_time += time() - g
            try:
                if self.time_limit is not None and popen.cur_time > self.time_limit:
                    print('TL')
                    raise Exception
                arr = list(map(int, arr))
                if arr[0] != player_id or 3 < arr[1] or arr[1] < 0:
                    raise ValueError
                if len(arr[2:]) != arr[1] * 2:
                    raise ValueError
            except (Exception, ValueError, IndexError):
                print('Super incorrect move (or TL)')
                # check-fold him
                if not self.check_folded[player_id - 1]:
                    self.check_folded[player_id - 1] = True
                    self.popen[player_id - 1].terminate()
                arr = [player_id, 0]
            print(*arr, file=self.val.stdin, flush=True)
            status = list(map(int, self.val.stdout.readline().strip().split()))
            if status[0] == 0:
                # everything nominal
                return arr
            elif status[0] == -1:
                # bad move, check-fold him
                if not self.check_folded[player_id - 1]:
                    self.check_folded[player_id - 1] = True
                    self.popen[player_id - 1].terminate()
                return [player_id, 0]
            else:
                # someone has won
                self.endgame(status[0])
        else:
            arr = [player_id, 0]
            print(*arr, file=self.val.stdin, flush=True)
            status = list(map(int, self.val.stdout.readline().strip().split()))
            if status[0] == -1:
                # everything nominal
                # YES IT IS
                return arr
            else:
                # someone has won
                self.endgame(status[0])

    def mainloop(self):
        moves = deque()
        self.field.post(self.val)
        for p in range(self.k):
            self.field.post(self.popen[p], p + 1)
            moves.append(self.make_moves(self.popen[p], p + 1))
        p = 0
        while True:
            moves.popleft() # discarding previous move by this player
            if not self.check_folded[p]:
                print('r', file=self.popen[p].stdin, flush=True)
                for move in moves:
                    print(*move, file=self.popen[p].stdin, flush=True)
            moves.append(self.make_moves(self.popen[p], p + 1))
            p += 1
            if p == self.k:
                p = 0


if __name__ == '__main__':
    Gorill()
