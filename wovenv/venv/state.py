from enum import Enum

class State(Enum):

    RED_CROSS = 0
    RED_TOWER = 1
    BLUE_CROSS = 2
    BLUE_TOWER = 3
    EMPTY = 4
    STONE = 5

MAX_TURN, N, M = 3, 10, 10

DPATH = "/home/semen/Development/ml/source/best"

IN_PATH = "/home/semen/Development/ml/wovenv/.cache/input"
OUT_PATH = "/home/semen/Development/ml/wovenv/.cache/output"

coefs = (1, 5, -1, 0, 1, -1)