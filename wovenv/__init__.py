import torch
_version = 1.0

ROOT = '/home/semen/Development/WarOfVirusesAI/'

SAMPLE_SIZE = 100
BATCH_SIZE = 2 ** 14
MAX_TURN = 3
N = 10
M = 10

DPATH = f"{ROOT}source/best"

IN_PATH = f"{ROOT}wovenv/cache/input"
OUT_PATH = f"{ROOT}wovenv/cache/output"

coefs = torch.tensor([1., 1., -1., -1., 0., 0., 0.])

# (my_cross, my_tower, op_cross, op_tower, empty, turn, my_act)
