import configparser

_version = 1.0

parser = configparser.ConfigParser()
parser.read("./wovenv/conf.ini")

ROOT = parser['sys']['root']
SAMPLE_SIZE, BATCH_SIZE, MAX_TURN, N, M = int(parser['params']['sample_size']), int(parser['params']['batch_size']), int(parser['params']['max_turn']), int(parser['params']['n']), int(parser['params']['m'])

DPATH = f"{ROOT}source/best"

IN_PATH = f"{ROOT}wovenv/.cache/input"
OUT_PATH = f"{ROOT}wovenv/.cache/output"

coefs = (0, 1, 0, 0, 0, 0)

# (my_cross, my_tower, op_cross, op_tower, my_act, op_act)