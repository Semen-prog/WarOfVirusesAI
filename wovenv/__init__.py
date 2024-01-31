import configparser

_version = 1.0

parser = configparser.ConfigParser()
parser.read("./wovenv/conf.ini")

ROOT = parser['sys']['root']
BATCH_SIZE, MAX_TURN, N, M = int(parser['params']['batch_size']), int(parser['params']['max_turn']), int(parser['params']['n']), int(parser['params']['m'])

DPATH = f"{ROOT}source/best"

IN_PATH = f"{ROOT}wovenv/.cache/input"
OUT_PATH = f"{ROOT}wovenv/.cache/output"

coefs = (1, 2, -1, -1, 0, 0)