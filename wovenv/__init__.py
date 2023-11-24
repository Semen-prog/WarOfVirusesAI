import configparser

_version = 1.0

parser = configparser.ConfigParser()
parser.read("./wovenv/conf.ini")

ROOT = parser['sys']['root']
MAX_TURN, N, M = int(parser['params']['max_turn']), int(parser['params']['n']), int(parser['params']['m'])

DPATH = f"{ROOT}source/best"

IN_PATH = f"{ROOT}wovenv/.cache/input"
OUT_PATH = f"{ROOT}wovenv/.cache/output"

coefs = (1, 5, -1, 0, 1, -1)