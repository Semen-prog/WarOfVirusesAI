from datetime import datetime
from sys import exit
from wovenv import ROOT

def get_time() -> str:
    return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

def clear_access():
    open(f"{ROOT}wovenv/log/access.log", "w").close()

def write_error(error: str):
    f = open(f"{ROOT}wovenv/log/error.log", "+a")
    f.write(f"\n{get_time()}: {error}")
    f.close()

    exit(0)

def write_access(data: str):
    f = open(f"{ROOT}wovenv/log/access.log", "+a")
    f.write(data)
    f.close()