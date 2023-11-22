from datetime import datetime
from sys import exit

def get_time() -> str:
    return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

def clear_access():
    open("/home/semen/Development/ml/wovenv/log/access.log", "w").close()

def write_error(error: str):
    f = open("/home/semen/Development/ml/wovenv/log/error.log", "+a")
    f.write(f"\n{get_time()}: {error}")
    f.close()

    exit(0)

def write_access(data: str):
    f = open("/home/semen/Development/ml/wovenv/log/access.log", "+a")
    f.write(data)
    f.close()