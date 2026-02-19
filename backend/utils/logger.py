import datetime

def log(message):
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")
