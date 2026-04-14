"""
Mutual exclusion lock for Zeno/Kronos — only one bot may run at a time.
"""
from __future__ import annotations

import os
import sys

LOCK_FILE = os.path.join(os.path.dirname(__file__), "..", "bot.lock")


def acquire(bot_name: str, force: bool = False) -> None:
    """Write a lock file with this bot's name and PID. Exit if another bot is running."""
    if force:
        return
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            contents = f.read().strip().splitlines()
        if len(contents) == 2:
            running_name, pid_str = contents
            try:
                pid = int(pid_str)
                os.kill(pid, 0)  # raises if process is dead
                print(f"[LOCK] {running_name} is already running (PID {pid}). Stop it first.")
                sys.exit(1)
            except (ProcessLookupError, PermissionError):
                pass  # stale lock — overwrite it

    with open(LOCK_FILE, "w") as f:
        f.write(f"{bot_name}\n{os.getpid()}\n")


def release() -> None:
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass
