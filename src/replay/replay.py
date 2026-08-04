#!/usr/bin/env python3
"""
Short entry point — same as replay_client.py (dataset replay + aggregate table).

  python tools/replay/replay.py --backend cca --host <IP> --mat <path/to/s1.mat
"""
from replay_client import main

if __name__ == "__main__":
    raise SystemExit(main())
