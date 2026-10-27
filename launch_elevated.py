"""未管理员时快照 PATH 并 UAC 重开目标脚本。给 start.bat / start_binds.bat 用。"""
from __future__ import annotations

import sys

from admin_runtime import restart_as_admin

if __name__ == "__main__":
    script = sys.argv[1] if len(sys.argv) > 1 else None
    ok, msg = restart_as_admin(script=script)
    print(msg)
    raise SystemExit(0 if ok else 1)
