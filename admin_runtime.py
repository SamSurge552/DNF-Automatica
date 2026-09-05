"""检测/提升 Windows 管理员权限，用于全局按键钩子能钩到游戏进程。"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

_ENV_KEYS = (
    "PATH",
    "CUDA_PATH",
    "CUDA_HOME",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONNOUSERSITE",
)

ENV_FILE_NAME = ".elevate_env.json"


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _quote(arg: str) -> str:
    if not arg:
        return '""'
    if any(c in arg for c in ' \t"'):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def _project_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def suppress_windows_error_dialogs() -> None:
    """避免 cuDNN 入口点不匹配时弹出系统错误框（加载失败仍会走代码里的回退）。"""
    try:
        sem_failcritical = 0x0001
        sem_noopenfile = 0x8000
        ctypes.windll.kernel32.SetErrorMode(sem_failcritical | sem_noopenfile)
        new_mode = ctypes.c_uint()
        ctypes.windll.kernel32.SetThreadErrorMode(
            sem_failcritical | sem_noopenfile, ctypes.byref(new_mode)
        )
    except Exception:
        pass


def apply_saved_env() -> bool:
    """管理员新进程启动时恢复提权前的 PATH，必须在 import torch / ORT 之前调用。"""
    path = _project_dir() / ENV_FILE_NAME
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    finally:
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            try:
                path.unlink()
            except OSError:
                pass
        except OSError:
            pass

    cwd = data.get("cwd")
    if cwd and os.path.isdir(cwd):
        try:
            os.chdir(cwd)
        except OSError:
            pass
    for key, val in (data.get("env") or {}).items():
        if isinstance(key, str) and isinstance(val, str) and val:
            os.environ[key] = val
    return True


def _snapshot_env() -> dict[str, str]:
    out = {}
    for key in _ENV_KEYS:
        val = os.environ.get(key)
        if val:
            out[key] = val
    return out


def restart_as_admin(cwd: str | None = None) -> tuple[bool, str]:
    """
    弹出 UAC，用同一解释器/exe 重新启动，并带上当前 PATH（含 torch/cuDNN）。
    成功发起后由调用方退出本进程；用户取消 UAC 则返回 False。
    """
    if is_admin():
        return False, "当前已是管理员，无需重启。"

    workdir = str(Path(cwd or os.getcwd()).resolve())
    payload = {
        "cwd": workdir,
        "env": _snapshot_env(),
    }
    env_file = Path(workdir) / ENV_FILE_NAME
    try:
        env_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        return False, f"无法写入提权环境快照: {e}"

    frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        exe = sys.executable
        params = "--elevated"
    else:
        script = Path(sys.argv[0]).resolve()
        if not script.is_file():
            return False, f"找不到启动脚本: {script}"
        exe = sys.executable
        params = _quote(str(script))

    rc = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        exe,
        params,
        workdir,
        1,
    )
    if rc <= 32:
        try:
            env_file.unlink()
        except OSError:
            pass
        if int(rc) == 1223 or int(rc) == 0:
            return False, "已取消 UAC，仍以普通权限运行。"
        return False, f"提权启动失败（ShellExecute={rc}）。"
    return True, "已请求管理员进程，本窗口即将关闭。"
