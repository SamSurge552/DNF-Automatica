"""启动前准备 CUDA 运行库：Torch(cu128) 与 RapidOCR/ORT(CUDA13) 共存。"""
import os
import site
from pathlib import Path

_ENSURE_STATUS = None  # "default ok" | "fallback cuda12.6" | "both failed"

_CUDA126_DEFAULT_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")


def _site_package_dirs():
    dirs = []
    try:
        dirs.extend(Path(sp) for sp in site.getsitepackages())
    except Exception:
        pass
    try:
        import sys

        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                dirs.append(Path(meipass))
            dirs.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    return dirs


def _torch_lib_dir():
    for sp in _site_package_dirs():
        lib = sp / "torch" / "lib"
        if lib.is_dir():
            return lib
    return None


def _nvidia_bin_dirs():
    paths = []
    for sp in _site_package_dirs():
        root = sp / "nvidia"
        if not root.exists():
            continue
        for p in root.glob("*/bin"):
            paths.append(str(p))
        for p in root.glob("*/lib"):
            paths.append(str(p))
    return paths


def _cuda126_bin_dirs():
    """仅回退用：系统 CUDA 12.6 bin + pip nvidia 包里带 cu12/12.6 的 bin/lib。"""
    out = []
    seen = set()

    def _add(p):
        if p is None:
            return
        path = Path(p)
        if not path.is_dir():
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    for var in ("CUDA_PATH", "CUDA_PATH_V12_6"):
        raw = (os.environ.get(var) or "").strip()
        if raw:
            _add(Path(raw) / "bin")
    _add(_CUDA126_DEFAULT_BIN)

    for sp in _site_package_dirs():
        root = sp / "nvidia"
        if not root.is_dir():
            continue
        for p in list(root.glob("*/bin")) + list(root.glob("*/lib")):
            s = str(p).replace("\\", "/").lower()
            if "cu12" in s or "12.6" in s:
                _add(p)
        try:
            kids = list(root.iterdir())
        except OSError:
            kids = []
        for kid in kids:
            if not kid.is_dir():
                continue
            name = kid.name.lower()
            if "cu12" not in name and "12.6" not in name:
                continue
            _add(kid / "bin")
            _add(kid / "lib")
    return out


def _prepend_dll_dirs(dirs):
    seen = []
    for d in dirs:
        if not d or d in seen:
            continue
        seen.append(d)
        if not os.path.isdir(d):
            continue
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
    if not seen:
        return
    path = os.environ.get("PATH", "")
    prefix = os.pathsep.join(seen)
    parts = [p for p in path.split(os.pathsep) if p and p not in seen]
    os.environ["PATH"] = prefix + os.pathsep + os.pathsep.join(parts)


def _try_cudnn_conv():
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return False
    x = torch.randn(1, 3, 64, 64, device="cuda")
    w = torch.randn(8, 3, 3, 3, device="cuda")
    F.conv2d(x, w)
    return True


def _try_ort_preload():
    import onnxruntime as ort

    # directory="" 表示从 site-packages/nvidia 搜索 CUDA 13 / cuDNN DLL
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
    return True


def _emit(log, kind: str) -> None:
    line = f"cuda_runtime: {kind}"
    if log is not None:
        try:
            log(line)
            return
        except Exception:
            pass
    print(line)


def ensure_torch_cuda():
    """
    必须在 onnxruntime.preload_dlls 之前调用，并真正跑一次 cuDNN 卷积。
    仅 torch.zeros 不够：ORT 预加载 CUDA13 cuDNN 后，YOLO 再加载 cudnn_cnn
    会绑到错误依赖，推理置信度会塌成 ~0.001（表现为一直「无目标」）。
    """
    torch_lib = _torch_lib_dir()
    nvidia_dirs = _nvidia_bin_dirs()
    # Torch 自带 cuDNN 必须排在系统 CUDA / nvidia-cudnn-cu13 前面
    ordered = []
    if torch_lib is not None:
        ordered.append(str(torch_lib))
    ordered.extend(nvidia_dirs)
    _prepend_dll_dirs(ordered)
    return _try_cudnn_conv()


def ensure_cuda_runtime(log=None):
    """注入 nvidia pip 包 DLL 路径，并预加载 onnxruntime CUDA/cuDNN。"""
    global _ENSURE_STATUS
    if _ENSURE_STATUS is not None:
        return

    torch_ok = False
    try:
        torch_ok = bool(ensure_torch_cuda())
    except Exception:
        torch_ok = False

    ort_ok = False
    try:
        ort_ok = bool(_try_ort_preload())
    except Exception:
        ort_ok = False

    if torch_ok and ort_ok:
        _ENSURE_STATUS = "default ok"
        _emit(log, "default ok")
        return

    _prepend_dll_dirs(_cuda126_bin_dirs())
    try:
        torch_ok = bool(_try_cudnn_conv())
    except Exception:
        torch_ok = False
    try:
        ort_ok = bool(_try_ort_preload())
    except Exception:
        ort_ok = False

    if torch_ok and ort_ok:
        _ENSURE_STATUS = "fallback cuda12.6"
        _emit(log, "fallback cuda12.6")
        return

    _ENSURE_STATUS = "both failed"
    _emit(log, "both failed")
