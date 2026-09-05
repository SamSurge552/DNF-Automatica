"""启动前准备 CUDA 运行库：Torch(cu128) 与 RapidOCR/ORT(CUDA13) 共存。"""
import os
import site
from pathlib import Path


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

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return False

    x = torch.randn(1, 3, 64, 64, device="cuda")
    w = torch.randn(8, 3, 3, 3, device="cuda")
    F.conv2d(x, w)
    return True


def ensure_cuda_runtime():
    """注入 nvidia pip 包 DLL 路径，并预加载 onnxruntime CUDA/cuDNN。"""
    try:
        ensure_torch_cuda()
    except Exception:
        pass

    try:
        import onnxruntime as ort

        # directory="" 表示从 site-packages/nvidia 搜索 CUDA 13 / cuDNN DLL
        ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
    except Exception:
        pass
