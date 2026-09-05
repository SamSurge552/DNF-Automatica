"""检测 Windows HDR，并把 scRGB 线性帧映射成 8bit sRGB（OBS 同款）。"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt

_user32 = ctypes.windll.user32
_QDC_ONLY_ACTIVE_PATHS = 0x00000002
_DINFO_GET_ADVANCED_COLOR = 9
_DINFO_GET_SDR_WHITE_LEVEL = 11
_DINFO_GET_ADVANCED_COLOR_INFO_2 = 15
_ERROR_SUCCESS = 0

# DISPLAYCONFIG_ADVANCED_COLOR_MODE
_COLOR_MODE_HDR = 2
# INFO2.value bit5 = highDynamicRangeUserEnabled
_INFO2_HDR_USER = 0x20

# scRGB 里 1.0 = 80 nits
_SCRGB_WHITE_NITS = 80.0
_DEFAULT_SDR_WHITE_NITS = 203.0


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]


class _DCDI_HEADER(ctypes.Structure):
    _fields_ = [
        ("type", wt.UINT),
        ("size", wt.UINT),
        ("adapterId", _LUID),
        ("id", wt.UINT),
    ]


class _ADV_COLOR_INFO(ctypes.Structure):
    _fields_ = [
        ("header", _DCDI_HEADER),
        ("value", wt.UINT),
        ("colorEncoding", wt.UINT),
        ("bitsPerColorChannel", wt.UINT),
    ]


class _SDR_WHITE(ctypes.Structure):
    _fields_ = [
        ("header", _DCDI_HEADER),
        ("SDRWhiteLevel", wt.ULONG),
    ]


class _ADV_COLOR_INFO_2(ctypes.Structure):
    """Win11：能区分 HDR 开关和 ACM/WCG（后者也会把 type9 的 enabled 置位）。"""

    _fields_ = [
        ("header", _DCDI_HEADER),
        ("value", wt.UINT),
        ("colorEncoding", wt.UINT),
        ("bitsPerColorChannel", wt.UINT),
        ("activeColorMode", wt.UINT),
    ]


class _PATH_SOURCE(ctypes.Structure):
    _fields_ = [
        ("adapterId", _LUID),
        ("id", wt.UINT),
        ("modeInfoIdx", wt.UINT),
        ("statusFlags", wt.UINT),
    ]


class _PATH_TARGET(ctypes.Structure):
    _fields_ = [
        ("adapterId", _LUID),
        ("id", wt.UINT),
        ("modeInfoIdx", wt.UINT),
        ("outputTechnology", wt.UINT),
        ("rotation", wt.UINT),
        ("scaling", wt.UINT),
        ("refreshRate_N", wt.UINT),
        ("refreshRate_D", wt.UINT),
        ("scanLineOrdering", wt.UINT),
        ("targetAvailable", wt.BOOL),
        ("statusFlags", wt.UINT),
    ]


class _PATH_INFO(ctypes.Structure):
    _fields_ = [
        ("sourceInfo", _PATH_SOURCE),
        ("targetInfo", _PATH_TARGET),
        ("flags", wt.UINT),
    ]


class _MODE_INFO(ctypes.Structure):
    _fields_ = [("_raw", ctypes.c_byte * 64)]


def _active_paths():
    path_n = wt.UINT(0)
    mode_n = wt.UINT(0)
    if _user32.GetDisplayConfigBufferSizes(
        _QDC_ONLY_ACTIVE_PATHS, ctypes.byref(path_n), ctypes.byref(mode_n)
    ) != _ERROR_SUCCESS:
        return None
    paths = (_PATH_INFO * path_n.value)()
    modes = (_MODE_INFO * mode_n.value)()
    if _user32.QueryDisplayConfig(
        _QDC_ONLY_ACTIVE_PATHS,
        ctypes.byref(path_n),
        paths,
        ctypes.byref(mode_n),
        modes,
        None,
    ) != _ERROR_SUCCESS:
        return None
    return [paths[i] for i in range(path_n.value)]


def _hdr_from_path(path) -> bool:
    t = path.targetInfo
    info2 = _ADV_COLOR_INFO_2()
    info2.header.type = _DINFO_GET_ADVANCED_COLOR_INFO_2
    info2.header.size = ctypes.sizeof(_ADV_COLOR_INFO_2)
    info2.header.adapterId = t.adapterId
    info2.header.id = t.id
    if _user32.DisplayConfigGetDeviceInfo(ctypes.byref(info2.header)) == _ERROR_SUCCESS:
        if int(info2.activeColorMode) == _COLOR_MODE_HDR:
            return True
        return bool(info2.value & _INFO2_HDR_USER)

    info = _ADV_COLOR_INFO()
    info.header.type = _DINFO_GET_ADVANCED_COLOR
    info.header.size = ctypes.sizeof(_ADV_COLOR_INFO)
    info.header.adapterId = t.adapterId
    info.header.id = t.id
    if _user32.DisplayConfigGetDeviceInfo(ctypes.byref(info.header)) != _ERROR_SUCCESS:
        return False
    supported = bool(info.value & 0x1)
    enabled = bool(info.value & 0x2)
    wide_enforced = bool(info.value & 0x4)
    # Win11 ACM：supported+enabled+wideEnforced，并不是用户开了 HDR
    return bool(supported and enabled and not wide_enforced)


def hdr_status_per_monitor() -> list[bool]:
    paths = _active_paths()
    if not paths:
        return []
    return [_hdr_from_path(p) for p in paths]


def is_hdr_on(monitor_idx: int = 0) -> bool:
    try:
        st = hdr_status_per_monitor()
        return bool(st[monitor_idx]) if monitor_idx < len(st) else False
    except Exception:
        return False


def sdr_white_nits(monitor_idx: int = 0) -> float:
    """当前显示器「SDR 内容亮度」滑条，单位 nits。失败则 203。"""
    paths = _active_paths()
    if not paths or monitor_idx >= len(paths):
        return _DEFAULT_SDR_WHITE_NITS
    t = paths[monitor_idx].targetInfo
    info = _SDR_WHITE()
    info.header.type = _DINFO_GET_SDR_WHITE_LEVEL
    info.header.size = ctypes.sizeof(_SDR_WHITE)
    info.header.adapterId = t.adapterId
    info.header.id = t.id
    if _user32.DisplayConfigGetDeviceInfo(ctypes.byref(info.header)) != _ERROR_SUCCESS:
        return _DEFAULT_SDR_WHITE_NITS
    nits = float(info.SDRWhiteLevel) / 1000.0
    # 本机实测 raw≈4850，不是 MSDN 的「nits×1000」。1000 对应 80 nits。
    if nits < 40.0:
        nits = 80.0 * float(info.SDRWhiteLevel) / 1000.0
    if nits < 80.0 or nits > 1000.0:
        return _DEFAULT_SDR_WHITE_NITS
    return nits


def scrgb_to_sdr_bgr_u8(bgra_f32, sdr_white: float):
    """scRGB 线性 BGRA float（1.0=80 nits）→ BGR uint8 sRGB。"""
    import numpy as np

    rgb = np.clip(bgra_f32[:, :, 2::-1] / max(sdr_white / _SCRGB_WHITE_NITS, 1e-6), 0.0, 1.0)
    srgb = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(np.maximum(rgb, 1e-10), 1.0 / 2.4) - 0.055,
    )
    u8 = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(u8[:, :, ::-1])
