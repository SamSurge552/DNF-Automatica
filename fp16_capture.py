"""HDR 桌面：DXGI DuplicateOutput1 采 R16G16B16A16_FLOAT（scRGB 线性）。"""
from __future__ import annotations

import ctypes
from typing import Any, cast

import comtypes
import numpy as np

from dxcam._libs.d3d11 import (
    D3D11_CPU_ACCESS_READ,
    D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_STAGING,
    D3D_FEATURE_LEVEL_10_0,
    D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0,
    ID3D11Device,
    ID3D11DeviceContext,
    ID3D11Texture2D,
)
from dxcam._libs.dxgi import (
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_WAIT_TIMEOUT,
    DXGI_OUTDUPL_FLAG_NONE,
    DXGI_OUTDUPL_FRAME_INFO,
    DXGI_OUTPUT_DESC,
    IDXGIAdapter1,
    IDXGIFactory1,
    IDXGIOutput,
    IDXGIOutput5,
    IDXGIOutputDuplication,
    IDXGIResource,
)

DXGI_FORMAT_R16G16B16A16_FLOAT = 10
D3D11_MAP_READ = 1
_PTR_SZ = ctypes.sizeof(ctypes.c_void_p)
_SLOT_CTX_MAP = 14
_SLOT_CTX_UNMAP = 15
_SLOT_CTX_COPYRESOURCE = 47
_HR_TIMEOUT = ctypes.c_int32(DXGI_ERROR_WAIT_TIMEOUT).value
_HR_ACCESS_LOST = ctypes.c_int32(DXGI_ERROR_ACCESS_LOST).value


class D3D11_MAPPED_SUBRESOURCE(ctypes.Structure):
    _fields_ = [
        ("pData", ctypes.c_void_p),
        ("RowPitch", ctypes.c_uint),
        ("DepthPitch", ctypes.c_uint),
    ]


_MapFn = ctypes.WINFUNCTYPE(
    ctypes.HRESULT,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
)
_UnmapFn = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
_CopyFn = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def _vtable_fn(com_ptr, slot: int, fn_type):
    this = ctypes.cast(com_ptr, ctypes.c_void_p).value
    vtable = ctypes.cast(this, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn_raw = ctypes.cast(vtable + slot * _PTR_SZ, ctypes.POINTER(ctypes.c_void_p)).contents.value
    return fn_type(fn_raw), this


class FP16CaptureError(RuntimeError):
    pass


class AccessLostError(FP16CaptureError):
    pass


def _find_adapter_and_output(target_idx: int):
    dxgi_dll = ctypes.windll.dxgi
    dxgi_dll.CreateDXGIFactory1.argtypes = (comtypes.GUID, ctypes.POINTER(ctypes.c_void_p))
    dxgi_dll.CreateDXGIFactory1.restype = ctypes.c_int32
    pf = ctypes.c_void_p(0)
    if dxgi_dll.CreateDXGIFactory1(IDXGIFactory1._iid_, ctypes.byref(pf)) < 0:
        return None, None
    factory: Any = cast(Any, ctypes.cast(pf, ctypes.POINTER(IDXGIFactory1)))
    flat_idx = 0
    ai = 0
    while True:
        try:
            adp = ctypes.POINTER(IDXGIAdapter1)()
            factory.EnumAdapters1(ai, ctypes.byref(adp))
        except comtypes.COMError:
            break
        oi = 0
        while True:
            try:
                out = ctypes.POINTER(IDXGIOutput)()
                cast(Any, adp).EnumOutputs(oi, ctypes.byref(out))
            except comtypes.COMError:
                break
            if flat_idx == target_idx:
                return adp, out
            flat_idx += 1
            oi += 1
        ai += 1
    return None, None


class FP16Capture:
    def __init__(self, output_idx: int = 0) -> None:
        self._output_idx = output_idx
        self._width = 0
        self._height = 0
        self._origin = (0, 0)
        self._dev = None
        self._ctx = None
        self._dupl = None
        self._stg_ptr = None
        self._frame_held = False
        self._released = False
        self._last_frame = None
        self._setup()

    def _setup(self) -> None:
        adapter, output = _find_adapter_and_output(self._output_idx)
        if adapter is None:
            raise FP16CaptureError(f"Output {self._output_idx} not found")

        dev_ptr = ctypes.POINTER(ID3D11Device)()
        ctx_ptr = ctypes.POINTER(ID3D11DeviceContext)()
        feat = (ctypes.c_uint * 3)(
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
        )
        hr = ctypes.windll.d3d11.D3D11CreateDevice(
            adapter, 0, None, 0, feat, 3, 7, ctypes.byref(dev_ptr), None, ctypes.byref(ctx_ptr)
        )
        if hr < 0:
            raise FP16CaptureError(f"D3D11CreateDevice failed: 0x{hr & 0xFFFFFFFF:08X}")
        self._dev = cast(Any, dev_ptr)
        self._ctx = cast(Any, ctx_ptr)
        self._dev_ptr = dev_ptr
        self._ctx_ptr = ctx_ptr

        desc = DXGI_OUTPUT_DESC()
        cast(Any, output).GetDesc(ctypes.byref(desc))
        self._origin = (int(desc.DesktopCoordinates.left), int(desc.DesktopCoordinates.top))
        self._width = int(desc.DesktopCoordinates.right - desc.DesktopCoordinates.left)
        self._height = int(desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top)

        try:
            out5 = cast(Any, output).QueryInterface(IDXGIOutput5)
        except comtypes.COMError as e:
            raise FP16CaptureError(f"IDXGIOutput5 not supported: {e}") from e

        formats = (ctypes.c_uint * 1)(DXGI_FORMAT_R16G16B16A16_FLOAT)
        dupl_ptr = ctypes.POINTER(IDXGIOutputDuplication)()
        try:
            out5.DuplicateOutput1(
                ctypes.cast(dev_ptr, ctypes.c_void_p),
                DXGI_OUTDUPL_FLAG_NONE,
                1,
                formats,
                ctypes.byref(dupl_ptr),
            )
        except comtypes.COMError as e:
            raise FP16CaptureError(f"DuplicateOutput1 fp16 failed: {e}") from e
        self._dupl = cast(Any, dupl_ptr)
        self._dupl_ptr = dupl_ptr

    def _ensure_staging(self, src_tex: Any) -> None:
        if self._stg_ptr is not None:
            return
        sd = D3D11_TEXTURE2D_DESC()
        src_tex.GetDesc(ctypes.byref(sd))
        s = D3D11_TEXTURE2D_DESC()
        s.Width = sd.Width
        s.Height = sd.Height
        s.MipLevels = 1
        s.ArraySize = 1
        s.Format = DXGI_FORMAT_R16G16B16A16_FLOAT
        s.SampleDesc.Count = 1
        s.SampleDesc.Quality = 0
        s.Usage = D3D11_USAGE_STAGING
        s.CPUAccessFlags = D3D11_CPU_ACCESS_READ
        s.BindFlags = 0
        s.MiscFlags = 0
        stg = ctypes.POINTER(ID3D11Texture2D)()
        self._dev.CreateTexture2D(ctypes.byref(s), None, ctypes.byref(stg))
        self._stg_ptr = stg

    def grab(self, timeout_ms: int = 50):
        if self._released:
            raise FP16CaptureError("released")
        if self._frame_held:
            try:
                self._dupl.ReleaseFrame()
            except Exception:
                pass
            self._frame_held = False

        info = DXGI_OUTDUPL_FRAME_INFO()
        res_ptr = ctypes.POINTER(IDXGIResource)()
        try:
            self._dupl.AcquireNextFrame(timeout_ms, ctypes.byref(info), ctypes.byref(res_ptr))
        except comtypes.COMError as e:
            hr = ctypes.c_int32(e.hresult).value
            if hr == _HR_TIMEOUT:
                return self._last_frame
            if hr == _HR_ACCESS_LOST:
                raise AccessLostError("access lost") from e
            raise FP16CaptureError(f"AcquireNextFrame 0x{e.hresult & 0xFFFFFFFF:08X}") from e

        self._frame_held = True
        if int(info.LastPresentTime) == 0:
            self._dupl.ReleaseFrame()
            self._frame_held = False
            return self._last_frame

        src_tex: Any = cast(Any, res_ptr).QueryInterface(ID3D11Texture2D)
        self._ensure_staging(src_tex)
        src_raw = ctypes.cast(src_tex, ctypes.c_void_p).value
        stg_raw = ctypes.cast(self._stg_ptr, ctypes.c_void_p).value
        copy_fn, ctx_copy = _vtable_fn(self._ctx_ptr, _SLOT_CTX_COPYRESOURCE, _CopyFn)
        copy_fn(ctx_copy, stg_raw, src_raw)
        self._dupl.ReleaseFrame()
        self._frame_held = False

        mapped = D3D11_MAPPED_SUBRESOURCE()
        map_fn, ctx_this = _vtable_fn(self._ctx_ptr, _SLOT_CTX_MAP, _MapFn)
        hr = map_fn(ctx_this, stg_raw, 0, D3D11_MAP_READ, 0, ctypes.addressof(mapped))
        if hr < 0:
            raise FP16CaptureError(f"Map failed 0x{hr & 0xFFFFFFFF:08X}")
        try:
            w, h = self._width, self._height
            row_pitch = int(mapped.RowPitch)
            ptr = ctypes.cast(mapped.pData, ctypes.POINTER(ctypes.c_uint8))
            raw = np.ctypeslib.as_array(ptr, shape=(row_pitch * h,))
            fp16_rows = raw.view(np.float16).reshape(h, row_pitch // 2)
            fp16_rgba = fp16_rows[:, : w * 4].reshape(h, w, 4).copy()
        finally:
            unmap_fn, ctx2 = _vtable_fn(self._ctx_ptr, _SLOT_CTX_UNMAP, _UnmapFn)
            unmap_fn(ctx2, stg_raw, 0)

        bgra = np.empty((h, w, 4), dtype=np.float32)
        bgra[:, :, 0] = fp16_rgba[:, :, 2]
        bgra[:, :, 1] = fp16_rgba[:, :, 1]
        bgra[:, :, 2] = fp16_rgba[:, :, 0]
        bgra[:, :, 3] = fp16_rgba[:, :, 3]
        self._last_frame = bgra
        return bgra

    def crop_virtual(self, left: int, top: int, right: int, bottom: int, timeout_ms: int = 50):
        frame = self.grab(timeout_ms=timeout_ms)
        if frame is None:
            return None
        ox, oy = self._origin
        x0 = max(0, left - ox)
        y0 = max(0, top - oy)
        x1 = min(self._width, right - ox)
        y1 = min(self._height, bottom - oy)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1]

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._frame_held:
            try:
                self._dupl.ReleaseFrame()
            except Exception:
                pass
            self._frame_held = False
        self._stg_ptr = None
        self._last_frame = None
        self._dupl = None
        self._dupl_ptr = None
        self._ctx = None
        self._ctx_ptr = None
        self._dev = None
        self._dev_ptr = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
