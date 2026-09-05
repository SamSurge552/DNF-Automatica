"""用 Windows API 把选区对齐到真实窗口客户区，坐标与 ImageGrab(all_screens=True) 一致。"""
import ctypes
import os
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
dwmapi = ctypes.windll.dwmapi

GA_ROOT = 2
GW_OWNER = 4
DWMWA_EXTENDED_FRAME_BOUNDS = 9
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_SKIP_CLASSES = {
    "Progman",
    "WorkerW",
    "Shell_TrayWnd",
    "Shell_SecondaryTrayWnd",
    "NotifyIconOverflowWindow",
}

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def enable_dpi_awareness():
    """让 WinAPI 窗口矩形与物理像素截图对齐。必须在创建 Tk 窗口之前调用。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def get_virtual_screen():
    return {
        "x": user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
        "y": user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
        "width": user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
        "height": user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
    }


def _window_title(hwnd):
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buf = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buf, length)
    return buf.value.strip()


def _window_class(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _client_rect(hwnd):
    rect = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return {
        "x": int(origin.x),
        "y": int(origin.y),
        "width": int(rect.right),
        "height": int(rect.bottom),
    }


def _visible_frame_rect(hwnd):
    rect = RECT()
    hr = dwmapi.DwmGetWindowAttribute(
        hwnd,
        DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if hr == 0:
        return {
            "x": int(rect.left),
            "y": int(rect.top),
            "width": int(rect.right - rect.left),
            "height": int(rect.bottom - rect.top),
        }
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return {
        "x": int(rect.left),
        "y": int(rect.top),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }


def get_window_at_point(x, y, exclude_hwnds=None):
    """返回点击位置最外层可见窗口的客户区矩形；对齐失败返回 None。"""
    exclude_hwnds = set(exclude_hwnds or [])
    hwnd = user32.WindowFromPoint(POINT(int(x), int(y)))
    if not hwnd:
        return None

    root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    if root in exclude_hwnds:
        return None

    class_name = _window_class(root)
    if class_name in _SKIP_CLASSES:
        return None

    title = _window_title(root)
    rect = _client_rect(root)
    if rect["width"] <= 0 or rect["height"] <= 0:
        rect = _visible_frame_rect(root)
    if rect["width"] <= 0 or rect["height"] <= 0:
        return None

    rect["title"] = title or class_name or str(root)
    rect["hwnd"] = int(root)
    return rect


def _process_exe_name(hwnd):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        kernel32.CloseHandle(handle)


def _window_rect(hwnd):
    class_name = _window_class(hwnd)
    if class_name in _SKIP_CLASSES:
        return None
    title = _window_title(hwnd)
    rect = _client_rect(hwnd)
    if rect["width"] <= 0 or rect["height"] <= 0:
        rect = _visible_frame_rect(hwnd)
    if rect["width"] <= 0 or rect["height"] <= 0:
        return None
    rect["title"] = title or class_name or str(hwnd)
    rect["hwnd"] = int(hwnd)
    return rect


SW_RESTORE = 9
GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
ASFW_ANY = -1


def _long_ptr():
    try:
        return user32.GetWindowLongPtrW, user32.SetWindowLongPtrW
    except AttributeError:
        return user32.GetWindowLongW, user32.SetWindowLongW


def foreground_hwnd() -> int:
    try:
        return int(user32.GetForegroundWindow() or 0)
    except Exception:
        return 0


def find_window_by_title(title: str) -> int | None:
    if not title:
        return None
    hwnd = user32.FindWindowW(None, str(title))
    if hwnd:
        return int(hwnd)
    found: list[int] = []
    target = str(title)

    def _enum(h, _lparam):
        if not user32.IsWindowVisible(h):
            return True
        name = _window_title(h)
        if name == target or name.startswith(target):
            found.append(int(h))
        return True

    enum_proc = WNDENUMPROC(_enum)
    user32.EnumWindows(enum_proc, 0)
    return found[0] if found else None


def prevent_activate(hwnd) -> bool:
    """预览窗：不抢激活、置顶，避免 cv2.imshow 把游戏焦点抢走。"""
    try:
        hwnd = int(hwnd)
    except (TypeError, ValueError):
        return False
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    get_long, set_long = _long_ptr()
    try:
        style = int(get_long(hwnd, GWL_EXSTYLE) or 0)
        set_long(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
    except Exception:
        return False
    return True


def focus_hwnd(hwnd) -> bool:
    """把窗口拉到前台。点开始后 GUI 占着焦点时也能切走。"""
    try:
        hwnd = int(hwnd)
    except (TypeError, ValueError):
        return False
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    try:
        user32.AllowSetForegroundWindow(ASFW_ANY)
    except Exception:
        pass
    cur_tid = kernel32.GetCurrentThreadId()
    fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = False
    if fg_tid:
        attached = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
    try:
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(cur_tid, fg_tid, False)
    return user32.GetForegroundWindow() == hwnd


def focus_point(x: int, y: int) -> bool:
    hwnd = user32.WindowFromPoint(POINT(int(x), int(y)))
    if not hwnd:
        return False
    hwnd = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    return focus_hwnd(hwnd)


def focus_region(rect) -> bool:
    """对齐结果优先用 hwnd；失效则找 DNF.exe，再点客户区中心。"""
    if not isinstance(rect, dict):
        return False
    hwnd = rect.get("hwnd")
    if hwnd and focus_hwnd(hwnd):
        return True
    found = find_window_by_process("dnf.exe")
    if found and focus_hwnd(found.get("hwnd")):
        return True
    try:
        x = int(rect.get("x"))
        y = int(rect.get("y"))
        w = max(int(rect.get("width") or 1), 1)
        h = max(int(rect.get("height") or 1), 1)
    except (TypeError, ValueError):
        return False
    return focus_point(x + w // 2, y + h // 2)


def find_window_by_process(exe_name):
    """按进程名查找可见窗口（忽略大小写），多个时取客户区面积最大的。"""
    target = os.path.basename(exe_name).lower()
    if not target.endswith(".exe"):
        target += ".exe"
    found = []

    def _enum(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if _process_exe_name(hwnd).lower() != target:
            return True
        rect = _window_rect(hwnd)
        if rect:
            found.append(rect)
        return True

    # ctypes 回调必须保持引用，否则 EnumWindows 期间可能被回收
    enum_proc = WNDENUMPROC(_enum)
    user32.EnumWindows(enum_proc, 0)
    if not found:
        return None
    return max(found, key=lambda r: r["width"] * r["height"])
