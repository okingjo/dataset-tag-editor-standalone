import sys
import cmd_args
import torch


# ================================================================
# borrowed from AUTOMATIC1111/stable-diffusion-webui

# has_mps is only available in nightly pytorch (for now) and macOS 12.3+.
# check `getattr` and try it for compatibility
def check_for_mps() -> bool:
    if hasattr(torch, "backends"):
        return torch.backends.mps.is_available()
    elif not getattr(torch, "has_mps", False):
        return False
    
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


_has_mps = check_for_mps()

# ================================================================


def has_mps():
    if sys.platform != "darwin":
        return False
    else:
        return _has_mps


def get_cuda():
    if cmd_args.opts.device_id is not None:
        return torch.cuda.device(f"cuda:{cmd_args.opts.device_id}")
    else:
        return torch.cuda.device("cuda")


def get_cuda_device():
    if cmd_args.opts.device_id is not None:
        return torch.device(f"cuda:{cmd_args.opts.device_id}")
    else:
        return torch.device("cuda")


def get_optimal_device():
    if torch.cuda.is_available():
        return get_cuda_device()

    if has_mps():
        torch.device("mps")

    try:
        import torch_directml
        return torch_directml.device()
    except:
        pass

    return torch.device("cpu")


# from AUTOMATIC1111's SDwebUI
def torch_gc():
    if torch.cuda.is_available():
        with get_cuda():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


device = get_optimal_device()
cpu = torch.device("cpu")


# ================================================================
# DirectML device selection (Windows ONNX taggers).
# DML device_id follows DXGI adapter order, which on mixed-GPU machines
# (Intel iGPU + NVIDIA/AMD dGPU, virtual display adapters) is NOT always the
# discrete GPU at 0. Pick the first hardware adapter from a discrete-GPU
# vendor; fall back to the first hardware adapter, then 0.

_dml_device_id_cache = None


def _enum_dxgi_adapters():
    """Yield (vendor_id, device_id, flags, description) in DXGI order."""
    import ctypes
    from ctypes import byref, POINTER, Structure, c_void_p, c_uint, c_size_t, c_wchar, c_long, c_ulong

    class GUID(Structure):
        _fields_ = [("Data1", c_uint), ("Data2", ctypes.c_ushort),
                    ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

    class DXGI_ADAPTER_DESC1(Structure):
        _fields_ = [
            ("Description", c_wchar * 128),
            ("VendorId", c_uint), ("DeviceId", c_uint),
            ("SubSysId", c_uint), ("Revision", c_uint),
            ("DedicatedVideoMemory", c_size_t), ("DedicatedSystemMemory", c_size_t),
            ("SharedSystemMemory", c_size_t),
            ("AdapterLuid_lo", c_uint), ("AdapterLuid_hi", c_long),
            ("Flags", c_uint),
        ]

    dxgi = ctypes.WinDLL("dxgi")
    CreateDXGIFactory1 = dxgi.CreateDXGIFactory1
    CreateDXGIFactory1.argtypes = [POINTER(GUID), POINTER(c_void_p)]
    CreateDXGIFactory1.restype = c_long  # plain HRESULT, no auto-raise

    iid = GUID()  # IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    iid.Data1, iid.Data2, iid.Data3 = 0x770AAE78, 0xF26F, 0x4DBA
    iid.Data4 = (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)

    factory = c_void_p()
    if CreateDXGIFactory1(byref(iid), byref(factory)) != 0:
        return
    vt = ctypes.cast(factory, POINTER(POINTER(c_void_p))).contents
    EnumAdapters1 = ctypes.WINFUNCTYPE(c_long, c_void_p, c_uint, POINTER(c_void_p))(vt[12])
    try:
        i = 0
        while True:
            adapter = c_void_p()
            if EnumAdapters1(factory, i, byref(adapter)) != 0:
                break  # DXGI_ERROR_NOT_FOUND = end of list
            avt = ctypes.cast(adapter, POINTER(POINTER(c_void_p))).contents
            GetDesc1 = ctypes.WINFUNCTYPE(c_long, c_void_p, POINTER(DXGI_ADAPTER_DESC1))(avt[10])
            desc = DXGI_ADAPTER_DESC1()
            if GetDesc1(adapter, byref(desc)) == 0:
                yield desc.VendorId, desc.DeviceId, desc.Flags, desc.Description
            ctypes.WINFUNCTYPE(c_ulong, c_void_p)(avt[2])(adapter)  # Release
            i += 1
    finally:
        ctypes.WINFUNCTYPE(c_ulong, c_void_p)(vt[2])(factory)  # Release


def get_dml_device_id():
    global _dml_device_id_cache
    if _dml_device_id_cache is not None:
        return _dml_device_id_cache
    _dml_device_id_cache = 0
    if sys.platform == "win32":
        try:
            adapters = list(_enum_dxgi_adapters())
            hardware = [(i, a) for i, a in enumerate(adapters) if not (a[2] & 2)]
            chosen = next((i for i, a in hardware if a[0] in (0x10DE, 0x1002)), None)
            if chosen is None and hardware:
                chosen = hardware[0][0]
            if chosen is not None:
                _dml_device_id_cache = chosen
        except Exception:
            pass
    return _dml_device_id_cache
