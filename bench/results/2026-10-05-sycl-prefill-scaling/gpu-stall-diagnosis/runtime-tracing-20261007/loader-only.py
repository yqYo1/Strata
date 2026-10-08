import ctypes, json
l=ctypes.CDLL("libze_loader.so.1")
l.zeDriverGetApiVersion.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
l.zeDriverGetApiVersion.restype=ctypes.c_uint32
v=ctypes.c_uint32()
r=l.zeDriverGetApiVersion(None,ctypes.byref(v))
print(json.dumps({"probe":"null driver handle; no initialization, allocation or GPU commands", "result":hex(r),"api_version":v.value}),flush=True)
