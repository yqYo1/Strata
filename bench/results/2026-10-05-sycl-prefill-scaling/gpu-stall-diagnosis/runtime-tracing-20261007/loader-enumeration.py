import ctypes, json
l=ctypes.CDLL("libze_loader.so.1")
l.zeInit.argtypes=[ctypes.c_uint32];l.zeInit.restype=ctypes.c_uint32
l.zeDriverGet.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_void_p)];l.zeDriverGet.restype=ctypes.c_uint32
l.zeDriverGetApiVersion.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)];l.zeDriverGetApiVersion.restype=ctypes.c_uint32
def check(name,result):
 print(json.dumps({"api":name,"result":hex(result)}),flush=True)
 if result:raise RuntimeError(name+" failed")
check("zeInit(GPU_ONLY)",l.zeInit(1))
n=ctypes.c_uint32();check("zeDriverGet(count)",l.zeDriverGet(ctypes.byref(n),None))
if not n.value:raise RuntimeError("no GPU driver")
h=(ctypes.c_void_p*n.value)();check("zeDriverGet(handles)",l.zeDriverGet(ctypes.byref(n),h))
v=ctypes.c_uint32();check("zeDriverGetApiVersion(valid_handle)",l.zeDriverGetApiVersion(h[0],ctypes.byref(v)))
print(json.dumps({"drivers":n.value,"api_version":v.value,"contexts_created":0,"allocations":0,"gpu_commands":0}),flush=True)
