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

u=ctypes.CDLL("libur_loader.so.0")
u.urLoaderInit.argtypes=[ctypes.c_uint32,ctypes.c_void_p];u.urLoaderInit.restype=ctypes.c_uint32
u.urAdapterGet.argtypes=[ctypes.c_uint32,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_uint32)];u.urAdapterGet.restype=ctypes.c_uint32
u.urAdapterRelease.argtypes=[ctypes.c_void_p];u.urAdapterRelease.restype=ctypes.c_uint32
u.urLoaderTearDown.argtypes=[];u.urLoaderTearDown.restype=ctypes.c_uint32
check("urLoaderInit",u.urLoaderInit(0,None))
c=ctypes.c_uint32();check("urAdapterGet(count)",u.urAdapterGet(0,None,ctypes.byref(c)))
if not c.value:raise RuntimeError("no UR adapter")
a=(ctypes.c_void_p*c.value)();check("urAdapterGet(handles)",u.urAdapterGet(c.value,a,ctypes.byref(c)))
for h in a:check("urAdapterRelease",u.urAdapterRelease(h))
check("urLoaderTearDown",u.urLoaderTearDown())
print("READY_FOR_FLUSH_CHECK",flush=True)
import sys
sys.stdin.buffer.read(1)
