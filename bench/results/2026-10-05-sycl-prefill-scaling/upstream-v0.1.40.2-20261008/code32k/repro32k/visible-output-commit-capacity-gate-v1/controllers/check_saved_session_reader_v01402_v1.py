"""CPU-only corruption and last-cell evidence checks; no GPU/engine execution."""
from pathlib import Path
import ast
import hashlib
import json
import struct
from read_saved_session_v01402_v1 import read_session

base=Path(__file__).parent
out=base/'saved-session-reader-v01402-host-check-v1'
out.mkdir(mode=0o700)
pack=lambda n:struct.pack('<Q',n)
vec=lambda data,elem=1:pack(len(data)//elem)+data
def checkpoint(used):
    return vec(struct.pack('<4i',1,2,3,4),4)+pack(0)+b''.join(vec(bytes([i])*8) for i in range(5))+pack(used)
def file_data(used=7,last=0):
    geometry=struct.pack('<18q',*range(1,19))+struct.pack('<qqQ',0,1,1)
    kv=struct.pack('<7q',1,4,2,256,4,2,128)
    data=bytearray(2048);data[-1]=last
    kv+=vec(data)+vec(bytes(2048))+vec(bytes(64))+vec(bytes(64))+vec(bytes(1024))
    body=geometry+checkpoint(used)+pack(1)+checkpoint(used)+pack(1)+kv
    header=b'STRSESS\x01'+struct.pack('<II6Q',1,64,1,2,len(body),0,0,0)
    return header+body+b'STRSEND\x01'+pack(0)
cases=[]
def parsed(name,data):
    p=out/(name+'.bin');assert not p.exists();p.write_bytes(data)
    return read_session(p)
reference=parsed('reference',file_data())
lru=parsed('lru-only',file_data(used=8))
assert reference['semantic_sha256']==lru['semantic_sha256'] and reference['sha256']!=lru['sha256']
cases.append({'name':'only-LRU-metadata-excluded','passed':True})
last=parsed('last-physical-byte-changed',file_data(last=1))
assert reference['semantic_sha256']!=last['semantic_sha256']
assert reference['semantic']['kv'][0]['parts']['k']['sha256']!=last['semantic']['kv'][0]['parts']['k']['sha256']
cases.append({'name':'last-physical-KV-byte-compared','passed':True})
for name,data in [('truncated',file_data()[:-1]),('trailer-corrupt',file_data()[:-16]+bytes(16)),
                  ('oversized-count',file_data()[:232]+pack(2**63)+file_data()[240:])]:
    try:parsed(name,data)
    except ValueError:cases.append({'name':name+'-rejected','passed':True})
    else:raise AssertionError(name)
controller=base/'run_owned_full_kv_access_v01402_v1.py'
tree=ast.parse(controller.read_text())
assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and '\\n' in n.value for n in ast.walk(tree))
cases.append({'name':'protocol-newlines-are-actual-newlines','passed':True})
record={'active':False,'passed':True,'gpu_tested':False,'cases':cases,
        'reader_sha256':hashlib.sha256((base/'read_saved_session_v01402_v1.py').read_bytes()).hexdigest(),
        'controller_sha256':hashlib.sha256(controller.read_bytes()).hexdigest(),
        'limitation':'Synthetic files deliberately omit codec checksums; actual engine RESTORE validates checksum and compatibility.'}
(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
