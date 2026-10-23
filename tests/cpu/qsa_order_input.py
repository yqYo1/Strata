#!/usr/bin/env python3
"""Main-authored malformed capture tests; repository LICENSE.

Caller owns this process and its finite children. No runtime/device is created:
the replay's --host-check path parses before runtime_for. Fixture bytes are
generated from this committed recipe, consumed here and are not model captures.
"""
from array import array
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys


def require(value,message):
    if not value:raise RuntimeError(message)


def run(binary,expected_sha256,root):
    require(sys.byteorder=='little','capture ABI requires little-endian host')
    binary=Path(binary);root=Path(root)
    require(hashlib.sha256(binary.read_bytes()).hexdigest()==expected_sha256,'exact parser binary')
    root.mkdir(mode=0o700);base=root/'fixture';ids=Path(str(base)+'.ids');steps=Path(str(base)+'.steps')
    queries,cap,first=512,2051,8192
    id_record_bytes=16+queries*cap*4;step_record_bytes=16+queries*4*4
    def generate():
        with ids.open('wb') as a,steps.open('wb') as b:
            row=array('i',range(cap));require(row.itemsize==4,'int32 ABI')
            for layer in range(12):
                a.write(struct.pack('<4i',layer,first,queries,cap));b.write(struct.pack('<4i',layer,first,queries,4))
                for q in range(queries):
                    row.tofile(a);n=first+q+1;b.write(struct.pack('<4i',n-1,n,n//4,cap))
    samples=[]
    def check(name,success,error=None):
        p=subprocess.run([str(binary),'--host-check',str(base)],capture_output=True,text=True,timeout=30)
        require(len(p.stdout)+len(p.stderr)<65536,'bounded parser output')
        require((p.returncode==0)==success,'unexpected status: '+name)
        if success:require(p.stdout=='capture_parser=PASS records=12 gpu=UNRUN\n' and p.stderr=='','exact host-only success')
        else:require(p.returncode==1 and error in p.stderr,'expected parser refusal: '+name)
        samples.append(dict(case=name,exit_code=p.returncode,expected=success,passed=True))
    def change(path,offset,value,name,error):
        with path.open('r+b') as f:
            f.seek(offset);old=f.read(4);require(len(old)==4,'patch inside fixture')
            f.seek(offset);f.write(struct.pack('<i',value))
        try:check(name,False,error)
        finally:
            with path.open('r+b') as f:f.seek(offset);f.write(old)
    generate();check('valid',True)
    change(ids,4,first+1,'paired-first','paired actual record headers')
    change(ids,id_record_bytes,0,'duplicate-layer','paired actual record headers')
    change(ids,16,first+1,'noncausal-ID','active IDs must be unique ascending and causal')
    change(ids,20,0,'unordered-ID','active IDs must be unique ascending and causal')
    change(steps,16+12,cap+1,'width-over-cap','actual step contract')
    change(steps,16+4,0,'wrong-nkv','actual step contract')
    change(steps,12,7,'step-stride','paired actual record headers')
    for path,label in ((ids,'ids'),(steps,'steps')):
        size=path.stat().st_size
        with path.open('r+b') as f:f.seek(size-4);tail=f.read(4);f.truncate(size-4)
        try:check('truncated-'+label,False,'short capture payload')
        finally:
            with path.open('ab') as f:f.write(tail)
    with steps.open('ab') as f:f.write(struct.pack('<i',0))
    try:check('trailing-steps',False,'short capture header')
    finally:
        with steps.open('r+b') as f:f.truncate(step_record_bytes*12)
    missing=root/'missing.steps';steps.rename(missing)
    try:check('missing-pair',False,'paired capture open')
    finally:missing.rename(steps)
    with ids.open('r+b') as f:f.truncate(id_record_bytes*11)
    with steps.open('r+b') as f:f.truncate(step_record_bytes*11)
    check('missing-layer',False,'exactly twelve paired layers');generate();check('valid-after-restoration',True)
    print(json.dumps(dict(passed=True,GPU_executed=False,samples=samples,
        generated_bytes=ids.stat().st_size+steps.stat().st_size,
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ids,steps)})),flush=True)


if __name__=='__main__':
    require(len(sys.argv)==4,'BINARY SHA256 FRESH_PRIVATE_ROOT')
    run(*sys.argv[1:])
