#!/usr/bin/env python3
"""MAIN ONLY: standalone CPU fixture build. Never modifies product source or runs it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

A=Path(__file__).resolve().parent
P=A.parents[3]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--sanitize',action='store_true')
    a=p.parse_args()
    pins=json.loads((A/'pins.json').read_text())
    manifest=json.loads((A/'manifest.json').read_text())
    for path,h in pins['source_sha256'].items():
        if sha(path)!=h:raise RuntimeError('source pin changed: '+path)
    for name,h in manifest['artifact_sha256'].items():
        if sha(A/name)!=h:raise RuntimeError('artifact changed: '+name)
    compiler=Path(pins['compiler']['path'])
    if sha(compiler)!=pins['compiler']['sha256']:raise RuntimeError('compiler pin changed')
    for path,h in pins['archive_sha256'].items():
        if sha(path)!=h:raise RuntimeError('archive pin changed: '+path)
    dep=Path(pins['configured_dependency'])
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('output directory must be empty')
    flags=['-std=c++20','-march=znver3','-mtune=znver3','-mavx2','-mfma','-mf16c',
           '-fno-fast-math','-fno-associative-math','-ffp-contract=off','-Wall','-Wextra',
           '-I'+str(P),'-I'+str(P/'include'),'-I'+str(P/'third_party/ggml'),'-I'+str(dep/'include')]
    flags+=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if a.sanitize else ['-O3','-DNDEBUG']
    archives=list(pins['archive_sha256'])
    command=[str(compiler),*flags,str(A/'probe.cpp'),*archives,'-lpthread','-lm','-ldl','-o',str(out/'fixture')]
    r=dict(status='building',command=command,cwd=str(P),sanitized=a.sanitize,
           compiler=pins['compiler'],archive_sha256=pins['archive_sha256'],
           source_sha256=pins['source_sha256'],manifest_sha256=sha(A/'manifest.json'),
           product_applied=False,GPU_executed=False,source_includes_originals=True,
           sanitizer_scope='Fixture and included iq_avx2/iq_single are instrumented; prebuilt GGML archive internals are not.')
    def persist():(out/'build-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
    persist()
    try:
        v=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,check=True,timeout=15)
        r['compiler_version']=v.stdout+v.stderr
        # Compile-only macro proof; ensures exact helper does not become its stub.
        m=subprocess.run([str(compiler),'-march=znver3','-dM','-E','-x','c++','/dev/null'],capture_output=True,text=True,check=True,timeout=15)
        r['target_macros']=[s for s in m.stdout.splitlines() if any(x in s for x in ['__znver3','__AVX2__','__FMA__','__F16C__','__clang__'])]
        if '__znver3' not in m.stdout or '__clang__' not in m.stdout:raise RuntimeError('actual target must be clang/znver3')
        with (out/'compile.log').open('w') as log:
            subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        r.update(status='built_unrun',binary_sha256=sha(out/'fixture'))
    except BaseException as e:
        r.update(status='build_failed',error=repr(e));raise
    finally:persist()

if __name__=='__main__':main()
