"""Freeze this official compiler's host linker inputs before a NEW build.

Repository LICENSE. The previous scope remains failed; this is prospective
inventory from the compiler's queries under the exact build environment.
"""
import hashlib
from pathlib import Path
import subprocess

NAMES=('crtbeginS.o','crtendS.o','libgcc.a','libgcc_s.so','Scrt1.o','crti.o','crtn.o',
       'libc.so','libc_nonshared.a','libdl.a','libm.so','libmvec.so.1','libpthread.a')


def inventory(compiler,environment):
    files={};queries=[];owners=set()
    for option in ['-print-prog-name=ld',*['-print-file-name='+name for name in NAMES]]:
        argv=[compiler,'-fsycl',option]
        reported=subprocess.check_output(argv,env=environment,text=True,timeout=15).strip()
        path=Path(reported)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError('compiler did not resolve host link input: '+option+':'+reported)
        path=path.resolve()
        if not path.is_relative_to('/usr'):
            raise RuntimeError('unexpected non-system host linker input: '+str(path))
        files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        queries.append(dict(argv=argv,reported=reported,resolved=str(path)))
        ownership=subprocess.check_output(['dpkg-query','-S',str(path)],env=environment,text=True,timeout=15)
        matches=[line.split(': /',1)[0] for line in ownership.splitlines() if ': /' in line]
        if not matches:raise RuntimeError('unowned host linker input: '+str(path))
        owners.update(matches)
    versions=subprocess.check_output(['dpkg-query','-W','-f=${binary:Package} ${Version}\n',*sorted(owners)],
                                    env=environment,text=True,timeout=15).splitlines()
    return dict(files=files,compiler_queries=queries,packages=versions,
                scope='Exact host linker,CRT,archive/linker-script and math-library bytes from official driver queries. No GPU runtime,safety key or old failure status is changed.')
