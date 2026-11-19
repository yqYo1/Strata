#!/usr/bin/env python3
"""Prove an existing-ELF queue wait observer on CPU first. Repository LICENSE."""

import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import measurement_guards as guards
import measurement_owner as ownership
import prefill_prefix_expert_build as stage

B, P, RESULT = basis.B, basis.P, basis.RESULT
ROOT = B / 'prefill-compat-wait-audit-v1'
KEY = 'prefill_compat_wait_audit_host_v1'
PREVIOUS = 'prefill_prefix_expert_cpu_wait_v2'
C = 'tools/sycl/compat_wait_audit.c'
CXX = 'tools/sycl/compat_wait_canary.cpp'
SYMBOL = '_ZN4sycl3_V15queue20wait_and_throw_proxyERKNS0_6detail13code_locationE'
WRAPPER = '_Z21cudaStreamSynchronizePN4sycl3_V15queueE'
GATE = 'compat_wait_canary PASS value=42 exception=passed'
A = ROOT / 'wait_audit.so'
L = ROOT / 'libstrata_wait_canary.so'
E = ROOT / 'strata_wait_canary'
ENV = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', TMPDIR=str(ROOT / 'tmp'))
RUNTIME = dict(wall=45, cpu=30, rss_cap=1 << 30, text_cap=128 << 10,
               file_cap=8 << 20, total_cap=16 << 20)
require, sha, signature = basis.require, basis.sha, basis.signature


def clean_command(command, env=ENV):
    return subprocess.check_output(command, text=True, env=env, timeout=20)


def source_pins(revision):
    paths = (C, CXX, 'tools/sycl/compat_wait_audit.py', 'tools/sycl/measurement_owner.py',
             'tools/sycl/measurement_guards.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
    data = dict(stage.source.blobs(P, revision, paths))
    require(set(data) == set(paths) and all((P / p).read_bytes() == v for p, v in data.items()),
            'commit exact source and policies before preparing the host experiment')
    return stage.hashed(data)


def identify_native():
    record = basis.validation.committed(RESULT, (PREVIOUS,))
    require(record['stage'] == 'closed' and record['ownership_closed'] and not record['active']
            and record['decision']['normal_route_attribution_qualified'], 'original wait attribution required')
    binary = record['binary']
    elf = binary['path']
    require(sha(elf) == binary['sha256'], 'retained engine ELF differs')
    header = clean_command(['/usr/bin/readelf', '-h', elf])
    dynamic = clean_command(['/usr/bin/readelf', '-d', elf])
    require('EXEC (Executable file)' in header and 'X86-64' in header and 'BIND_NOW' not in dynamic
            and 'Flags: NOW' not in dynamic, 'fixed-address lazy-PLT engine required')
    text = clean_command(['/usr/bin/objdump', '-d', '--disassemble='+WRAPPER, elf])
    matches = re.findall(r'^\s*([0-9a-f]+):\s*((?:[0-9a-f]{2}\s+)+)call\s+[0-9a-f]+ <'
                         + re.escape(SYMBOL) + r'@plt>$', text, flags=re.M)
    require(len(matches) == 1, 'the compatibility wrapper must call the one expected PLT boundary')
    matched, code = matches[0]
    pc = int(matched, 16)
    ret = pc + len(code.split())
    cfi = clean_command(['/usr/bin/readelf', '--debug-dump=frames-interp', elf])
    rows = {}
    for block in cfi.split('\n\n'):
        match = re.search(r'pc=([0-9a-f]+)\.\.([0-9a-f]+)', block)
        if match and int(match[1],16) <= pc < int(match[2],16):
            rows = {int(a,16):int(n) for a,n in re.findall(r'^([0-9a-f]+) rsp\+(\d+).* c-8\s*$',block,re.M)}
            break
    require(rows and min(rows) <= pc, 'no applicable x86_64 return-address CFI row')
    offset = rows[max(a for a in rows if a <= pc)]
    require(offset >= 16 and offset <= 2048 and not offset % 8,
            'unsupported compatibility wrapper stack displacement')
    return dict(binary=binary, reference=signature(record), wait_symbol=SYMBOL,
                wrapper=WRAPPER, native_call=hex(pc), native_return=hex(ret), stack_offset=offset)


def compact(row):
    return {k:v for k,v in row.items() if k not in ('sources','dependencies')}


def save(row):
    stage.write(ROOT / 'record.json',row)
    data = json.loads(RESULT.read_bytes())
    data[KEY] = compact(row)
    stage.write(RESULT,data)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(RESULT), 'one fresh observer build')
    source_commit = stage.source.git(P,'rev-parse','HEAD').decode().strip()
    sources = source_pins(source_commit)
    ROOT.mkdir(mode=0o700)
    (ROOT/'tmp').mkdir(mode=0o700)
    cc, cxx = '/usr/bin/gcc', '/usr/bin/g++'
    cflags = ['-std=c11','-Wall','-Wextra','-Werror','-O2','-fPIC']
    cxxflags = ['-std=c++20','-Wall','-Wextra','-Werror','-O2']
    dependency_paths = {str(Path(x).resolve()) for x in (cc,cxx,'/usr/bin/ld','/usr/bin/as',
                    '/usr/bin/objdump','/usr/bin/readelf','/usr/bin/nm',sys.executable)}
    for compiler, flags, name in ((cc,cflags,C),(cxx,cxxflags,CXX)):
        text=clean_command([compiler,*flags,'-M',str(P/name)])
        parts=shlex.split(text.split(':',1)[1].replace('\\\n',' '))
        dependency_paths.update(str(Path(x).resolve()) for x in parts)
        for arg in ('cc1','cc1plus','collect2','libgcc_s.so.1','libstdc++.so','libc.so','Scrt1.o','crt1.o','crti.o','crtn.o'):
            path=Path(clean_command([compiler,'-print-file-name='+arg]).strip())
            if path.is_file(): dependency_paths.add(str(path.resolve()))
    for path in ('/usr/lib/x86_64-linux-gnu/libc.so.6','/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'):
        dependency_paths.add(str(Path(path).resolve()))
    deps={x:sha(x) for x in sorted(dependency_paths)}
    commands=[
        [cc,*cflags,'-shared','-Wl,-z,defs','-Wl,-t',str(P/C),'-o',str(A)],
        [cxx,*cxxflags,'-fPIC','-shared','-DSTRATA_WAIT_CANARY_PROVIDER','-Wl,-t',
         str(P/CXX),'-o',str(L)],
        [cxx,*cxxflags,'-fno-pie','-no-pie','-Wl,-z,lazy','-Wl,-t',
         str(P/CXX),'-L'+str(ROOT),'-Wl,-rpath,'+str(ROOT),'-lstrata_wait_canary','-o',str(E)],
    ]
    save(dict(stage='prepared', active=False, ownership_closed=True, passed=False, complete=False,
              source_commit=source_commit, sources=sources, source_sha=signature(sources),
              dependencies=deps, dependency_sha=signature(deps), native=identify_native(),
              build=commands, environment=ENV, runtime=RUNTIME,
              perf=dict(prctl=Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),
                  command=['/usr/bin/perf','stat','-e','task-clock','--','/usr/bin/true'],
                  result='Access to performance monitoring limited for this account'),
              plan='Build only a diagnostic shared library and a CPU canary. Verify 2 known caller return '
                   'addresses, 3 entries including one propagated C++ exception and 2 recorded returns. '
                   'Compare exactly the same control without LD_AUDIT. No GPU, model or system changes.'))
    print('Prepared host canary recipe. Commit before building.',flush=True)


def current(row):
    require(basis.validation.committed(RESULT,(KEY,))==compact(row)
            and source_pins(row['source_commit'])==row['sources']
            and signature(row['sources'])==row['source_sha']
            and signature(row['dependencies'])==row['dependency_sha']
            and all(sha(path)==digest for path,digest in row['dependencies'].items())
            and identify_native()==row['native'] and row['environment']==ENV
            and row['runtime']==RUNTIME, 'frozen host observer control changed')
    if row.get('artifacts'):
        require(all(sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
                    for v in row['artifacts'].values()), 'observer artifact changed')


def verify_owner(owner, row):
    row['ownership_closed'] = owner.active is None and all(ownership.closed(x) for x in owner.commands)
    row['active'] = bool(owner.active)


def build(row):
    current(row)
    require(row['stage']=='prepared' and not (ROOT/'compile').exists(), 'one prepared CPU-only build')
    owner=ownership.Owner(ROOT/'compile')
    try:
        for i, cmd in enumerate(row['build']):
            c,_,_=owner.run(str(i),cmd,ENV,P,**RUNTIME)
            require(ownership.closed(c) and c['exit_code']==0, 'observer build failed')
        row.update(stage='built', artifacts={f.name:dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)
                                            for f in (A,L,E)})
    except Exception as error:
        row.update(stage='failed', error=repr(error))
    finally:
        verify_owner(owner,row)
        save(row)
    require(row['stage']=='built' and row['ownership_closed'],row.get('error','observer build did not close'))
    print('Built CPU-only audit control. Commit before testing.',flush=True)


def test(row):
    current(row)
    require(row['stage']=='built' and not (ROOT/'host').exists(),'fresh CPU-only canary required')
    symbols={}
    for line in clean_command(['/usr/bin/nm','-n',str(E)]).splitlines():
        fields=line.split()
        if len(fields)==3 and fields[2].startswith(('strata_wait_canary_return','strata_wait_canary_parent_')):
            symbols[fields[2]]=int(fields[0],16)
    require(len(symbols)==3,'exact canary PC evidence required')
    ret=symbols['strata_wait_canary_return']
    owner=ownership.Owner(ROOT/'host')
    rows={}
    try:
        for label, env in [('clean',ENV),('audit',dict(ENV, LD_AUDIT=str(A),
              STRATA_WAIT_AUDIT_RETURN=format(ret,'x'),STRATA_WAIT_AUDIT_PARENT='16'))]:
            command,out,err=owner.run(label,[str(E)],env,P,**RUNTIME)
            output=out.read_text()
            log=err.read_text()
            rows[label]=dict(exit=command.get('exit_code'),output=output.strip(),stderr=log.strip())
            require(ownership.closed(command) and command['exit_code']==0 and output.strip()==GATE,
                    'active loader observer must preserve native return and C++ exception')
        require(not rows['clean']['stderr'],'clean canary stderr differs')
        log=rows['audit']['stderr']
        samples=[(int(n),int(pc,16),int(w),int(c)) for n,pc,w,c in re.findall(
            r'^strata wait site: ordinal=(\d+) caller=([0-9a-f]+) wall_ns=(\d+) cpu_ns=(\d+)$',log,re.M)]
        require(f'strata wait audit: ready=1 return_pc={ret:x} parent_offset=16' in log
                and 'strata wait audit: entered=3 exited=2 rejected=0' in log
                and [(n,pc) for n,pc,_,_ in samples]==[
                    (1,symbols['strata_wait_canary_parent_a']),(3,symbols['strata_wait_canary_parent_b'])]
                and len(log.splitlines())==4 and all(0<=c<=w for _,_,w,c in samples),
                'auditor activation, caller or exception accounting differs')
        row.update(stage='closed',complete=True,passed=True,symbols=symbols)
    except Exception as error:
        row.update(stage='failed',error=repr(error))
    finally:
        row['host_control']=rows
        verify_owner(owner,row)
        save(row)
    require(row['passed'] and row['ownership_closed'],row.get('error','host owner did not close'))
    print('Known CPU-only caller/exception/return controls passed.',flush=True)


if __name__=='__main__':
    require(sys.executable==basis.reference.model.PYTHON,'use pinned Python')
    argument,=sys.argv[1:]
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([])
        if argument=='prepare':
            prepare()
        else:
            {'build':build,'test':test}[argument](json.loads((ROOT/'record.json').read_bytes()))
