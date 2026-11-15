#!/usr/bin/env python3
"""Owned native shape query; independent project code under repository LICENSE.

Reuses qualified source/runtime/owner and exact-link evidence helpers. No kernel,
model run, inference selection, runtime reset or service change is performed.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import build_fused_profile as closure
import relink_qsa_k_model as linking
r, P, B = closure.r, closure.P, closure.B
ROOT = B / 'qsa-matrix-shapes-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'qsa_head_geometry'
SOURCE = 'bench/sycl/matrix_shapes.cpp'
require, digest, write = r.require, r.digest, r.write


def sources(revision, controller=None, extra_sources=()):
    pending, seen, out = [controller or Path(__file__).resolve()], set(), {}
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path); name = str(path.relative_to(P))
        raw = r.build.git(P, 'show', revision + ':' + name)
        require(path.read_bytes() == raw, 'commit exact controller: ' + name)
        out[name] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules = [v.name for v in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file(): pending.append(local)
    for name in [SOURCE, 'tools/sycl/measurement_environment.json', *extra_sources]:
        raw = r.build.git(P, 'show', revision + ':' + name)
        require((P / name).read_bytes() == raw, 'commit exact source: ' + name)
        out[name] = hashlib.sha256(raw).hexdigest()
    return out


def prospective_inputs(compiler, rt, pins):
    allowed = {str(Path(v['path']).resolve()): closure.SYSTEM_HEADER_UPDATES.get(v['path'], (None, v['sha256']))[1]
               for v in json.loads(closure.lineage.SDK.read_text())}
    for directory in [Path('/usr/lib/x86_64-linux-gnu'), Path('/usr/lib/gcc/x86_64-linux-gnu/13')]:
        for pattern in ['*.so', '*.a', '*.o']:
            for path in directory.glob(pattern):
                if path.is_file(): allowed[str(path.resolve())] = digest(path)
    for path in (Path(compiler['path']).parents[1] / 'bin').iterdir():
        if path.is_file(): allowed[str(path.resolve())] = digest(path)
    for path in ['/usr/bin/ld', '/usr/bin/as', '/usr/bin/ar', '/usr/bin/ranlib', '/lib64/ld-linux-x86-64.so.2']:
        allowed[str(Path(path).resolve())] = digest(path)
    allowed.update(rt['runtime_files'])
    allowed.update({str(P / n): h for n,h in pins.items()})
    return allowed


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('source_sha256', 'commands_executed')}
    out['source_inputs'] = dict(files=len(row['source_sha256']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['source_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return out


def admitted(row):
    require(sources(row['source_commit']) == row['source_sha256'], 'source inputs changed')
    require(closure.sdk_inputs() == row['sdk'], 'SDK changed')
    rt = r.validation.committed(r.RESULT)
    require(digest(r.RESULT) == row['runtime_sha256'], 'runtime qualification changed')
    require(r.current(rt)['binary'] == row['selected_binary'], 'selected fallback changed')
    require(digest(row['input_inventory']['path']) == row['input_inventory']['sha256'], 'prospective inputs changed')
    allowed = json.loads(Path(row['input_inventory']['path']).read_text())
    require(len(allowed) == row['input_inventory']['files'], 'input count differs')
    for p,h in allowed.items(): require(digest(p) == h, 'frozen input changed: ' + p)
    if 'binary' in row: require(digest(row['binary']['path']) == row['binary']['sha256'], 'native binary changed')
    return allowed, rt


def prepare():
    require(not ROOT.exists(), 'fresh property generation only; inspect prior owner')
    record = r.validation.committed(RESULT)
    require(KEY not in record, 'no overwrite of prior shape-query result')
    retired = record['qsa_pair_model']['retirement']
    require(retired['verified_absent'] and retired['stage'] == 'retired' and
            all(not Path(p).exists() for p in retired['retired_roots']), 'retire closed extra generation first')
    rt = r.validation.committed(r.RESULT); full = r.current(rt)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
    pins = sources(revision); sdk = closure.sdk_inputs(); compiler = full['compiler']
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    allowed = prospective_inputs(compiler, rt, pins)
    write(ROOT / 'input-inventory.json', allowed)
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    native = dict(rt['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(ROOT / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    command = [compiler['path'], '-std=c++20', '-O2', '-fsycl', '-save-temps=obj', '-v', '-Wl,-t',
        '-MD', '-MF', str(ROOT / 'matrix.d'), str(P / SOURCE), '-o', str(ROOT / 'matrix_shapes')]
    row = dict(stage='prepared', source_commit=revision, source_sha256=pins, sdk=sdk,
        compiler=compiler, runtime_sha256=digest(r.RESULT), selected_binary=full['binary'], boot=rt['boot'],
        input_inventory=dict(path=str(ROOT / 'input-inventory.json'), sha256=digest(ROOT / 'input-inventory.json'), files=len(allowed)),
        build_command=command, build_environment=env, native_environment=native,
        budgets=dict(wall=240, cpu=600, rss_cap=4<<30, text_cap=16<<20, file_cap=128<<20, total_cap=512<<20),
        active=False, ownership_closed=True, passed=False, GPU_kernel_executed=False, inference_adopted=False,
        criterion='Property-only exact B570:report all matrix tuples and actual subgroup/workgroup/local limits. Negative M4 support closes proposal normally;no unsupported kernel submits. Frozen source,SDK,compiler+actual dependencies,link inputs,pre-main validation,immutable providers,finite ownership,no new fault required.',
        retention='One imminent property probe only;selected6e40+12fixtures remain unchanged. Retire property ELF/objects/successful trace after compact capability evidence and exact source/recipe recovery,before the next component generation.')
    admitted(row); write(ROOT / 'record.json', row); record[KEY] = dict(shape_query=compact(row)); write(RESULT, record)
    print('Shape query prepared;commit frozen plan before compile.', flush=True)


def build():
    record = r.validation.committed(RESULT); row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and compact(row) == record[KEY]['shape_query'], 'fresh committed build plan')
    allowed,_ = admitted(row); owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands_executed=owner.commands if owner else []))
    owner = r.supervision.Owner(ROOT / 'build-owned', persist)
    try:
        entry,out,err = owner.run('compile', row['build_command'], row['build_environment'], ROOT, **row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'compile failed: ' + err.read_text()[-2500:])
        observed = {}
        for file in ROOT.glob('*.d'):
            for p in shlex.split(file.read_text().replace('\\\n', ' ').split(':',1)[1]):
                path = Path(p).resolve(); h = digest(path)
                require(allowed.get(str(path)) == h, 'unfrozen actual compiler input: ' + str(path)); observed[str(path)] = h
        require(str(P / SOURCE) in observed and len(observed) >= 200, 'incomplete compiler dependency evidence')
        resolved = linking.traced_inputs(out.read_text(), err.read_text(), allowed, ROOT, ROOT)
        write(ROOT / 'compiler-inputs.json', observed); write(ROOT / 'link-inputs.json', resolved)
        binary = ROOT / 'matrix_shapes'
        row.update(stage='built', build_passed=True, binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary)),
            actual_compiler_inputs=dict(path=str(ROOT / 'compiler-inputs.json'), sha256=digest(ROOT / 'compiler-inputs.json'), files=len(observed)),
            actual_link_inputs=dict(path=str(ROOT / 'link-inputs.json'), sha256=digest(ROOT / 'link-inputs.json'), frozen=len(resolved['frozen_inputs']), uncovered=resolved['uncovered_inputs']))
        admitted(row)
    except BaseException as error: row.update(stage='failed', build_passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
        row['active'] = False
        row['commands_executed'] = owner.commands; write(ROOT / 'record.json', row)
        record[KEY]['shape_query'] = compact(row); write(RESULT, record)
    print(json.dumps({k:row.get(k) for k in ['stage','build_passed','ownership_closed','error']}),flush=True)
    require(row.get('build_passed') and row['ownership_closed'], 'native query build failed')


def observed_native(pid, env, rt, binary):
    proc = Path('/proc',str(pid)); actual = dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(actual.get(k.encode()) == v.encode() for k,v in env.items()), 'child environment changed')
    mapped = r.executable_mappings((proc/'maps').read_text())
    require(mapped <= set(rt['runtime_files']) | {binary['path']}, 'unknown executable provider')
    neo = next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p)
    require(neo in mapped and rt['preload']['loader'] in mapped, 'expected actual driver/provider absent')
    fds=[]
    for file in (proc/'fdinfo').iterdir():
        fields = re.findall(r'^(drm-driver|drm-pdev):\s*(\S+)\s*$',file.read_text(),re.M)
        if fields:
            require(dict(fields)=={'drm-driver':'xe','drm-pdev':'0000:05:00.0'}, 'wrong native device');fds.append(file.name)
    require(fds, 'B570descriptor missing')
    return dict(immutable_environment=True, providers_qualified=True, B570_DRM=True)


def run():
    record = r.validation.committed(RESULT); row = json.loads((ROOT/'record.json').read_text())
    require(row['stage']=='built' and row['build_passed'] and compact(row)==record[KEY]['shape_query'] and not (ROOT/'native-owned').exists(), 'fresh source-bound built query required')
    allowed,rt = admitted(row)
    text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor = next(v.split(': ',1)[1] for v in text.splitlines() if v.startswith('-- cursor: '))
    owner=None;outcome=dict(complete=False,passed=False,host_probe_accepted=False)
    build_commands=list(row.get('commands_executed',[]))
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(row,native_progress=outcome,
            commands_executed=[*build_commands,*(owner.commands if owner else [])]))
    owner=r.supervision.Owner(ROOT/'native-owned',persist)
    seen={}
    try:
        def interact(io,entry):
            require(io.line(30)=='MATRIX_GATE before-runtime','entry gate differs');io.send(b'X')
            require(io.line(60)=='MATRIX_GATE runtime-ready','runtime gate differs')
            seen['start']=observed_native(io.proc.pid,row['native_environment'],rt,row['binary']);io.send(b'X')
            line=io.line(30);require(line.startswith('MATRIX_PROPERTIES '),'property row absent')
            seen['properties']=json.loads(line[len('MATRIX_PROPERTIES '):])
            require(io.line(30)=='MATRIX_GATE before-exit','exit gate differs')
            seen['end']=observed_native(io.proc.pid,row['native_environment'],rt,row['binary']);io.send(b'X');io.proc.stdin.close();io.exit(20)
        entry,out,err=owner.run('properties',[row['binary']['path']],row['native_environment'],ROOT,interaction=interact,**row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code']==0,'native property process failed')
        proof=r.validation.preload_evidence(err.read_text(),entry['pid']);require(proof['loader']==rt['preload']['loader'],'pre-main provider differs')
        warnings=ROOT/'validation.log';require(not warnings.exists() or warnings.stat().st_size==0,'validation warning/error')
        trace=(ROOT/'api.trace').read_text();require('urDeviceGetInfo' in trace and not re.search(r'urEnqueue\w*|urQueueCreate|urUSM\w*Alloc',trace),'property-only trace differs')
        require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|MATRIX_ERROR',err.read_text()),'native runtime error')
        v=seen['properties'];require(v['SG16']==1 and v['bf16_M8N16K16']==1,'known matrix control absent')
        admitted(row)
        outcome.update(complete=True,host_probe_accepted=True,properties=v,observed=seen['end'],parameter_validation=True,
            M4_component_eligible=bool(v['bf16_M4N16K16'] and v['workgroup']>=96 and v['workitem1d']>=96 and v['local_bytes']>=19188+4096),
            kernel_submitted=False,model_adopted=False,ordinary_teardown_qualified=False)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        r.validation.finish(outcome,owner,cursor,row['boot'])
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],active=False,ownership_closed=outcome['ownership_closed'],outcome=outcome,commands_executed=[*build_commands,*owner.commands])
        write(ROOT/'record.json',row);record[KEY]['shape_query']=compact(row);write(RESULT,record)
    print(json.dumps(outcome),flush=True);require(outcome['passed'],'shape query did not qualify')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare,'build':build,'run':run}[sys.argv[1]]()
