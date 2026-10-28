#!/usr/bin/env python3
"""Main-owned property-only admission for the direct-K component.

Own b0b863cc9 loader/device/ownership checks;repository LICENSE applies.
This controller does not enqueue kernels or qualify numerical results/speed.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_qsa_k_loop as b
import measurement_guards as guards
import measurement_owner as supervision
import runtime_probe as runtime
import ur_validation as validation
from measure_down_scatter import postchecks
P,B,ROOT,RESULT=b.P,b.B,b.ROOT,b.RESULT
PROPERTY='native_properties'

def inputs(revision):
    pending = [Path(__file__).resolve()]; seen = set(); pins = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = b.gen.git('show', revision + ':' + name)
        guards.require(raw == path.read_bytes(), 'commit exact controller input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return pins


def single_device_inventory(rows, expected):
    """This workstation's explicit one-eligible-Intel-device constraint."""
    eligible=[r for r in rows if r['vendor']=='0x8086' and
        int(r['class'],16)>>16 in (0x03,0x12)]
    guards.require(eligible==[expected],'single qualified Intel GPU/accelerator inventory required')
    return dict(name='Intel(R) Arc(TM) B570 Graphics',eligible_Intel_PCI_devices=eligible,
        scope='Single eligible physical Intel GPU/accelerator plus exact native device name;not a general multi-GPU UUID mapping.')


def device_inventory(rt):
    rows=[]
    for path in sorted(Path('/sys/bus/pci/devices').iterdir()):
        rows.append(dict(pci=path.name,vendor=(path/'vendor').read_text().strip(),
            device=(path/'device').read_text().strip(),**{'class':(path/'class').read_text().strip()}))
    driver=rt['driver']
    guards.require(driver['vendor']=='0x8086' and driver['device']=='0xe20c' and
        driver['driver']=='xe','qualified runtime is not the B570')
    expected=dict(pci=Path(driver['pci']).name,vendor=driver['vendor'],device=driver['device'],
        **{'class':'0x030000'})
    return single_device_inventory(rows,expected)


def device_fds(name, texts, policy):
    guards.require(name==policy['name'],'native selected another device')
    matched=0
    expected=policy['eligible_Intel_PCI_devices'][0]['pci']
    for text in texts:
        pairs=re.findall(r'^(drm-driver|drm-pdev):\s*(\S+)\s*$',text,re.M)
        if not pairs:
            continue
        guards.require(len(pairs)==2 and dict(pairs)=={'drm-driver':'xe','drm-pdev':expected},
            'owned child has an unqualified or incomplete DRM device binding')
        matched+=1
    guards.require(matched>0,'owned child has no qualified DRM device binding')
    return dict(name=name,pci=expected,driver='xe',matching_DRM_descriptors=matched)


def live_device(pid, name, policy, rt):
    guards.require(device_inventory(rt)==policy,'eligible device inventory changed')
    native=Path('/proc',str(pid),'fdinfo')
    return device_fds(name,[p.read_text() for p in native.iterdir()],policy)


def loader_bindings(text, observed):
    """Correlate glibc namespace/object evidence with the live mapped loader."""
    path = observed['path']
    bindings = re.findall(r"binding file (\S+) \[(\d+)\] to (\S+) \[(\d+)\]:[^\n]*"
        r"symbol [`'](zeDeviceGetComputeProperties|zeKernelGetProperties)['`]", text)
    guards.require(bindings and {r[4] for r in bindings} ==
        {'zeDeviceGetComputeProperties', 'zeKernelGetProperties'}, 'missing loader symbol bindings')
    for _, source_namespace, target, namespace, _ in bindings:
        guards.require(source_namespace == namespace == '0' and
            str(Path(target).resolve()) == path, 'property symbol resolved to another loader object')
    objects = re.findall(r'file=(\S*libze_loader\.so[^\s]*) \[(\d+)\]', text)
    guards.require(objects and all(namespace == '0' and
        (not name.startswith('/') or str(Path(name).resolve()) == path)
        for name, namespace in objects), 'multiple loader objects/namespaces')
    return dict(observed, namespace=0, symbols=sorted({r[4] for r in bindings}),
        evidence='Live executable map plus canonical namespace-0 glibc object/dlsym bindings;helper requires RTLD_NOLOAD. This is this child only,not a general interop guarantee.')


def admission(record, plan):
    source = record[b.KEY]
    native = source['native_build']
    build = native['outcome']
    guards.require(native['stage'] == 'closed' and build['passed'] and build['complete'] and
        build['ownership_closed'] and not build['active'], 'closed native build required')
    b.admit(record, native)
    for asset in build['binaries'].values():
        guards.require(b.digest(asset['path']) == asset['sha256'], 'component executable changed')
    guards.require(b.kernel_ids(build['device_IR']) == build['attention_kernel_ids'] == plan['kernel_ids']
        and inputs(plan['source_commit']) == plan['controller_inputs'], 'controller or compiled layout changed')
    rt = validation.committed(runtime.RESULT)
    runtime.current(rt)
    guards.require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and rt['fresh_faults'] == []
        and rt['runtime_files'] == plan['runtime_files'] and rt['preload'] == plan['preload']
        and device_inventory(rt) == plan['physical_device'], 'runtime qualification changed')
    return rt


def prepare():
    record = b.gen.committed(RESULT)
    source = record[b.KEY]
    guards.require(PROPERTY not in source, 'fresh property plan only;no retry')
    build = source['native_build']['outcome']
    folder = ROOT / 'properties'
    guards.require(not folder.exists(), 'property output already exists')
    rt = validation.committed(runtime.RESULT)
    names = b.kernel_ids(build['device_IR'])
    loader = [p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')]
    guards.require(len(loader) == 1, 'one qualified Level Zero loader required')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    env = dict(validation.committed(runtime.SETTINGS)['environment'],
        TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'),
        LD_PRELOAD=rt['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        LD_DEBUG='files,bindings',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
    plan = dict(stage='prepared', source_commit=revision, controller_inputs=inputs(revision),
        kernel_ids=names, runtime_files=rt['runtime_files'], preload=rt['preload'],
        physical_device=device_inventory(rt), environment=env,
        command=[build['binaries']['fixture']['path'], '--admit', *names, loader[0]],
        budgets=dict(wall=120, cpu=300, rss_cap=4<<30, text_cap=32<<20,
                     file_cap=32<<20, total_cap=128<<20),
        criteria='One owned property-only child. Three exact attention kernels pass WG128/SG16/18args and conservative local-memory checks. Live B570 DRM descriptors,immutable preexec environment,pinned code maps,namespace0 loader/dlsym and pre-main UR validation. Query trace with zero kernel enqueue,no runtime/validation errors,no fresh GPU faults,complete physical process closure. No numerical,speed,occupancy,actual bound-local-argument or model qualification.',
        retention='Retain compact queried properties and fault/closure outcome;retire successful API/loader logs after verification. Native generation remains only for imminent bounded smoke/boundary/oracle and frozen performance decision.')
    admission(record, plan)
    folder.mkdir(mode=0o700)
    for name in ['tmp', 'cache']: (folder/name).mkdir()
    b.write(folder/'plan.json', plan)
    source[PROPERTY] = plan
    b.write(RESULT, record)
    print('Prepared property-only plan;commit before run.')


def run():
    record = b.gen.committed(RESULT)
    plan = record[b.KEY][PROPERTY]
    folder = ROOT/'properties'
    guards.require(plan['stage'] == 'prepared' and plan == json.loads((folder/'plan.json').read_text())
        and not (folder/'owned.json').exists(), 'fresh committed property plan required')
    rt = admission(record, plan)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
                                  text=True, timeout=15)
    cursor = re.findall(r'^-- cursor: (.+)$',text,re.M)
    guards.require(len(cursor) == 1, 'fresh fault cursor unavailable')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
                  device_initialized=None, kernel_submissions=None, model_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        b.write(folder/'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = supervision.Owner(folder/'owned',persist)
    try:
        observed = {}
        def interact(io, entry):
            guards.require(io.line(30) == 'admission_stage=before-runtime', 'unexpected property entry')
            name = plan['physical_device']['name']
            guards.require(io.line(60) == 'admission_stage=runtime-ready device='+name,
                           'property runtime selected another device')
            result['device_identity'] = live_device(io.proc.pid,name,plan['physical_device'],rt)
            native = Path('/proc',str(io.proc.pid))
            actual = dict(v.split(b'=',1) for v in (native/'environ').read_bytes().split(b'\0') if b'=' in v)
            guards.require(all(actual.get(k.encode()) == v.encode() for k,v in plan['environment'].items()),
                           'property environment changed')
            mapped_text = (native/'maps').read_text()
            mapped = runtime.executable_mappings(mapped_text)
            allowed = set(rt['runtime_files']) | {str(Path(plan['command'][0]).resolve())}
            guards.require(mapped <= allowed, 'unknown property code mapping: '+str(sorted(mapped-allowed)))
            loader = plan['command'][-1]
            guards.require(loader in mapped and str(Path(plan['environment']['UR_ADAPTERS_FORCE_LOAD']).resolve()) in mapped,
                           'qualified adapter/loader not mapped')
            rows = [line.split(maxsplit=5) for line in mapped_text.splitlines()]
            rows = [r for r in rows if len(r)==6 and 'x' in r[1] and r[5].startswith('/')
                    and str(Path(r[5]).resolve()) == loader]
            guards.require(len(rows)==1, 'one mapped executable loader object required')
            row = rows[0]
            observed.update(path=loader,sha256=rt['runtime_files'][loader],device=row[3],
                            inode=int(row[4]),executable_start=row[0].split('-')[0],file_offset=row[2])
            while True:
                line = io.line(60)
                if line.startswith('admission_status='):
                    guards.require(line.startswith('admission_status=PASS '),'native property refusal')
                    break
            io.proc.stdin.close();io.exit(20)
        entry,out,err = owner.run('properties',plan['command'],plan['environment'],P,
                                  interaction=interact,**plan['budgets'])
        stdout,stderr = out.read_text(),err.read_text()
        result['device_initialized'] = 'admission_stage=runtime-ready ' in stdout
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'property child failed or did not close')
        proof = validation.preload_evidence(stderr,entry['pid'])
        result['loader_object'] = loader_bindings(stderr,observed)
        guards.require(proof['loader'] == rt['preload']['loader'],'UR provider mismatch')
        body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
        guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body),
                       'native runtime error')
        warnings = folder/'validation.log'
        guards.require(not warnings.exists() or warnings.stat().st_size==0,'validation warning/error')
        trace = (folder/'api.trace').read_text()
        guards.require('urKernelGet' in trace and not re.search(r'urEnqueue\w*Kernel',trace),
                       'query trace absent or unexpected kernel submission')
        rows = []
        for line in stdout.splitlines():
            if line.startswith('arm='):
                fields = dict(part.split('=',1) for part in line.split())
                guards.require(fields.pop('status')=='PASS','compiled resource check failed')
                rows.append({key:int(value) for key,value in fields.items()})
        guards.require([r['arm'] for r in rows]==[0,1,2] and
            stdout.count('admission_status=PASS kernel_submissions=0 ')==1,'property coverage incomplete')
        result.update(passed=True,complete=True,kernel_submissions=0,properties=rows,validation=proof)
        admission(record,plan)
    except BaseException as error:
        result.update(passed=False,complete=False,error=repr(error))
    finally:
        result['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(result,boot,cursor[0]);persist()
        plan['stage'] = 'closed' if result['passed'] else 'failed'
        plan['outcome'] = result
        b.write(RESULT,record)
    print(json.dumps(result),flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sys.exit({'prepare':prepare,'run':run}[sys.argv[1]]())
