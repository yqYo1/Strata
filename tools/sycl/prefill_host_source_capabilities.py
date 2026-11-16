#!/usr/bin/env python3
"""Bounded read-only admission for the next ordinary-RAM upload experiment.

Main-authored, repository LICENSE. No allocation, queue, copy, kernel or import.
Commit source -> prepare -> commit plan -> run. No new native build generation.
"""
import ctypes as C
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
import prefill_upload_overlap_reuse as reference
import ur_validation

P, B, RESULT = reference.P, reference.B, reference.RESULT
ROOT = B/'prefill-host-source-capabilities-v2'
KEY = 'prefill_host_source_capabilities_v2'
SOURCE = 'tools/sycl/prefill_host_source_capabilities.py'
PREFIXES = ['tools/sycl', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md']
component = reference.component
require, sha, signature = reference.require, reference.sha, reference.signature
write = component.write


class Extension(C.Structure):
    _fields_ = [('name', C.c_char*256), ('version', C.c_uint32)]


class Device(C.Structure):
    # Installed ze_api.h:ze_device_properties_t, natural x86-64 C alignment.
    _fields_ = [('stype', C.c_uint32), ('pNext', C.c_void_p)] + [
        (n, C.c_uint32) for n in ('type', 'vendor', 'device', 'flags', 'subdevice', 'clock')
    ] + [('max_allocation', C.c_uint64)] + [
        (n, C.c_uint32) for n in ('contexts', 'priority', 'threads', 'simd', 'eus', 'subslices', 'slices')
    ] + [('timer', C.c_uint64), ('timestamp_bits', C.c_uint32),
         ('kernel_timestamp_bits', C.c_uint32), ('uuid', C.c_ubyte*16), ('name', C.c_char*256)]


def native():
    def gate(name):
        print(json.dumps(dict(kind='gate', name=name)), flush=True)
        require(input() == 'continue', 'Main gate refused')
    require(C.sizeof(Extension) == 260 and C.sizeof(Device) == 368
            and Device.max_allocation.offset == 40 and Device.name.offset == 112,
            'pinned x86-64 LevelZero ABI required')
    gate('before-runtime')
    library = C.CDLL(component.event.LOADER)
    def call(name, types, *args):
        fn = getattr(library, name); fn.argtypes = types; fn.restype = C.c_uint32
        code = fn(*args); require(code == 0, name+' result='+hex(code))
    count = C.c_uint32()
    # Use only the original zeInit/zeDriverGet family; do not mix zeInitDrivers.
    call('zeInit', [C.c_uint32], 1)
    call('zeDriverGet', [C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)], C.byref(count), None)
    require(count.value == 1, 'one unambiguous GPU driver required')
    drivers = (C.c_void_p*1)()
    call('zeDriverGet', [C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)], C.byref(count), drivers)
    require(count.value == 1 and drivers[0], 'driver enumeration changed')
    driver = drivers[0]; api = C.c_uint32()
    call('zeDriverGetApiVersion', [C.c_void_p, C.POINTER(C.c_uint32)], driver, C.byref(api))
    count.value = 0
    call('zeDeviceGet', [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)], driver, C.byref(count), None)
    require(count.value == 1, 'one unambiguous root device required')
    devices = (C.c_void_p*1)()
    call('zeDeviceGet', [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)], driver, C.byref(count), devices)
    require(count.value == 1 and devices[0], 'device enumeration changed')
    properties = Device(stype=3)
    call('zeDeviceGetProperties', [C.c_void_p, C.POINTER(Device)], devices[0], C.byref(properties))
    name = properties.name.decode('utf-8', errors='strict')
    require(properties.type == 1 and properties.vendor == 0x8086 and 'B570' in name
            and properties.device == int(component.guards.driver_guard()['device'], 16), 'actual B570 required')
    count.value = 0
    call('zeDriverGetExtensionProperties', [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(Extension)], driver, C.byref(count), None)
    require(0 < count.value <= 1024, 'bounded extension count required')
    allocated = count.value; extensions = (Extension*allocated)()
    call('zeDriverGetExtensionProperties', [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(Extension)], driver, C.byref(count), extensions)
    require(0 < count.value <= allocated, 'extension enumeration grew')
    all_extensions = {e.name.decode('ascii', errors='strict'): e.version for e in extensions[:count.value]}
    require(len(all_extensions) == count.value, 'unique extension names required')
    external = all_extensions.get('ZE_extension_external_memmap_sysmem', 0)
    print(json.dumps(dict(kind='capabilities', device=name, vendor=properties.vendor, device_id=properties.device,
        api_version=api.value, api_major=api.value >> 16, api_minor=api.value & 65535,
        max_single_allocation_bytes=properties.max_allocation,
        driver_extension_count=count.value, driver_extensions_sha256=signature(all_extensions),
        relevant_extensions={k:v for k,v in all_extensions.items() if re.search('external|memmap|host_pointer', k, re.I)},
        external_sysmem_mapping_version=external,
        read_only_mapping_reported_eligible=bool(external >= (1 << 16) and api.value >= (1 << 16 | 16)),
        mapping_or_transfer_attempted=False)), flush=True)
    gate('after-queries')
    print(json.dumps(dict(kind='done', passed=True)), flush=True)


def executable_maps(pid):
    found = {}
    for line in Path('/proc', str(pid), 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
            path = str(Path(fields[5]).resolve()); found[path] = sha(path)
    return found


def compact(row):
    public = {k:v for k,v in row.items() if k not in ['reference', 'allowed', 'commands', 'project']}
    public.update(reference_sha256=signature(row['reference']),
        allowed=dict(files=len(row['allowed']), sha256=signature(row['allowed'])),
        project={k:v for k,v in row['project'].items() if k != 'source_sha256'})
    return public


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'existing capability job;inspect,no retry')
    basis, proof = reference.verify()
    failed = ur_validation.committed(RESULT, ('prefill_upload_overlap_profile_v2',))
    require(failed['stage'] == 'failed' and not failed['active'] and failed['physical_ownership_closed'],
            'close failed collector cohort before independent capability query')
    prior = ur_validation.committed(RESULT, ('prefill_host_source_capabilities_v1',))
    require(prior['stage'] == 'failed' and not prior['active'] and prior['physical_ownership_closed'],
            'preserve original count-initialization failure before corrected discovery')
    project = component.original.coherent.build.common.source.pin(P, PREFIXES)
    allowed = json.loads(Path(basis['runtime_manifest']['path']).read_text()) | executable_maps(__import__('os').getpid())
    allowed[component.event.VALIDATION] = sha(component.event.VALIDATION)
    env = dict(basis['cases'][0]['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    # Native L0 enumeration only: avoid UR preinit and its zeInitDrivers family.
    for key in ['LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'UR_LOG_TRACING', 'UR_LOG_VALIDATION']:
        env.pop(key, None)
    env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    ROOT.mkdir(mode=0o700)
    for directory in ['tmp', 'cache', 'owned']: (ROOT/directory).mkdir(mode=0o700)
    row = dict(stage='prepared', passed=False, complete=False, active=False, ownership_closed=True,
        project=project, reference=proof, allowed=allowed,
        interpreter=dict(path=str(Path(sys.executable).resolve()), sha256=sha(sys.executable)),
        header=dict(path='/usr/include/level_zero/ze_api.h', sha256=sha('/usr/include/level_zero/ze_api.h')),
        argv=[str(Path(sys.executable).resolve()), '-B', str(P/SOURCE), 'native'], environment=env,
        boot=basis['boot'], driver=basis['driver'], commands=[],
        plan=dict(question='Does the qualified installed B570 driver advertise the standard read-only external-system-memory mapping prerequisite?',
            units='Read-only capability discovery only. No allocation/import/unimport,copy,queue,kernel,PTI,new native generation or change to E2math/wait ordering. Source adaptation and scheduling remain independently compared units.',
            acceptance='One owned normal0 process;exact native argv/environment,committed source/interpreter/header/runtime identities;one Intel B570 driver/device;bounded API/extensions enumeration;before/after provider checks,quiet validation,unchanged boot/driver,no fresh fault. An absent extension is a discovery result,not a transfer test.',
            budget='30s wall,30sCPU,1GiB RSS,4MiB text/file,16MiB total;stop first mismatch/failure,no retry or GPUreset.',
            next='If reported API>=1.16 and external-map extension>=1.0,plan a bounded same-context read-only import and event-lifetime fixture. Otherwise use the documented bounded host-USM staging baseline;do not relax pointer admission or assume vendor raw-RAM accessibility.',
            retention='Keep the compact feature result and exact source/header/runtime IDs. Retire routine provider/log output after closure;no performance CSVrow or model/adoption claim. Existing E2component generation receives no extended lifetime from this read-only query.'))
    save(row); print('Prepared zero-command source-contract discovery;commit plan before run.', flush=True)


def run():
    row = json.loads((ROOT/'record.json').read_text())
    expected = ur_validation.committed(RESULT, (KEY,))
    require(row['stage'] == 'prepared' and not row['active'], 'fresh committed capability plan required')
    pin = component.original.coherent.build.common.source.pin(P, PREFIXES)
    require(pin['source_sha256'] == row['project']['source_sha256'], 'frozen capability sources changed')
    for group in [row['interpreter'], row['header']]: require(sha(group['path']) == group['sha256'], 'interpreter/header changed')
    require(all(sha(p) == h for p,h in row['allowed'].items()), 'native provider changed')
    require(signature(compact(row)) == signature(expected), 'typed capability plan differs from committed evidence')
    require(row['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and row['driver'] == component.guards.driver_guard(), 'boot/driver changed before discovery')
    journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15)
    cursor = journal.split('-- cursor: ', 1)[1].strip()
    owner = component.owner_api.Owner(ROOT/'owned', lambda:write(ROOT/'ownership.json', owner.commands))
    try:
        row.update(stage='running', active=True, ownership_closed=False); save(row)
        gates, records, providers = [], [], []
        def interact(channel, entry):
            while True:
                item = json.loads(channel.line(20))
                if item['kind'] == 'gate':
                    require(item['name'] == ['before-runtime','after-queries'][len(gates)], 'capability gate order differs')
                    actual = dict(v.split('=',1) for v in Path('/proc',str(channel.proc.pid),'environ').read_text().strip('\0').split('\0'))
                    require(actual == row['environment'], 'actual pre-exec environment differs')
                    require(Path('/proc',str(channel.proc.pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0') == row['argv'], 'actual native argv differs')
                    mapped = executable_maps(channel.proc.pid)
                    require(all(row['allowed'].get(p) == h for p,h in mapped.items()), 'unknown native provider')
                    if item['name'] == 'after-queries':
                        for path in [component.event.LOADER, component.event.VALIDATION, row['environment']['ZE_ENABLE_ALT_DRIVERS']]:
                            require(str(Path(path).resolve()) in mapped, 'required initialized LevelZero provider absent')
                    providers.append(dict(gate=item['name'], files=len(mapped), sha256=signature(mapped)))
                    gates.append(item['name']); channel.send(b'continue\n')
                else:
                    records.append(item)
                    if item['kind'] == 'done': channel.exit(15); break
        entry, out, err = owner.run('capabilities', row['argv'], row['environment'], P,
            interaction=interact, wall=30, cpu=30, rss_cap=1<<30, text_cap=4<<20, file_cap=4<<20, total_cap=16<<20)
        require(component.owner_api.closed(entry) and entry['exit_code'] == 0, 'capability child failed:'+str(err))
        require(gates == ['before-runtime','after-queries'] and len(records) == 2
                and records[0]['kind'] == 'capabilities' and records[0]['mapping_or_transfer_attempted'] is False
                and records[1] == dict(kind='done',passed=True), 'incomplete capability response')
        require(not re.search(r'\[(?:error|critical|warning)\]', err.read_text(), re.I), 'LevelZero validation/loader diagnostic')
        row.update(stage='closed', passed=True, complete=True, capabilities=records[0], providers=providers,
            model_trial_allowed=False, mapping_or_transfer_qualified=False)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error)); raise
    finally:
        row['commands'].extend(owner.commands); row['active'] = owner.active is not None
        row['ownership_closed'] = owner.active is None and all(component.owner_api.closed(c) for c in row['commands'])
        try:
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'], text=True, timeout=15)
            row['fresh_faults'] = [s for s in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)', s, re.I)]
            row['boot_unchanged'] = row['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            require(not row['fresh_faults'] and row['boot_unchanged'] and row['driver'] == component.guards.driver_guard(), 'post-discovery health mismatch')
        except BaseException as error:
            row.update(stage='failed', passed=False, complete=False, final_health_error=repr(error))
        save(row)
    require(row['passed'], 'capability discovery failed;inspect existing receipt,no retry')
    print(json.dumps({k:row[k] for k in ['stage','passed','ownership_closed','capabilities']}), flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in [['native'],['prepare'],['run']], 'native|prepare|run')
    if sys.argv[1] == 'native': native()
    else:
        with (B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); component.guards.no_model_owners([])
            prepare() if sys.argv[1] == 'prepare' else run()
