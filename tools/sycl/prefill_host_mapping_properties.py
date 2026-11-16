#!/usr/bin/env python3
"""Observe native allocation properties across a read-only host mapping.

Main-authored, repository LICENSE. No queue, copy, kernel or source admission.
Owned capability observer at822f52c0c is the foundation; no external code copied.
Commit source -> prepare -> commit plan -> run. No new native build generation.
"""
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
import coherent_original_reuse as reference
import upload_event_lane as component
import ur_validation

P, B, RESULT = reference.P, reference.B, component.RESULT
ROOT = B/'prefill-host-mapping-properties-v1'
KEY = 'prefill_host_mapping_properties_v1'
SOURCE = 'tools/sycl/prefill_host_mapping_properties.py'
PREFIXES = ['tools/sycl', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md']
require, sha, signature = reference.require, reference.sha, reference.fingerprint
write = component.write


class Descriptor(C.Structure):
    _fields_ = [('stype', C.c_uint32), ('pNext', C.c_void_p), ('flags', C.c_uint32)]


class External(C.Structure):
    _fields_ = [('stype', C.c_uint32), ('pNext', C.c_void_p),
                ('memory', C.c_void_p), ('size', C.c_size_t)]


class Allocation(C.Structure):
    _fields_ = [('stype', C.c_uint32), ('pNext', C.c_void_p),
                ('type', C.c_uint32), ('id', C.c_uint64), ('page', C.c_uint64)]


def mapping_properties(library, driver, call):
    # x86-64 layouts/constants are from the pinned installed ze_api.h. The
    # native guard executes before allocation. This is discovery, not a lane.
    require(C.sizeof(Descriptor) == 24 and Descriptor.flags.offset == 16
            and C.sizeof(External) == 32 and External.memory.offset == 16
            and C.sizeof(Allocation) == 40 and Allocation.type.offset == 16
            and Allocation.id.offset == 24 and Allocation.page.offset == 32,
            'pinned native descriptor/property ABI required')
    libc = C.CDLL('libc.so.6', use_errno=True)
    libc.posix_memalign.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_size_t]
    libc.posix_memalign.restype = C.c_int
    libc.mprotect.argtypes = [C.c_void_p, C.c_size_t, C.c_int]
    libc.mprotect.restype = C.c_int
    libc.free.argtypes = [C.c_void_p]; libc.free.restype = None
    context, heap, mapped = C.c_void_p(), C.c_void_p(), C.c_void_p()
    page = os.sysconf('SC_PAGESIZE')
    require(0 < page <= 65536, 'bounded actual host page size required')
    imported = protected = False
    result = None
    def snapshot():
        props = Allocation(stype=0x17)
        call('zeMemGetAllocProperties', [C.c_void_p, C.c_void_p,
             C.POINTER(Allocation), C.POINTER(C.c_void_p)], context, heap, C.byref(props), None)
        return dict(type=props.type, allocation_id=props.id, page_bytes=props.page)
    try:
        desc = Descriptor(stype=0xd)
        call('zeContextCreate', [C.c_void_p, C.POINTER(Descriptor), C.POINTER(C.c_void_p)],
             driver, C.byref(desc), C.byref(context))
        require(context.value, 'non-null native context required')
        require(libc.posix_memalign(C.byref(heap), page, page) == 0 and heap.value
                and heap.value % page == 0, 'whole-page heap allocation required')
        pattern = bytes((i * 13 + 7) & 255 for i in range(page))
        C.memmove(heap, pattern, page)
        before = snapshot()
        require(before['type'] == 0, 'ordinary heap must not already be a native allocation')
        require(libc.mprotect(heap, page, 1) == 0, 'read-only heap protection failed')
        protected = True
        external = External(stype=0x00020037, memory=heap.value, size=page)
        host = Descriptor(stype=0x16, pNext=C.addressof(external), flags=16)
        # Do not infer the resulting property value from a different source
        # release, and do not admit this pointer to a copy/kernel API.
        call('zeMemAllocHost', [C.c_void_p, C.POINTER(Descriptor), C.c_size_t,
             C.c_size_t, C.POINTER(C.c_void_p)], context, C.byref(host), page, 0, C.byref(mapped))
        imported = bool(mapped.value)
        require(imported and mapped.value == heap.value, 'mapping must preserve the live heap address')
        during = snapshot()
        base, extent = C.c_void_p(), C.c_size_t()
        fn = library.zeMemGetAddressRange
        fn.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
        fn.restype = C.c_uint32
        range_code = fn(context, heap, C.byref(base), C.byref(extent))
        call('zeMemFree', [C.c_void_p, C.c_void_p], context, mapped)
        imported = False
        after = snapshot()
        unchanged = C.string_at(heap, page) == pattern
        require(unchanged and after['type'] == 0, 'unimport must preserve the heap and remove its native registration')
        result = dict(kind='mapping-properties', bytes=page, read_only=True,
            same_address=True, before=before, during=during, after=after,
            address_range=dict(result=range_code, same_base=base.value == heap.value,
                               bytes=extent.value), original_heap_unchanged=True,
            explicit_native_context=True, production_runtime_context_qualified=False,
            allocation_and_unimport_executed=True, application_commands=0,
            source_admission_or_transfer_qualified=False)
    finally:
        # No commands exist. A failed release must not be followed by freeing
        # storage still registered with the device; fail-stop this owned child.
        if imported:
            try: call('zeMemFree', [C.c_void_p, C.c_void_p], context, mapped)
            except BaseException: os._exit(93)
        if protected and libc.mprotect(heap, page, 3) != 0: os._exit(94)
        if heap.value: libc.free(heap)
        if context.value:
            try: call('zeContextDestroy', [C.c_void_p], context)
            except BaseException: os._exit(95)
    return result


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
    library = C.CDLL(component.LOADER)
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
    require(external >= (1 << 16) and api.value >= (1 << 16 | 16),
            'reported standard external/read-only mapping prerequisite required')
    print(json.dumps(mapping_properties(library, driver, call)), flush=True)
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
    _, proof = reference.verify_original()
    basis = ur_validation.committed(RESULT, ('prefill_copy_engine_v6',))
    failed = ur_validation.committed(RESULT, ('prefill_mapped_source_v1',))
    require(failed['stage'] == 'failed' and not failed['active']
            and failed['failure_closure']['physical_process_closed'] is True
            and failed['retirement']['stage'] == 'complete'
            and failed['retirement']['all_selected_paths_absent'] is True,
            'preserve/close/retire failed source cohort before independent property query')
    prior = ur_validation.committed(RESULT, ('prefill_host_source_capabilities_v2',))
    require(prior['stage'] == 'closed' and prior['passed'] is True
            and prior['ownership_closed'] is True and prior['active'] is False
            and prior['capabilities']['api_version'] == 65553,
            'original closed API1.17 discovery required')
    project = component.original.coherent.build.common.source.pin(P, PREFIXES)
    allowed = json.loads(Path(basis['runtime_manifest']['path']).read_text()) | executable_maps(__import__('os').getpid())
    allowed[component.VALIDATION] = sha(component.VALIDATION)
    env = dict(basis['observer_environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    # Native L0 enumeration only: avoid UR preinit and its zeInitDrivers family.
    for key in ['LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'UR_LOG_TRACING', 'UR_LOG_VALIDATION']:
        env.pop(key, None)
    env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    ROOT.mkdir(mode=0o700)
    for directory in ['tmp', 'cache', 'owned']: (ROOT/directory).mkdir(mode=0o700)
    row = dict(stage='prepared', passed=False, complete=False, active=False, ownership_closed=True,
        model_trial_allowed=False,performance_qualified=False,
        source_admission_qualified=False,mapping_or_transfer_qualified=False,
        project=project, reference=proof, allowed=allowed,
        interpreter=dict(path=str(Path(sys.executable).resolve()), sha256=sha(sys.executable)),
        header=dict(path='/usr/include/level_zero/ze_api.h', sha256=sha('/usr/include/level_zero/ze_api.h')),
        argv=[str(Path(sys.executable).resolve()), '-B', str(P/SOURCE), 'native'], environment=env,
        boot=basis['boot'], driver=basis['driver'], commands=[],
        parent_failure=dict(archive=failed['archive'],failure_closure=failed['failure_closure']),
        capability_identity=signature(prior['capabilities']),
        plan=dict(question='What numeric allocation type,id,page size and address range does this exact installed NEO report before,during and after standard read-only external heap mapping?',
            units='Independent API-property discovery after stopped U3. One host page from posix_memalign,CPUpattern initialization,mprotect read-only,native explicit context,standard external descriptor with read-only flag,allocation queries,unimport,original-heap verification and free. No queue,event,copy,kernel,PTI,source-admission change,new native generation,engine math or scheduling change.',
            acceptance='One owned normal0 process;exact committed source/interpreter/header/runtime/typed plan and actual argv/environment;one B570 driver/device;before/after provider checks;quiet validation,unchanged boot/driver/no fresh fault. Assert original unknown heap,same-address allocation,successful unimport/removal and unchanged original heap. Observe numeric registered type and range result without accepting them for GPU use. Source/property/model qualification stays false.',
            budget='30s wall,30sCPU,1GiB RSS,4MiB text/file,16MiB total;first mismatch/failure stops,no retry/reset. No model access or performance timing.',
            interpretation='Official API1.17.24 EXT_ExternalMemMap and installed header require HOST_IMPORTED=4. Referenced Intel source at724697d73 expects HOST=1 but is not proven installed-package source. Record actual numeric evidence before any new contract/compatibility plan;do not infer general conformance,rejection or pointer accessibility.',
            next='Reconcile observed numeric type/range/lifecycle with the documented API and pinned provider before a fresh source-adapter admission plan. Any compatibility exception must keep exact owner/context/range/id/lifetime proof and unknown-pointer refusal. Never alter the original failed outcome.',
            retention='Keep one compact property observation and exact source/header/runtime/provider IDs. Retire routine trace/transaction output and this one-question observer after the compact source decision;no CSVspeed row or new native generation. The failed U3build is already retired;both phase references stay unchanged.'))
    save(row); print('Prepared zero-command mapping-property discovery;commit plan before run.', flush=True)


def run():
    row = json.loads((ROOT/'record.json').read_text())
    expected = ur_validation.committed(RESULT, (KEY,))
    require(row['stage'] == 'prepared' and not row['active'], 'fresh committed capability plan required')
    pin = component.original.coherent.build.common.source.pin(P, PREFIXES)
    require(pin['source_sha256'] == row['project']['source_sha256'], 'frozen capability sources changed')
    reference.verify_original(expected=row['reference'])
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
                        for path in [component.LOADER, component.VALIDATION, row['environment']['ZE_ENABLE_ALT_DRIVERS']]:
                            require(str(Path(path).resolve()) in mapped, 'required initialized LevelZero provider absent')
                    providers.append(dict(gate=item['name'], files=len(mapped), sha256=signature(mapped)))
                    gates.append(item['name']); channel.send(b'continue\n')
                else:
                    records.append(item)
                    if item['kind'] == 'done': channel.exit(15); break
        entry, out, err = owner.run('mapping-properties', row['argv'], row['environment'], P,
            interaction=interact, wall=30, cpu=30, rss_cap=1<<30, text_cap=4<<20, file_cap=4<<20, total_cap=16<<20)
        require(component.owner_api.closed(entry) and entry['exit_code'] == 0, 'capability child failed:'+str(err))
        require(gates == ['before-runtime','after-queries'] and len(records) == 3
                and records[0]['kind'] == 'capabilities' and records[0]['mapping_or_transfer_attempted'] is False
                and signature(records[0]) == row['capability_identity']
                and records[1]['kind'] == 'mapping-properties'
                and records[1]['allocation_and_unimport_executed'] is True
                and records[1]['application_commands'] == 0
                and records[1]['source_admission_or_transfer_qualified'] is False
                and records[2] == dict(kind='done',passed=True), 'incomplete mapping property response')
        require(not re.search(r'\[(?:error|critical|warning)\]', err.read_text(), re.I), 'LevelZero validation/loader diagnostic')
        require(not re.search(r' in ze(?:CommandList|CommandQueue|Kernel|Event)\w*\(',err.read_text()),
                'unexpected application command/queue/event/kernel API')
        pin_after = component.original.coherent.build.common.source.pin(P, PREFIXES)
        require(pin_after['source_sha256'] == row['project']['source_sha256'], 'source changed during discovery')
        require(all(sha(p) == h for p,h in row['allowed'].items()), 'provider changed during discovery')
        require(all(sha(v['path']) == v['sha256'] for v in [row['header'],row['interpreter']]),
                'header/interpreter changed during discovery')
        row.update(stage='closed', passed=True, complete=True, capabilities=records[0],
            properties=records[1],providers=providers,model_trial_allowed=False,
            source_admission_qualified=False,mapping_or_transfer_qualified=False,performance_qualified=False)
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
    print(json.dumps({k:row[k] for k in ['stage','passed','ownership_closed','properties']}), flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in [['native'],['prepare'],['run']], 'native|prepare|run')
    if sys.argv[1] == 'native': native()
    else:
        with (B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); component.guards.no_model_owners([])
            prepare() if sys.argv[1] == 'prepare' else run()
