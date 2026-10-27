#!/usr/bin/env python3
"""Bounded offline association/disassembly of the captured QSA images.

Main-authored; repository LICENSE. Uses Intel's public MIT ocloc1.0 API.
Memory output preserves the captured ELF and avoids overlong kernel filenames.
"""
import ctypes as ct
import fcntl
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_kernel_image as image
b = image.b
P, RESULT = b.P, b.RESULT
ROOT = image.ROOT / 'offline'
KEY = image.KEY
SELF = str(Path(__file__).resolve().relative_to(P))
require, fingerprint = image.require, image.fingerprint


def sources(revision, record):
    shared = image.controller(revision, record)
    pins = {}
    for path in [SELF, image.SELF]:
        raw = b.gen.git('show', revision + ':' + path)
        require(raw == (P / path).read_bytes(), 'commit exact offline controller: ' + path)
        pins[path] = hashlib.sha256(raw).hexdigest()
    return dict(files=pins, shared_source_files=shared['shared_source_files'],
        shared_source_sha256=shared['shared_source_sha256'])


def captured(record):
    result = record[KEY]['extraction_check']
    require(result['passed'] and result['complete'] and result['ownership_closed'] and
        not result['active'] and result['fresh_faults'] == [] and result['fresh_native_getters_passed'] and
        result['physical_device']['pci'] == '0000:05:00.0', 'qualified actual native captures required')
    require([x['arm'] for x in result['captures']] == [0, 1, 2], 'three ordered captured arms required')
    for entry in result['captures']:
        for key in ['kernel', 'module']:
            row = entry[key]
            require(b.digest(row['path']) == row['sha256'] and Path(row['path']).stat().st_size == row['bytes'],
                'captured native bytes changed')
    return result


def sections(raw):
    require(len(raw) >= 64 and raw[:7] == b'\x7fELF\x02\x01\x01', 'ELF64LE required')
    h = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    require(h[1] == 1 and h[2] == 205 and h[3] == 1 and h[8] == h[11] == 64 and
        0 < h[12] <= 1024 and h[13] < h[12] and 64 <= h[6] <= len(raw) - 64 * h[12],
        'captured IntelGT relocatable ELF header differs')
    headers = [struct.unpack_from('<IIQQQQIIQQ', raw, h[6] + i * 64) for i in range(h[12])]
    string = headers[h[13]]
    require(string[1] == 3 and string[4] + string[5] <= len(raw), 'ELF string table bounds')
    names = raw[string[4]:string[4] + string[5]];result = []
    for i, sec in enumerate(headers):
        require(sec[0] < len(names), 'ELF section name offset')
        end = names.find(b'\0', sec[0]);require(end >= sec[0], 'ELF section name termination')
        name = names[sec[0]:end].decode('utf-8')
        require(sec[1] == 8 or sec[4] + sec[5] <= len(raw), 'ELF section payload bounds')
        result.append(dict(index=i, name=name, type=sec[1], flags=sec[2], offset=sec[4], size=sec[5]))
    return result


def associate(entry):
    raw = Path(entry['module']['path']).read_bytes();gen = Path(entry['kernel']['path']).read_bytes()
    rows = sections(raw);name = '.text.' + entry['kernel']['name']
    text = [s for s in rows if s['name'] == name]
    require(len(text) == 1 and text[0]['type'] == 1 and text[0]['flags'] & 4, 'one exact executable text section')
    text = text[0]
    require(raw[text['offset']:text['offset'] + text['size']] == gen, 'GEN differs from exact module text')
    ze = [s for s in rows if s['name'] == '.ze_info']
    require(len(ze) == 1, 'one ZeInfo section required')
    metadata = raw[ze[0]['offset']:ze[0]['offset'] + ze[0]['size']].decode('utf-8').rstrip('\0')
    blocks = list(re.finditer(r'^  - name:\s*(.*?)\s*$', metadata, re.M))
    selected = [i for i, match in enumerate(blocks) if match[1].strip('\"\'') == entry['kernel']['name']]
    require(len(selected) == 1, 'one selected ZeInfo kernel block required')
    index = selected[0];stop = blocks[index + 1].start() if index + 1 < len(blocks) else len(metadata)
    block = metadata[blocks[index].start():stop]
    return dict(machine=205, type=1, text_section_index=text['index'], offset=text['offset'],
        bytes=text['size'], sha256=hashlib.sha256(gen).hexdigest(), GEN_equals_module_text=True), block


def prepare():
    record = b.gen.committed(RESULT);data = record[KEY];capture = captured(record)
    require('offline_plan' not in data and not ROOT.exists(), 'fresh offline disassembly only;no retry')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip();pins = sources(revision, record)
    qualification_ref = b.gen.git('log', '-1', '--format=%H', '--', str(RESULT.relative_to(P))).decode().strip()
    old = json.loads(b.gen.git('show', qualification_ref + ':' + str(RESULT.relative_to(P))))[KEY]['extraction_check']
    require(old == capture, 'original committed capture identity changed')
    library = Path('/usr/lib/x86_64-linux-gnu/libocloc.so').resolve()
    iga = Path('/usr/lib/x86_64-linux-gnu/libiga64.so.2').resolve()
    require(library.is_file() and iga.is_file(), 'installed offline disassembler libraries absent')
    ROOT.mkdir(mode=0o700)
    host = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'}
    deps = b.qualified_runtime.executable_mappings(Path('/proc/self/maps').read_text())
    deps |= {str(library), str(iga), str(Path(sys.executable).resolve()), '/usr/bin/ldd'}
    owner = b.supervision.Owner(ROOT / 'discover-owned')
    for label, binary in [('ocloc', library), ('iga', iga)]:
        entry, out, err = owner.run(label, ['/usr/bin/ldd', str(binary)], host, P,
            wall=20, cpu=20, rss_cap=1 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20)
        require(b.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
            'offline dependency discovery failed')
        for path in re.findall(r'(?:=>\s+)?(/\S+)\s+\(0x[0-9a-f]+\)', out.read_text()):
            deps.add(str(Path(path).resolve()))
    reference = Path('/home/yayoi/ghq/github.com/intel/compute-runtime')
    names = ['shared/offline_compiler/source/ocloc_api.h', 'shared/offline_compiler/source/ocloc_arg_helper.cpp']
    source_ref = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    api = {}
    for name in names:
        raw = subprocess.check_output(['git', '-C', str(reference), 'show', source_ref + ':' + name])
        require(raw == (reference / name).read_bytes() and b'SPDX-License-Identifier: MIT' in raw,
            'official MIT API reference differs')
        api[name] = hashlib.sha256(raw).hexdigest()
    plan = dict(stage='prepared', source_commit=revision, inputs=pins,
        capture_ref=dict(commit=qualification_ref, path=str(RESULT.relative_to(P)), key=[KEY, 'extraction_check'],
            canonical_json_sha256=fingerprint(capture)), captures=[dict(arm=x['arm'],
                kernel={k:x['kernel'][k] for k in ['path','sha256','bytes','name']},
                module={k:x['module'][k] for k in ['path','sha256','bytes']}) for x in capture['captures']],
        runtime_files={p: b.digest(p) for p in sorted(deps)}, library=str(library), iga=str(iga),
        python=str(Path(sys.executable).resolve()), host_environment=host,
        API_reference=dict(repository='intel/compute-runtime', revision=source_ref, files=api,
            license='MIT', deployed_source_match_proven=False, required_runtime_API_version=65536),
        output_root=str(ROOT), budgets=dict(wall=60, cpu=60, rss_cap=2 << 30,
            text_cap=2 << 20, file_cap=8 << 20, total_cap=32 << 20),
        criteria='All3captured unchanged ELF64LE/IntelGT modules contain the exact named executable text equal byte-for-byte to their direct kernel GEN copies. Invoke installed ocloc1.0 in memory,select only exact named assembly despite273+byte output names,free all outputs successfully. Owned CPU-only child gates bind pinned ocLoc/IGA libraries and environment,exclude GPU runtime/DRM descriptors,and close0. Original captures stay unchanged. No compilation,GPU work,speed/counter or model claim.',
        retention='Only selected3assemblies and3kernel ZeInfo blocks through imminent source-bound comparison. Raw module/GEN retire after that consumer ends;drop normal dependency/log/ownership/API outputs after compact evidence. Do not retain the other module kernels or duplicate sections.')
    data['offline_plan'] = plan;b.write(ROOT / 'plan.json', plan);b.write(RESULT, record)
    print('Prepared frozen CPU-only ELF/ocloc memory-output gate;commit before run.')


def decode():
    arm = int(sys.argv[2]);require(arm in range(3), 'one captured arm required')
    record = b.gen.committed(RESULT);plan = record[KEY]['offline_plan']
    require(sources(plan['source_commit'], record) == plan['inputs'], 'offline decode source changed')
    entry = plan['captures'][arm]
    for key in ['kernel', 'module']:
        require(b.digest(entry[key]['path']) == entry[key]['sha256'], 'offline input bytes changed')
    proof, metadata = associate(entry)
    iga = ct.CDLL(plan['iga']);library = ct.CDLL(plan['library'])
    library.oclocVersion.argtypes = [];library.oclocVersion.restype = ct.c_int
    version = library.oclocVersion();require(version == 65536, 'ocloc1.0 C ABI required')
    print('OCLOC_IMAGE_GATE ready', flush=True);require(sys.stdin.read(1) == 'X', 'offline ready token')
    u8p = ct.POINTER(ct.c_uint8);u64p = ct.POINTER(ct.c_uint64);namesp = ct.POINTER(ct.c_char_p)
    library.oclocInvoke.argtypes = [ct.c_uint32, namesp,
        ct.c_uint32, ct.POINTER(u8p), u64p, namesp,
        ct.c_uint32, ct.POINTER(u8p), u64p, namesp,
        ct.POINTER(ct.c_uint32), ct.POINTER(ct.POINTER(u8p)), ct.POINTER(u64p), ct.POINTER(namesp)]
    library.oclocInvoke.restype = ct.c_int
    library.oclocFreeOutput.argtypes = library.oclocInvoke.argtypes[-4:]
    library.oclocFreeOutput.restype = ct.c_int
    binary = Path(entry['module']['path']).read_bytes()
    buffer = (ct.c_uint8 * len(binary)).from_buffer_copy(binary)
    inputs = (u8p * 1)(ct.cast(buffer, u8p));sizes = (ct.c_uint64 * 1)(len(binary))
    input_names = (ct.c_char_p * 1)(b'module.zebin')
    argv = [b'ocloc', b'disasm', b'-file', b'module.zebin', b'-dump', b'memory/']
    args = (ct.c_char_p * len(argv))(*argv)
    count = ct.c_uint32();outputs = ct.POINTER(u8p)();lengths = u64p();names = namesp()
    selected = None;log = '';seen = 0;total = 0
    try:
        status = library.oclocInvoke(len(argv), args, 1, inputs, sizes, input_names,
            0, None, None, None, ct.byref(count), ct.byref(outputs), ct.byref(lengths), ct.byref(names))
        seen = count.value;require(0 < seen <= 128, 'ocloc output count budget')
        target = 'memory/.text.' + entry['kernel']['name'] + '.asm'
        for i in range(seen):
            name = names[i].decode('utf-8');length = int(lengths[i]);total += length
            require(length <= 8 << 20 and total <= 64 << 20, 'ocloc memory-output budget')
            if name == target:
                require(selected is None, 'duplicate selected assembly')
                selected = ct.string_at(outputs[i], length)
            elif name == 'stdout.log':log = ct.string_at(outputs[i], min(length, 65536)).decode('utf-8', 'replace').rstrip('\0')
        require(status == 0, 'ocloc disassembly failed: ' + log[:2048])
        require(selected is not None and len(selected) > 0, 'exact selected assembly absent: ' + log[:2048])
        selected.decode('utf-8')
    finally:
        freed = library.oclocFreeOutput(ct.byref(count), ct.byref(outputs), ct.byref(lengths), ct.byref(names))
        require(freed == 0, 'ocloc output release failed')
    folder = ROOT / str(arm)
    assembly = folder / 'kernel.asm';ze = folder / 'kernel.ze_info.yaml'
    require(not assembly.exists() and not ze.exists(), 'no offline output overwrite')
    assembly.write_bytes(selected);ze.write_text('kernels:\n' + metadata)
    result = dict(arm=arm, passed=True, association=proof, API_version=version, ocloc_return=status,
        output_release_return=freed, in_memory_outputs=seen, in_memory_bytes=total,
        selected_name_bytes=len(target.encode()), log_excerpt=log[:2048],
        assembly=dict(path=str(assembly), sha256=b.digest(assembly), bytes=len(selected)),
        metadata=dict(path=str(ze), sha256=b.digest(ze), bytes=ze.stat().st_size))
    require(iga is not None, 'IGA reference lifetime')
    print('OCLOC_IMAGE_GATE decoded', flush=True);require(sys.stdin.read(1) == 'X', 'offline decoded token')
    print(json.dumps(result), flush=True)


def run():
    record = b.gen.committed(RESULT);data = record[KEY];plan = data['offline_plan']
    require(plan['stage'] == 'prepared' and plan == json.loads((ROOT / 'plan.json').read_text()) and
        'offline_check' not in data, 'fresh committed offline plan only')
    require(sources(plan['source_commit'], record) == plan['inputs'], 'offline controller changed')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
        source_commit=plan['source_commit'], GPU_executed=False, observations=[])
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        b.write(ROOT / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = b.supervision.Owner(ROOT / 'owned', persist)
    try:
        require(fingerprint(captured(record)) == plan['capture_ref']['canonical_json_sha256'], 'captured qualification changed')
        for path, sha in plan['runtime_files'].items():require(b.digest(path) == sha, 'offline tool runtime changed: ' + path)
        for arm in range(3):
            folder = ROOT / str(arm);folder.mkdir(mode=0o700);(folder / 'tmp').mkdir()
            env = dict(plan['host_environment'], TMPDIR=str(folder / 'tmp'))
            def interact(io, entry):
                for stage in ['ready', 'decoded']:
                    while True:
                        line = io.line(50)
                        if line.startswith('OCLOC_IMAGE_GATE '):break
                    require(line == 'OCLOC_IMAGE_GATE ' + stage, 'offline gate differs')
                    proc = Path('/proc', str(io.proc.pid))
                    require(str((proc / 'exe').resolve()) == plan['python'], 'offline interpreter differs')
                    actual = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
                    require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()), 'offline environment changed')
                    mapped = b.qualified_runtime.executable_mappings((proc / 'maps').read_text())
                    require(mapped <= set(plan['runtime_files']) and {plan['library'], plan['iga']} <= mapped,
                        'unqualified offline library mapping: ' + str(sorted(mapped - set(plan['runtime_files']))))
                    require(not any(x in p for p in mapped for x in ['libze_loader', 'libze_intel_gpu', 'libur_adapter', 'libsycl']),
                        'GPU runtime loaded by offline disassembler')
                    require(not any('drm-driver:' in x.read_text() for x in (proc / 'fdinfo').iterdir()),
                        'offline disassembler opened DRM device')
                    io.send(b'X')
                observation = json.loads(io.line(20));result['observations'].append(observation)
                io.proc.stdin.close();io.exit(20)
            entry, out, err = owner.run(str(arm), [plan['python'], '-I', '-S', '-B', str(P / SELF), 'decode', str(arm)],
                env, P, interaction=interact, **plan['budgets'])
            require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'offline decode child failed')
            require(result['observations'][-1]['passed'] and result['observations'][-1]['arm'] == arm,
                'offline result arm differs')
        captured(record)
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error);raise
    finally:
        result['ownership_closed'] = owner.active is None and all(b.supervision.closed(x) for x in owner.commands)
        persist();data['offline_check'] = result;data['offline_plan']['stage'] = 'closed'
        if result['passed']:data['stage'] = 'exact_GEN_ELF_text_and_selected_assembly_qualified'
        b.write(RESULT, record)
    print('PASS3exact GEN/ELF-text associations and CPU-only selected-kernel disassemblies.')


if __name__ == '__main__':
    if sys.argv[1] == 'decode':decode()
    else:
        with (b.B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {'prepare': prepare, 'run': run}[sys.argv[1]]()
