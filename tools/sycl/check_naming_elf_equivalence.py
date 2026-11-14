#!/usr/bin/env python3
"""Bounded production-ELF comparison for the non-optimizing KV rename.

Main-authored ELF64 reader,repository LICENSE. This prospective evidence test
does not waive the original performance plan or qualify model/task behavior.
"""
import collections
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'naming-elf-equivalence-v1'
THIS = 'tools/sycl/check_naming_elf_equivalence.py'
INPUTS = {
    'reference': ('decode-kv-native-v1',
        '1ab18db544e7af8cf00e355550dc6e25aed721ab09b011bbdb5e045947ae7923',
        '1eec28c77f393d34814dc748502113514973ea8e',
        '330405370dab1d52995540e79c9dad5a3decf46d4e2007ab096dcc57e602d21e'),
    'candidate': ('phase-kv-naming-native-v1',
        '2dc7d4f3ead6068be12afa6ef36c589324f75d31ddd95f6fa96421552da57c9c',
        '9849e35803f011aaafb2753c1e0f31411e40308e',
        '6d2b4f7d6a79424c3da5ac6e35d7e1336016604e2f7bdb68b84031d2879903de'),
}
RENAMES = {'verify_coalesced_kv': 'decode_verify_coalesced_kv',
           'kv_stream_resolve_with_coalesced_plan': 'decode_kv_stream_resolve_coalesced'}
PLAN = dict(question='Are all production host/device image sections and load layout unchanged by Unit A?',
    input_scope='Only the two exact original full-build strata ELFs;not the extra fixture or a new build.',
    equality='ELF identity/type/machine/entry/flags/header counts and every program header must match. '
        'Section-name sets,ordinary section headers and all section contents must match,including '
        'non-allocated device-image sections. NOBITS size/address/flags must match. Allocated section '
        'offsets and every program load layout remain exact.',
    exceptions='Ignore only ELF section-table file offset;nonallocated .symtab/.strtab/.shstrtab raw '
        'bookkeeping and their offsets/sizes;and the GNU build-ID descriptor after validating its '
        'identical note structure. Compare all non-file static symbols by address/size/type/visibility '
        'and section,normalizing only the two declared C++ name-length prefixes. Ignore STT_FILE '
        'debug filenames. Dynamic symbols/strings/relocations and all other bytes remain exact.',
    stopping='Require positive self-comparison and refusals for text/data/dynamic-string/load-layout/'
        'symbol perturbations and truncated ELF before comparing arms. One committed attempt,'
        'finite CPU/address-space/file budgets. A mismatched section or '
        'symbol is an ineligible equivalence result;do not add exceptions or refresh inputs afterward.',
    limits='A pass is static production image compatibility evidence only. Numerical/state/model '
        'handoff remain separate. The original A speed plan is unchanged;replacement needs a new '
        'prospective committed decision. No B1/B2,physical262144,task,teardown or speed qualification.')
require = guards.require
sha = lambda data: hashlib.sha256(data).hexdigest()


def field(data, offset, length):
    require(0 <= offset <= len(data) and 0 <= length <= len(data) - offset, 'ELF extent invalid')
    return data[offset:offset + length]


def string(data, offset):
    require(0 <= offset < len(data), 'ELF string offset invalid')
    end = data.find(b'\0', offset)
    require(end >= offset, 'ELF string terminator absent')
    return data[offset:end].decode('utf-8')


def parse(data):
    require(len(data) <= 64 << 20, 'bounded original ELF required')
    h = struct.unpack('<16sHHIQQQIHHHHHH', field(data, 0, 64))
    require(h[0][:7] == b'\x7fELF\x02\x01\x01' and h[2] == 62
            and h[8] == 64 and h[9] == 56 and h[11] == 64
            and 0 < h[10] < 256 and 0 < h[12] < 4096 and h[13] < h[12], 'supported ELF64 required')
    headers = [struct.unpack('<IIQQQQIIQQ', field(data, h[6] + i * 64, 64)) for i in range(h[12])]
    names = field(data, headers[h[13]][4], headers[h[13]][5])
    sections = {}
    for item in headers:
        name = string(names, item[0])
        require(name not in sections, 'duplicate section name')
        sections[name] = dict(header=item[1:], data=b'' if item[1] == 8 else field(data, item[4], item[5]))
    program = [field(data, h[5] + i * 56, 56).hex() for i in range(h[10])]
    symbols = []
    for name, item in sections.items():
        if item['header'][0] != 2: continue  # SHT_SYMTAB;dynamic bytes remain ordinary sections.
        s = item['header']; require(s[8] == 24 and s[4] % 24 == 0 and s[5] < len(headers), 'symbol extent invalid')
        table = headers[s[5]]; strings = field(data, table[4], table[5])
        for at in range(0, len(item['data']), 24):
            symbol = struct.unpack('<IBBHQQ', item['data'][at:at + 24])
            if symbol[1] & 15 == 4: continue  # STT_FILE is debug provenance,not execution.
            label = string(strings, symbol[0])
            for before, after in RENAMES.items():
                label = label.replace(str(len(before)) + before, str(len(after)) + after)
            symbols.append((label, *symbol[1:]))
    return dict(header=h[:6] + h[7:], section_table_file_offset=h[6],
                programs=program, sections=sections, symbols=collections.Counter(symbols))


def compare(old, new):
    differences = []
    if old['header'] != new['header']: differences.append('ELF semantic header')
    if old['programs'] != new['programs']: differences.append('program headers/load layout')
    if old['sections'].keys() != new['sections'].keys(): differences.append('section name set')
    matched, metadata = {}, []
    for name in old['sections'].keys() & new['sections'].keys():
        a, b = old['sections'][name], new['sections'][name]
        if name in ('.symtab', '.strtab', '.shstrtab'):
            require(not (a['header'][1] & 2 or b['header'][1] & 2), 'metadata exception is allocated')
            metadata.append(name); continue
        if a['header'] != b['header']: differences.append(name + ':header')
        if name == '.note.gnu.build-id':
            for item in (a, b):
                ns, ds, kind = struct.unpack('<III', field(item['data'], 0, 12))
                require(ns == 4 and kind == 3 and ds in (16, 20, 32)
                        and item['data'][12:16] == b'GNU\0' and len(item['data']) == 16 + ds,
                        'only a complete GNU build-ID descriptor may differ')
            if a['data'][:16] != b['data'][:16]: differences.append(name + ':note structure')
            metadata.append(name); continue
        if a['data'] != b['data']: differences.append(name + ':contents')
        else: matched[name] = dict(bytes=len(a['data']), sha256=sha(a['data']))
    if old['symbols'] != new['symbols']: differences.append('normalized static symbol semantics')
    return dict(equivalent=not differences, differences=differences,
        equal_section_count=len(matched), equal_section_bytes=sum(v['bytes'] for v in matched.values()),
        equal_sections_sha256=sha(json.dumps(matched, sort_keys=True).encode()),
        equal_static_symbol_count=sum(old['symbols'].values()) if old['symbols'] == new['symbols'] else None,
        metadata_exceptions=sorted(metadata),
        section_table_file_offsets=[old['section_table_file_offset'], new['section_table_file_offset']])


def main():
    source = subprocess.check_output(['git', '-C', str(P), 'show', 'HEAD:' + THIS], timeout=20)
    require(source == (P / THIS).read_bytes(), 'commit exact comparison source/plan before execution')
    require(not ROOT.exists(), 'one fresh equivalence attempt only')
    ROOT.mkdir(mode=0o700)
    result = dict(plan=PLAN, controller_sha256=sha(source),
        controller_commit=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
        inputs={}, complete=False, passed=False, GPU_executed=False, performance_qualified=False)
    (ROOT / 'attempt.json').write_text(json.dumps(result, indent=2) + '\n')
    try:
        images, raw_images = {}, {}
        for name, (directory, record_sha, revision, binary_sha) in INPUTS.items():
            record = B / directory / 'record.json'; raw = record.read_bytes()
            require(sha(raw) == record_sha, 'original build receipt changed')
            row = json.loads(raw); image = row['binaries']['strata']; data = Path(image['path']).read_bytes()
            require(row['native_source_commit'] == revision and row['stage'] == 'built'
                    and row['passed'] and row['ownership_closed'] and not row['active']
                    and len(data) == image['bytes'] and sha(data) == binary_sha == image['sha256'],
                    'exact closed original full-build ELF required')
            result['inputs'][name] = dict(record_sha256=record_sha, source_commit=revision, binary=image)
            raw_images[name] = data
            images[name] = parse(data)
        reference = images['reference']
        require(compare(reference, reference)['equivalent'], 'self-comparison control failed')
        controls = []
        for name in ('.text', '.rodata', '.dynstr'):
            changed = copy.deepcopy(reference); value = changed['sections'][name]['data']
            require(bool(value), 'nonempty real control section required')
            changed['sections'][name]['data'] = bytes([value[0] ^ 1]) + value[1:]
            require(not compare(reference, changed)['equivalent'], 'changed section was accepted: ' + name)
            controls.append(name)
        changed = copy.deepcopy(reference); changed['programs'][0] = 'changed-load-layout'
        require(not compare(reference, changed)['equivalent'], 'changed program layout accepted')
        controls.append('program layout')
        changed = copy.deepcopy(reference); symbol = next(iter(changed['symbols']))
        changed['symbols'][symbol] -= 1; changed['symbols'][(symbol[0] + '_unexpected', *symbol[1:])] += 1
        require(not compare(reference, changed)['equivalent'], 'changed symbol semantics accepted')
        controls.append('symbol semantics')
        try: parse(raw_images['reference'][:-1])
        except RuntimeError: controls.append('truncated ELF')
        else: raise RuntimeError('truncated ELF accepted')
        result['controls'] = dict(positive=1, refused=controls, passed=True)
        result.update(comparison=compare(images['reference'], images['candidate']), complete=True)
        result['passed'] = result['comparison']['equivalent']
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        (ROOT / 'record.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    require(result['complete'] and result['passed'], 'equivalence incomplete/ineligible;original performance gate remains')


if __name__ == '__main__':
    require(len(sys.argv) == 1, 'no arguments;exact original inputs are committed in the controller')
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 << 20, 2 << 20))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        main()
