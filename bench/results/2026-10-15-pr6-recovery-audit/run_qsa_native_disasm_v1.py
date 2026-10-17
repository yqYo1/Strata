#!/usr/bin/env python3
"""Offline decoding of the captured, source-bound current QSA native module.

New diagnostic only: no GPU execution, product edit, or performance claim.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HOST = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8',
            LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), realpath=str(path.resolve()),
                bytes=path.stat().st_size, sha256=digest)


def sections(path):
    blob = Path(path).read_bytes()
    assert blob[:6] == b'\x7fELF\x02\x01'
    offset = struct.unpack_from('<Q', blob, 40)[0]
    size, count, string_index = struct.unpack_from('<HHH', blob, 58)
    assert size == 64 and 0 < count < 4096
    header = struct.unpack_from('<IIQQQQIIQQ', blob, offset + size*string_index)
    strings = blob[header[4]:header[4]+header[5]]
    result = {}
    for index in range(count):
        header = struct.unpack_from('<IIQQQQIIQQ', blob, offset + size*index)
        name = strings[header[0]:].split(b'\0', 1)[0].decode()
        if header[1] != 8:
            assert header[4]+header[5] <= len(blob)
            result[name] = blob[header[4]:header[4]+header[5]]
    return result


def main():
    assert len(sys.argv) == 1
    capture_path = A/'own-qsa-native-zebin-serial-smoke-v1.json'
    capture_pin = pin(capture_path)
    capture = json.loads(capture_path.read_text())
    assert all(capture[key] for key in ('passed', 'complete', 'ownership_closed'))
    assert not capture['active'] and capture['fresh_faults'] == []
    module = next(x for x in capture['native_binaries'] if
                  x['sha256'] == '049221cb25a6c6e9c36f9e96a7cdee08d4309e857c21aa14e468beec9d8ab896')
    assert all(pin(module['path'])[key] == module[key] for key in
               ('path', 'realpath', 'bytes', 'sha256'))
    owner_path = A/'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('qsa_offline_owner', owner_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    root = B/'main-qsa-native-disasm-v1'
    destination = A/'qsa-native-disasm-v1.json'
    assert not root.exists() and not destination.exists()
    tools = [Path('/usr/bin/ocloc'), Path('/usr/lib/x86_64-linux-gnu/libocloc.so'),
             Path('/usr/lib/x86_64-linux-gnu/libiga64.so.2'),
             Path('/usr/lib/x86_64-linux-gnu/libigc.so.2'),
             Path('/usr/lib/x86_64-linux-gnu/libigdfcl.so.2')]
    frozen = [pin(x) for x in [Path(__file__), owner_path, capture_path, *tools]]
    record = dict(schema='strata-qsa-native-offline-disasm-v1', passed=False,
                  complete=False, active=False, GPU_executed=False,
                  capture=capture_pin, module=module, frozen_files=frozen,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  source_revision='896299c37162c458139aa4c5ea9f5678c856eb22',
                  scope='Decode current original full-engine-archive native prefix module. '
                        'Static ISA counts and binary metadata only; dynamic instruction counts, '
                        'runtime SLM allocations, occupancy and hardware saturation remain unproved.',
                  budget=dict(wall_seconds_per_command=60, child_CPU_seconds=120,
                              session_RSS_bytes=4 << 30, total_artifact_bytes=30 << 20,
                              raw_text_bytes=8 << 20),
                  retention='Main owns decoded files for current QSA DPAS/GRF/SLM question; '
                            'review after compact source-bound summary and follow-up pitch comparison. '
                            'Keep hashes and relevant metadata; retire redundant successful logs.',
                  commands=[])
    owner = None

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')

    def verify():
        for item in frozen:
            assert pin(item['path']) == item
        assert all(pin(module['path'])[key] == module[key] for key in
                   ('path', 'realpath', 'bytes', 'sha256'))

    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        output = root/'decoded'
        output.mkdir(mode=0o700)
        owner = helper.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        try:
            verify()
            sec = sections(module['path'])
            record['native_sections'] = [dict(name=name, bytes=len(data),
                                             sha256=hashlib.sha256(data).hexdigest())
                                         for name, data in sec.items()]
            ze_info = sec['.ze_info'].decode()
            (root/'native-ze-info.yaml').write_text(ze_info)
            record['ze_info'] = pin(root/'native-ze-info.yaml')
            command, stdout, stderr = owner.run(
                'ocloc-disasm', ['/usr/bin/ocloc', 'disasm', '-file', module['path'],
                                 '-dump', str(output)+'/'], HOST, root,
                wall=60, cpu=120, rss_cap=4 << 30, text_cap=8 << 20,
                file_cap=30 << 20, total_cap=30 << 20)
            assert helper.closed(command) and command['exit_code'] == 0
            record['stdout'] = stdout.read_text()
            record['stderr'] = stderr.read_text()
            record['decoded_files'] = [pin(x) for x in sorted(output.rglob('*')) if x.is_file()]
            assembly = sorted(output.glob('*.asm'))
            assert len(assembly) == 4, [str(x) for x in assembly]
            record['assembly'] = []
            for path in assembly:
                text = path.read_text()
                item = dict(file=pin(path),
                            dpas_lines=[x for x in text.splitlines()
                                        if re.search(r'\bdpasw?(?:\.|\s|\()', x)],
                            send_line_count=sum(bool(re.search(r'\bsend(?:\.|\s|\()', x))
                                                for x in text.splitlines()))
                item['dpas_static_instruction_count'] = len(item['dpas_lines'])
                record['assembly'].append(item)
            verify()
            record.update(passed=True, complete=True)
        except BaseException as error:
            import traceback
            record.update(error=repr(error), traceback=traceback.format_exc())
        finally:
            record['ownership_closed'] = (owner.active is None and
                                          all(helper.closed(x) for x in owner.commands))
            if not record['ownership_closed']:
                record.update(passed=False, complete=False)
            persist()
            destination.write_bytes((root/'record.json').read_bytes())
    print(json.dumps(dict(receipt=str(destination), passed=record['passed'],
                          complete=record['complete'], error=record.get('error'))))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
