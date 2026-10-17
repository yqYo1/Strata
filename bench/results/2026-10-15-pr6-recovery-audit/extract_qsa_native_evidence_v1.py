#!/usr/bin/env python3
"""Extract source-bound static/native evidence, then retire verified duplicates."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
TARGET = '_ZTSZZN6strata7kernelsL17prefix_batch_implILi33ELb0ELb0'


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def check(item):
    actual = pin(item['path'])
    assert all(actual[key] == item[key] for key in ('path', 'bytes', 'sha256'))


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def main():
    assert len(sys.argv) == 1
    source = A.parents[2]/'src/kernels/sycl/qsa_bf16_prefix.cpp'
    assert pin(source)['sha256'] == 'fa505d02e2b24f7cb6948033921fa697c8aeb58e6f4df6d873f13e5a7918e36b'
    capture_path = A/'own-qsa-native-zebin-serial-smoke-v1.json'
    disasm_path = A/'qsa-native-disasm-v1.json'
    capture = json.loads(capture_path.read_text())
    disasm = json.loads(disasm_path.read_text())
    helper_path = A/'direct_owner_transient_artifacts_v1.py'
    assert pin(helper_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('native_evidence_owner', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    capture_root = Path(capture['native_cwd']).parent
    disasm_root = Path(disasm['ze_info']['path']).parent
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record in (capture, disasm):
            assert record['passed'] and record['complete'] and record['ownership_closed']
            assert not record['active']
            for command in record['commands']:
                assert helper.closed(command) and command['exit_code'] == 0
                for old in command['owners']+[command['direct_identity']]:
                    now = helper.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
        assert capture['fresh_faults'] == [] and capture['driver_dump_entries'] == []
        assert capture['boot_unchanged']
        for item in capture['native_binaries']+disasm['decoded_files']+[disasm['ze_info']]:
            check(item)
        for record, root in ((capture, capture_root), (disasm, disasm_root)):
            assert (root/'record.json').read_bytes() == (
                capture_path if record is capture else disasm_path).read_bytes()
            for command in record['commands']:
                for filename, expected in command['logs'].items():
                    actual = pin(root/'owned'/filename)
                    assert all(actual[key] == expected[key] for key in ('bytes', 'sha256'))
        trace_path = capture_root/'owned/qsa-bf16.stderr'
        lines = trace_path.read_text().splitlines()
        creations = [line for line in lines if '<--- urKernelCreate(' in line and TARGET in line]
        assert len(creations) == 1
        handle = re.search(r'\.phKernel = \S+ \((0x[0-9a-f]+)\)', creations[0])[1]
        launches = [line for line in lines if '<--- urEnqueueKernelLaunchWithArgsExp(' in line
                    and f'.hKernel = {handle},' in line]
        assert len(launches) == 12
        normalized = []
        for line in launches:
            args = [dict(kind=kind, index=int(index), bytes=int(size)) for kind, index, size in
                    re.findall(r'\.type = (UR_EXP_KERNEL_ARG_TYPE_\w+), \.index = (\d+), \.size = (\d+)', line)]
            local = [arg for arg in args if arg['kind'].endswith('_LOCAL')]
            assert len(args) == 19 and len(local) == 9
            assert sum(arg['bytes'] for arg in local) == 28224
            assert line.endswith('-> UR_RESULT_SUCCESS;')
            sizes = re.findall(r'\.p(?:Global|Local)WorkSize = \S+ \((\d+)\)', line)
            assert len(sizes) == 2 and sizes[1] == '128'
            normalized.append(dict(line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                                   global_work_size=int(sizes[0]), local_work_size=int(sizes[1]),
                                   arguments=args, local_argument_bytes=28224, result='UR_RESULT_SUCCESS'))
        ze_info = Path(disasm['ze_info']['path']).read_text()
        blocks = [block for block in ze_info.split('\n  - name:')[1:]
                  if TARGET in block.splitlines()[0] and 'execution_env:' in block]
        assert len(blocks) == 1
        metadata = '  - name:'+blocks[0]
        assert 'grf_count:       128' in metadata and 'slm_size:        32' in metadata
        assembly = next(item for item in disasm['assembly'] if TARGET in item['file']['path'])
        assert assembly['dpas_static_instruction_count'] == 16
        result = dict(schema='strata-qsa-native-bound-evidence-v1',
                      inputs=[pin(capture_path), pin(disasm_path), pin(source), pin(__file__), pin(trace_path)],
                      target_native_module=disasm['module'], target_assembly=assembly,
                      target_ze_info=metadata, runtime_kernel_create=creations[0],
                      representative_launch=launches[0], all_target_launches=normalized,
                      local_argument_bytes=28224, static_descriptor_SLM_bytes=32,
                      grf_count=128, simd_size=16,
                      limitations=[
                          'ISA captured from qualified same-archive numerical fixture, not a fresh native dump of the full-model process.',
                          '16 DPAS is a static instruction count; it is not dynamic work, XMX utilization or a hardware saturation measurement.',
                          'Nine local argument requests total28224 bytes. Descriptor32 excludes these; neither value proves encoded/rounded SLM, occupancy or bank conflicts.',
                          'No new spill/resource measurement of a pitch variant, no new PP measurement, no default adoption.',
                          'Physical262144 full-model, RESTORE and sustained-generation qualification remain pending.'])
        result_path = A/'qsa-native-bound-evidence-v1.json'
        retention_path = A/'qsa-native-retention-v1.json'
        assert not result_path.exists() and not retention_path.exists()
        write_new(result_path, result)
        assert json.loads(result_path.read_text()) == result
        retire = []
        for root in (capture_root, disasm_root):
            retire.extend((path, 'Successful verbose logs: source/exit/fault receipts and exact selected launch evidence verified.')
                          for path in sorted((root/'owned').iterdir()) if path.is_file())
            retire.append((root/'record.json', 'Byte-identical duplicate of canonical Git receipt.'))
        retire.extend((Path(item['path']), 'Nonselected native module or identical duplicate; selected target module retained.')
                      for item in capture['native_binaries'] if item['sha256'] != disasm['module']['sha256'])
        retire.extend((Path(item['path']), 'Offline decode reproducible from retained native ELF; target assembly and compact metadata retained.')
                      for item in disasm['decoded_files'] if item['path'] != assembly['file']['path'])
        retire.append((Path(disasm['ze_info']['path']), 'Exact target metadata preserved in compact bound evidence.'))
        manifest = dict(schema='strata-qsa-native-retention-v1', result=pin(result_path),
                        ownership='Main; all captured native/utility sessions closed; exclusive measurement lock held.',
                        retained_current_consumer='Target native ELF and assembly for Main QSA instruction/resource tuning and comparison with pitch variants. Review after that comparison; finite original30MiB job caps apply.',
                        paths=[dict(**pin(path), reason=reason) for path, reason in retire],
                        removed_bytes=sum(path.stat().st_size for path, _ in retire),
                        status='planned')
        write_new(retention_path, manifest)
        for path, _ in retire:
            path.unlink()
        manifest['status'] = 'complete'
        retention_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(result=str(result_path), target_dpas=16, local_argument_bytes=28224,
                              removed_files=len(retire), removed_bytes=manifest['removed_bytes'])))


if __name__ == '__main__':
    main()
