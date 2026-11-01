#!/usr/bin/env python3
"""Attest an unchanged GDN object and inspect its exact AOT image; MIT.

V1's four CPU commands succeeded, but its final allowed-input map omitted the
already pinned TLA dependency. Preserve that failed receipt. V2 adds the
original dependency maps, without recompilation or a new native qualification.
"""
import fcntl
import json
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_gdn_projection_fp32 as build

base, ROOT, RESULT = build.base, build.ROOT, build.RESULT
KEY = 'official_gdn_fp32_projection_check_v2'
RECORD = ROOT / 'record-v2.json'
require, digest, write = base.require, base.digest, base.write


def save(row):
    write(RECORD, row)
    data = json.loads(RESULT.read_text()); data[KEY] = base.compact(row); write(RESULT, data)


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not RECORD.exists(), 'fresh V2 attestation only')
    original = json.loads((ROOT / 'record.json').read_text())
    require(data[build.KEY] == base.compact(original) and original['stage'] == 'failed' and
            not original['passed'] and original['ownership_closed'] and not original['active'] and
            original['error'] == "RuntimeError('uncovered/changed actual compiler input')" and
            len(original['commands_executed']) == 4 and all(base.supervision.closed(v) and
                v['exit_code'] == 0 for v in original['commands_executed']), 'exact closed V1 failure required')
    outputs = original['post_failure_artifact_proof']['outputs']
    require(all(digest(v['path']) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes']
                for v in outputs.values()), 'original four objects changed')
    revision = base.basis.build.git(base.P, 'rev-parse', 'HEAD').decode().strip()
    tools = {'/usr/bin/ocloc': digest('/usr/bin/ocloc')}
    ldd = subprocess.check_output(['ldd', '/usr/bin/ocloc'], text=True, timeout=15)
    for path in re.findall(r'(?:=>\s+|^\s*)(/\S+)', ldd, re.M):
        tools[str(Path(path).resolve())] = digest(path)
    # ocloc loads IGA dynamically for ISA decoding. Pin installed candidates too.
    for path in Path('/usr/lib/x86_64-linux-gnu').glob('libiga*.so*'):
        tools[str(path.resolve())] = digest(path)
    row = dict(stage='prepared', source_commit=revision,
        compiled_source_commit=original['source_commit'],
        implementation_commit=original['implementation_commit'],
        native_sources=original['native_sources'],
        controller_inputs=base.controls(revision, controller=Path(__file__).resolve()),
        original_build_record=dict(path=str(ROOT / 'record.json'), sha256=digest(ROOT / 'record.json')),
        original_failure=dict(key=build.KEY, passed=False, error=original['error']),
        outputs=outputs, commands_executed=[], passed=False, compiler_inputs_attested=False,
        plan=dict(scope='Original CPU objects and compiler inputs only;no recompilation or device execution.',
            acceptance='All914 actual inputs must match original native/SDK/provider/dependency Git pins;four objects unchanged.',
            correction='Add the already frozen dependency source maps omitted by V1;do not refresh expected hashes.'),
        codegen=dict(stage='prepared', passed=False, tools=tools,
            command=['/usr/bin/ocloc', 'disasm', '-file', str(ROOT / 'device.bin'),
                     '-dump', str(ROOT / 'codegen/dump')],
            environment=dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                             TMPDIR=str(ROOT / 'tmp'), LD_DEBUG='libs'),
            budgets=dict(wall=120, cpu=240, rss_cap=2 << 30, text_cap=4 << 20,
                         file_cap=128 << 20, total_cap=256 << 20),
            scope='Decode this exact offline AOT image;no GPU execution or numeric/ownership/speed qualification.'),
        active=False, ownership_closed=True, GPU_executed=False, inference_adopted=False)
    save(row)
    print('Prepared unchanged-object V2 attestation and bounded offline disassembly.', flush=True)


def admission(row):
    require(base.committed(RESULT)[KEY] == base.compact(row), 'commit exact V2 plan')
    require(base.controls(row['source_commit'], controller=Path(__file__).resolve()) ==
            row['controller_inputs'], 'V2 controller inputs changed')
    ref = row['original_build_record']
    require(digest(ref['path']) == ref['sha256'], 'original failed record changed')
    original = json.loads(Path(ref['path']).read_text())
    require(original['source_commit'] == row['compiled_source_commit'] and
            original['native_sources'] == row['native_sources'], 'compiled native identity changed')
    for name, item in row['outputs'].items():
        require(digest(item['path']) == item['sha256'] and
                Path(item['path']).stat().st_size == item['bytes'], 'original object changed: ' + name)
    return original


def attest(row):
    require(row['stage'] == 'prepared', 'fresh input attestation only')
    original = admission(row)
    allowed = build.admit(original)
    full, _ = build.source.sources()
    require([{k:v for k,v in group.items() if k != 'source_sha256'} for group in full['dependencies']]
            == original['dependencies'], 'original dependency descriptors changed')
    for group in full['dependencies']:
        for name, sha in group['source_sha256'].items():
            path = str((Path(group['root']) / name).resolve())
            require(digest(path) == sha, 'original dependency bytes changed: ' + path)
            allowed[path] = sha
    found = set()
    for line in (ROOT / 'native16.d').read_text().replace('\\\n', '').splitlines():
        if not line.strip(): continue
        target, colon, dependencies = line.partition(':')
        require(colon and target.strip(), 'invalid original dependency rule')
        found.update(str((ROOT / name).resolve()) for name in shlex.split(dependencies))
    require(len(found) == 914 and str(ROOT / 'source/src/prefill/sycl/vllm_xe2/native16.hpp') in found and
            all('/MistVVK/' not in p and allowed.get(p) == digest(p) for p in found),
            'actual compiler inputs missing, changed or not originally pinned')
    path = ROOT / 'compiler-inputs-v2.json'; write(path, {p:allowed[p] for p in sorted(found)})
    row.update(stage='attested-compiled-not-executed', passed=True, compiler_inputs_attested=True,
               compiled_inputs=dict(path=str(path), sha256=digest(path), files=len(found), uncovered=0),
               recompilation=False)
    save(row)
    print('V2:914 original compiler inputs and four unchanged outputs pass;no recompilation/GPU execution.', flush=True)


def inspect(row):
    admission(row)
    require(row['passed'] and row['compiler_inputs_attested'] and row['codegen']['stage'] == 'prepared',
            'attested objects and fresh disassembly required')
    plan = row['codegen']; root = ROOT / 'codegen'; require(not root.exists(), 'fresh codegen directory')
    require(all(digest(p) == sha for p,sha in plan['tools'].items()), 'decoder tools changed')
    root.mkdir(mode=0o700); owner = None
    def persist():
        write(RECORD, dict(row, commands_executed=owner.commands if owner else [], active=bool(owner and owner.active)))
    def monitor(io, entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(base.supervision.artifact_bytes(root) <= plan['budgets']['total_cap'], 'codegen size budget')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        plan['stage'] = 'decoding'; owner = base.supervision.Owner(root / 'owned', persist)
        entry, out, err = owner.run('offline-disasm', plan['command'], plan['environment'], root,
                                   interaction=monitor, **plan['budgets'])
        require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                'offline disassembly failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-3000:])
        files = {str(p.relative_to(root)):dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in (root / 'dump').rglob('*') if p.is_file()}
        require(files, 'no decoded AOT sections')
        plan.update(stage='decoded-needs-store-ownership-review', passed=True, files=files,
                    manual_FP32_store_review_passed=False, numerical_or_device_qualification=False)
    except BaseException as error:
        plan.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=False, ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)),
                   commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None, 'decoder owner unresolved'); save(row)
    print(json.dumps(dict(stage=plan['stage'], passed=plan['passed'], GPU_executed=False)), flush=True)
    require(plan['passed'], 'offline codegen inspection failed')


if __name__ == '__main__':
    with (base.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare()
        else:
            row = json.loads(RECORD.read_text())
            if sys.argv[1:] == ['attest']: attest(row)
            elif sys.argv[1:] == ['inspect']: inspect(row)
            else: raise RuntimeError('check_gdn_projection_fp32.py prepare|attest|inspect')
