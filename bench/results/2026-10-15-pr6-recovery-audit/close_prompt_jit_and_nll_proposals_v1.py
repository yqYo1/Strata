#!/usr/bin/env python3
"""Verify compact JIT/provenance evidence; retire closed redundant captures."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ['prompt-jit-offload-tool-help-v1.json', 'prompt-jit-identity-v1.json']
        names += [f'prompt-jit-resources-a0-v{v}.json' for v in (1, 2, 3)]
        names += ['prompt-jit-resource-analysis-v1.json']
        records = {n: json.loads((A / n).read_text()) for n in names}
        pins = {n: sha(A / n) for n in names}
        for r in records.values():
            require(r['passed'] and r['complete'] and not r['active'], 'closed successful analysis/capture')
            for c in r.get('commands', []):
                require(module.closed(c), 'owned command closed')
                for old in c.get('owners', []):
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned identity absent')
        for v in (1, 2, 3):
            r = records[f'prompt-jit-resources-a0-v{v}.json']
            require(not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'healthy capture')
            for f in r['artifacts']:
                p = Path(f['path'])
                require(p.stat().st_size == f['bytes'] and sha(p) == f['sha256'], 'original IGC artifact identity')
        analysis = records['prompt-jit-resource-analysis-v1.json']
        raw = records['prompt-jit-resources-a0-v3.json']
        require(all(pins[n] == h for n, h in analysis['references'].items()), 'analysis input hash chain')
        for arm, k in analysis['kernels'].items():
            p = Path(k['assembly']['path']); text = p.read_text()
            require(sha(p) == k['assembly']['sha256'] and text.splitlines()[0] == '//.kernel ' + k['exact_symbol'], 'exact kernel ASM')
            require(text.splitlines()[:10] == k['assembly']['header'], 'verbatim compiler resource header')
            for pattern, key in ((r'//\.instCount (\d+)', 'instCount'), (r'numGRF=(\d+)', 'numGRF'),
                                 (r'//\.spill flag store (\d+)', 'flag_spill_store'),
                                 (r'//\.spill flag load (\d+)', 'flag_spill_load')):
                require(int(re.search(pattern, text)[1]) == k['assembly'][key], 'actual ASM field ' + key)
            blocks = []
            for m in re.finditer(r'^  - name:\s+' + re.escape(k['exact_symbol']) + r'\s*$', raw['stdout'], re.M):
                end = raw['stdout'].find('\n  - name:', m.end())
                block = raw['stdout'][m.start():end if end >= 0 else len(raw['stdout'])]
                if '    execution_env:\n' in block:
                    blocks.append(block.split('    execution_env:\n', 1)[1].split('    payload_arguments:', 1)[0])
            require(blocks == [k['resource_metadata_block']], 'exact ZEInfo symbol/resource block')
        trace = B / 'main-prompt-jit-resources-a0-root-v3/owned/component-jit-capture.stderr'
        launch = analysis['observed_arm0_launch']
        require(launch['kernel_create'] in trace.read_text().splitlines() and launch['first_launch'] in trace.read_text().splitlines(), 'actual exact-handle launch evidence')
        sizes = [[int(i), int(v)] for i, v in re.findall(r'\.type = UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = (\d+), \.size = (\d+)', launch['first_launch'])]
        require(sizes == launch['local_args'] and sum(v for _, v in sizes) == 44496, 'all13 launch-local argument byte counts')
        widths = [min(p + 1, 2051) for p in range(78452)]
        arithmetic = analysis['source_arithmetic']
        require(arithmetic['tiles_per_layer_per_head'] == sum((w + 31) // 32 for w in widths)
                and arithmetic['total_source_DPASFLOPs'] == 12 * 2 * 256 * sum((w + 31) // 32 for w in widths) * 4096
                and arithmetic['total_logical_KV_code_scale_bytes_upper'] == 12 * 2 * sum(widths) * 528,
                'correct12-layer deterministic source-work scope')
        v3 = A / 'own-dense-prefill-nll-v3'; v2 = A / 'own-dense-prefill-nll-v2'
        proposal = json.loads((v3 / 'manifest.json').read_text())
        provenance = json.loads((v3 / 'provenance.json').read_text())
        require(sha(v3 / 'provenance.json') == proposal['provenance_sha256'], 'proposal provenance identity')
        require(sha(v2 / 'manifest.json') == provenance['original_v2']['manifest_sha256']
                and sha(v2 / 'source.diff') == provenance['original_v2']['source_diff_sha256'], 'original v2 unchanged')
        for entry in proposal['package_files']:
            p = v3 / entry['path']
            require(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'current own v3 full package')
        require(sha(P / 'LICENSE') == provenance['root_license']['sha256'], 'actual MIT root license')
        for entry in proposal['production_source_snapshot']:
            require(sha(P / entry['path']) == entry['sha256'], 'current scoped source base stable')
        for entry in provenance['normalized_v2_to_v3_source_comparison']:
            before = (v2 / entry['path']).read_bytes(); after = (v3 / entry['path']).read_bytes()
            require(before.replace(b'SPDX-License-Identifier: LGPL-3.0-or-later', b'SPDX-License-Identifier: MIT') == after,
                    'own metadata correction only; not attribution removal or external source rewriting')
        require((v2 / 'source.diff').read_bytes().replace(b'SPDX-License-Identifier: LGPL-3.0-or-later', b'SPDX-License-Identifier: MIT') == (v3 / 'source.diff').read_bytes(), 'whole patch normalized byte equality')
        baseline = Path(analysis['kernels']['0']['assembly']['path'])
        keep = {baseline, baseline.with_suffix('.isaasm')}
        require(all(p.exists() for p in keep) and sum(p.stat().st_size for p in keep) < 4 << 20, 'bounded current baseline ISA consumer')
        roots = [B / 'main-prompt-jit-offload-tool-help-root-v1', B / 'main-prompt-jit-identity-root-v1']
        roots += [B / f'main-prompt-jit-resources-a0-root-v{v}' for v in (1, 2, 3)]
        files = []
        for root in roots:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p not in keep:
                    require(not p.is_symlink(), 'private regular capture')
                    files.append((p, 'Closed exact capture; names/hashes/loaded libraries/ZEInfo/ASM headers/launch evidence preserved; duplicate dumps, opaque RA, unrelated SIP, intermediate IR and rejected mapping ISA have no current consumer.'))
        for p in sorted(v2.rglob('*')):
            if p.is_file():
                require(not p.is_symlink(), 'own v2 regular proposal file')
                files.append((p, 'Superseded own source-only proposal; original hashes and metadata-origin explanation plus code-equivalent MIT v3 patch/provenance preserved. Never applied, built or tested.'))
        target = A / 'closed-prompt-jit-and-nll-retention-v1.json'
        require(not target.exists(), 'fresh retention manifest')
        result = dict(passed=True, complete=False, active=False, recipe_sha256=sha(Path(__file__)),
            original_receipts=pins, original_PASS_FAIL_unchanged=True,
            canonical_NLL=dict(manifest_sha256=sha(v3 / 'manifest.json'), provenance_sha256=sha(v3 / 'provenance.json'), source_diff_sha256=sha(v3 / 'source.diff'), status='UNAPPLIED_UNBUILT_UNTESTED', owner='Main', consumer='Current-input FP32 repeats / FP16 matched and allGPU quality review; source review before integration', full_package_budget_bytes=16 << 20, next_review='after scoped selector clean cohort and NLL integration/rejection'),
            kept_ISA=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p), owner='Main', consumer='Current default mapping memory/barrier ISA analysis for next independent prompt candidate', budget_bytes=4 << 20, next_review='after next candidate review or superseding compilation') for p in sorted(keep)],
            files=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p), reason=reason) for p, reason in files], finished_utc=module.utc())
        target.write_text(json.dumps(result, indent=2) + '\n')
        for entry in result['files']:
            p = Path(entry['path'])
            require(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'exact retirement file identity')
            p.unlink()
        require(all(sha(A / n) == h for n, h in pins.items()), 'original closed receipt bytes unchanged')
        result.update(complete=True, files_retired=len(files), logical_bytes_retired=sum(v['bytes'] for v in result['files']))
        target.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('files_retired', 'logical_bytes_retired')}), flush=True)


if __name__ == '__main__':
    main()
