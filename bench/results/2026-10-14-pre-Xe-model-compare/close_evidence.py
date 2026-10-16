#!/usr/bin/env python3
"""Close the owned pre-Xe model comparison; preserve receipts, never delete logs."""
import fcntl
import gzip
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent
D = P / 'bench/results/2026-10-14-primary-model-baseline'
Q = P / 'bench/results/2026-10-14-pre-Xe-replay'
ROOTS = ['main-pre-Xe-1eb-model-short4096-root-v1',
         'main-pre-Xe-1eb-model-stock4096-root-v1']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def alive_same(pid, ticks):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return int(raw[raw.rfind(')') + 2:].split()[19]) == ticks


def require_closed(record):
    assert not record['active'] and record['passed'] and record['complete']
    for command in record['commands']:
        assert command['direct_child_reaped'] and command['session_empty']
        assert not command.get('survivors') and command['exit_code'] == 0
        for owner in command.get('owners', []):
            assert not alive_same(owner['pid'], owner['start_ticks'])
    result = record['result']
    assert result['passed'] and result['server_exit_code'] == 0
    native = result['real_engine_exit']
    assert native['exit_code'] == 0
    assert not alive_same(native['native']['pid'], native['native']['start_ticks'])
    assert not record['fresh_faults'] and not record['driver_dump_exists']
    assert record['boot_unchanged'] and record['post_source_and_binary_unchanged']


def read_archive(directory, relative):
    path = directory / relative
    item = next(x for x in json.loads((directory / 'receipts-index.json').read_text())
                if x['path'] == relative)
    raw = gzip.decompress(path.read_bytes())
    assert digest(raw) == item['decoded_sha256']
    assert digest(path.read_bytes()) == item['archive_sha256']
    return json.loads(raw)


def stats(rows):
    return dict(prefill_median_tps=statistics.median(x['prompt_tps'] for x in rows),
        prefill_pooled_tps=sum(x['read_tokens'] for x in rows) * 1000 / sum(x['prompt_ms'] for x in rows),
        decode_median_tps=statistics.median(x['decode_tps'] for x in rows),
        decode_pooled_tps=sum(x['generated_tokens'] for x in rows) * 1000 / sum(x['decode_ms'] for x in rows))


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = {n: json.loads((B / n / 'record.json').read_text()) for n in ROOTS}
        for record in records.values():
            require_closed(record)
            assert not any('TOKEN' in key or 'API_KEY' in key for key in record['environment'])
            assert record['source_revision'] == '1eb89482a4afd20277ae0405780ed4f8eb98eb20'
            assert record['binary_sha256'] == '4a7e9eeeb5ddea917c8dc678f0df0ad4b5a181f8492c875bdcda5cb3026079bc'
            for item in record['source_manifest']:
                assert digest((Path(record['source_path']) / item['path']).read_bytes()) == item['sha256']
            assert digest(Path(record['result']['real_engine_exit']['engine']).read_bytes()) == record['binary_sha256']
            for path, pin in record['recipes'].items():
                assert digest((P / path).read_bytes()) == pin
        archive_index = []
        for name, record in records.items():
            root = B / name
            paths = [root / (x + '.json') for x in ['record', 'config', 'plan', 'result',
                'engine-start', 'engine-exit', 'sampler-summary', 'stderr-relay-summary']
                if (root / (x + '.json')).exists()]
            if 'stock_result_path' in record:
                stock_path = Path(record['stock_result_path'])
                assert digest(stock_path.read_bytes()) == record['stock_result_sha256']
                paths.append(stock_path)
            for original in paths:
                raw = original.read_bytes()
                relative = Path('receipts') / name / (original.name + '.gz')
                target = HERE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                packed = gzip.compress(raw, mtime=0)
                assert gzip.decompress(packed) == raw
                if target.exists():
                    assert target.read_bytes() == packed
                else:
                    target.write_bytes(packed)
                archive_index.append(dict(path=str(relative), original_path=str(original),
                    bytes=len(raw), decoded_sha256=digest(raw), archive_bytes=len(packed), archive_sha256=digest(packed)))
        write_json(HERE / 'receipts-index.json', archive_index)
        current = read_archive(D, 'receipts/main-primary-stock-bench-accepted-root-v2/record.json.gz')
        fp16 = read_archive(Q, 'receipts/main-primary-stock-fp16-root-v1/record.json.gz')
        old = records[ROOTS[1]]
        assert current['engine_args'] == fp16['engine_args'] == old['engine_args']
        assert current['standard_client_source'] == fp16['standard_client_source'] == old['standard_client_source']
        assert current['pack_metadata_sha256'] == fp16['pack_metadata_sha256'] == old['pack_metadata_sha256']
        assert current['MTP_metadata_sha256'] == fp16['MTP_metadata_sha256'] == old['MTP_metadata_sha256']
        arms = {'current_MMQ': current, 'current_FP16': fp16, 'old_1eb': old}
        for record in arms.values():
            rows = record['long_phase_samples']
            assert len(rows) == 3 and {x['run'] for x in rows} == {0, 1, 2}
            assert all(x['ok'] and x['read_tokens'] == x['prompt_tokens'] == 32049
                       and x['generated_tokens'] == 24 and x['reused_tokens'] == 0 for x in rows)
        write_json(HERE / 'stock-comparison.json', dict(
            arms={name: dict(source_revision=r.get('source_revision'),
                binary_sha256=r['binary_sha256'], samples=r['long_phase_samples'],
                statistics=stats(r['long_phase_samples']), environment=r['environment'])
                for name, r in arms.items()},
            engine_args=old['engine_args'], standard_client_source=old['standard_client_source'],
            prefill_median_ratio_old_to_current=stats(old['long_phase_samples'])['prefill_median_tps'] / stats(current['long_phase_samples'])['prefill_median_tps'],
            prefill_median_ratio_old_to_current_FP16=stats(old['long_phase_samples'])['prefill_median_tps'] / stats(fp16['long_phase_samples'])['prefill_median_tps'],
            exact_prompt_payload_identity_qualified=False,
            source_and_toolchain_matched=False, adopted=False, quality_qualified=False,
            physical_262144_qualified=False,
            scope='Same unmodified stock client, shape and engine CLI flags, one process per arm. Old/current source/toolchains differ. Numeric state/MTP work differs; short decode cannot establish sustained improvement.'))
        stderr_evidence = []
        for name in ROOTS:
            path = B / name / 'owned/engine.stderr'
            raw = path.read_bytes()
            lines = raw.decode().splitlines()
            selected = [dict(line=i + 1, text=line) for i, line in enumerate(lines)
                        if i < 32 or 'all queues completed' in line or
                        (name == ROOTS[1] and i >= len(lines) - 10) or
                        re.search(r'(?i)\b(error|fatal|failed|refused)\b', line)]
            stderr_evidence.append(dict(original_path=str(path), bytes=len(raw), sha256=digest(raw),
                selected=selected, scope='Startup/allocation/effective route, terminal long requests, diagnostics and normal queue drain; not a full trace'))
        write_json(HERE / 'model-diagnostics.json', stderr_evidence)
        reports = []
        names = {
            500: 'research-pre-Xe-decode-selected-dispatch-round500-v1.txt',
            501: 'research-pre-Xe-prefill-remaining-recovery-round501-v1.txt',
            502: 'research-primary-prefill-copy-queue-contract-round502-v1.txt',
            503: 'research-decode-VRAM-lifecycle-roofline-round503-v1.txt',
            504: 'research-threaded-skip-generation-round504-v1.txt',
            505: 'research-iq-single-current-cpu-route-round505-v1.txt',
            506: 'research-pre-Xe-stock-interpretation-round506-v1.txt',
            507: 'research-session-integration-claims-round507-v1.txt',
        }
        for number, name in names.items():
            original = B / 'research-20261009' / name
            raw = original.read_bytes()
            target = HERE / 'research' / name
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                assert target.read_bytes() == raw
            else:
                target.write_bytes(raw)
            reports.append(dict(round=number, model='gpt-6-luna', read_only=True,
                original_path=str(original), path=str(target.relative_to(HERE)), sha256=digest(raw)))
        write_json(HERE / 'research-index.json', dict(reports=reports,
            previous_registry='../2026-10-14-pre-Xe-replay/research-index.json',
            no_intervention_until_final=True, main_controls_builds_and_GPU=True,
            main_corrections=[
                'Main retracted its ring8-rounded-to16 claim. Both old/current ring_slots explicitly return STAGE8 for an explicit8; stream_all/threaded is inactive in the measured arms. No ring fix or ring-based causal claim.',
                'R501 incorrectly says current root has compact HC selectors and a separate copy queue. Current source6913 has neither; its init aliases copy=cs. Proposed patches are unapplied.',
                'R502 queue compatibility review does not close threaded skip host ordering. R504 independently proves the non-atomic used_of race when events are enabled; R8 arm does not exercise it. This is not proof of the historical GPU hang cause.',
                'R503 context256144 is a typo:262144. max(resource times) is an overlapped lower bound; non-overlap exclusive service would sum. Historical RAM read36.87GB/s,3processes,already exists in Git; not unknown or the DDR4 theoretical51.2GB/s.',
                'R505 describes the exact=1 NT1 route, not its activation in the canonical stock arm. STRATA_IQ_SINGLE_EXACT is absent in its complete recorded environment; native_expert.cpp parser returnsfalse on absence. No FP32 dequantization is used by the eligible exact helper.',
                'R506 calls the absent IQ_SINGLE_EXACT value unresolved, but the explicit whitelist and unchanged parser establish opt-inOFF. Its suggested current COMPACT2 toggle is nonexistent; port/qualify the proposed MoE/HC flags first, independently.',
                'R507 conflates the adopted0.1.41 baseline449.7 with the later copy-only candidate541.4. The8% decode loss withheld the copy-only candidate, not a repeal of all0.1.41 CPU/SYCL adoption. Both old source families are currently unselected.',
                'R507 real SYCL HTTP linkage is not wholly untested: current accepted stock and old replay use the real owned native under HTTP and exitnormally. Parallel slots, elasticVRAM and latest Niko functional delta remain unqualified.',
                'Historical and stock decode phases/settings differ. None of the short24token stock medians proves a decode restoration or a per-component adoption.',
            ]))
        historical = []
        sources = [
            ('7ea2650ba3ae1b15807a32a67b62d56131046569',
             'bench/results/2026-10-10-hardware-concurrency/HARDWARE_LIMITS.md', 'HARDWARE_LIMITS.md'),
            ('7ea2650ba3ae1b15807a32a67b62d56131046569',
             'bench/results/2026-10-10-parallel-round54/hardware/validated-three-process-summary.json', None),
            ('7ea2650ba3ae1b15807a32a67b62d56131046569',
             'bench/results/2026-10-10-cpu-capacity/fma-three-process-summary.json', 'fma-three-process-summary.json'),
            ('7ea2650ba3ae1b15807a32a67b62d56131046569',
             'bench/results/2026-10-10-cpu-capacity/README.md', 'CPU_CAPACITY.md'),
            ('7ea2650ba3ae1b15807a32a67b62d56131046569',
             'bench/results/2026-10-10-storage-capacity/three-process-summary.json', 'storage-three-process-summary.json'),
            ('b392b0f5f9919e4a5188e28c46273f5a3f459c5d',
             'bench/results/2026-10-10-hardware-architecture-decisions/ATTAINED_CAPACITY_AND_DECISION_BUDGET.md', 'ATTAINED_CAPACITY_AND_DECISION_BUDGET.md'),
            ('b392b0f5f9919e4a5188e28c46273f5a3f459c5d',
             'bench/results/2026-10-10-hardware-architecture-decisions/component-capacity-matrix-v2.json', 'component-capacity-matrix-v2.json'),
        ]
        for commit, source_path, name in sources:
            subprocess.run(['git', '-C', str(P), 'merge-base', '--is-ancestor', commit, 'HEAD'], check=True)
            original = subprocess.check_output(['git', '-C', str(P), 'show', commit + ':' + source_path])
            assert len(original) <= 1 << 20
            blob = subprocess.check_output(['git', '-C', str(P), 'rev-parse', commit + ':' + source_path], text=True).strip()
            item = dict(commit=commit, source_path=source_path, git_blob=blob,
                        bytes=len(original), sha256=digest(original))
            if name:
                target = HERE / 'historical-hardware' / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(original)
                item['retained'] = str(target.relative_to(HERE))
            else:
                full = json.loads(original)
                selected = [r for r in full['cells'] if
                    (r['cell'][0] == 'cpu' and r['cell'][1] in ['read', 'copy_cached', 'copy_nontemporal'] and r['cell'][2] == 6) or
                    (r['cell'][0] == 'transfer' and r['cell'][2] == 256 << 20) or
                    r['cell'] == ['vram', 'VRAM_queue_copy', 256 << 20] or
                    (r['cell'][0] == 'gemm' and ((r['cell'][2] == 8192 and r['cell'][3] == 'hot') or
                                               (r['cell'][2] in [80, 160] and r['cell'][3] == 'rotate8')))]
                assert len(selected) == 14
                for cell in selected:
                    assert len(cell['processes']) == 3
                    assert all(len(x['samples']) == 7 for x in cell['processes'])
                    assert statistics.median(x['median'] for x in cell['processes']) == cell['median_of_process_medians']
                target = HERE / 'historical-hardware/selected-individuals.json'
                write_json(target, dict(original_commit=commit, original_git_blob=blob,
                    original_sha256=digest(original), scope=full['scope'],
                    boot_id=full['boot_id'], processes=full['processes'], cells=selected,
                    selection='14 relevant operation cells; retain all21 individual samples/cell, not only medians. Other66 cells restore from pinned Git blob.'))
                item.update(derived=str(target.relative_to(HERE)), derived_sha256=digest(target.read_bytes()),
                            scope='Selected operation cells, not a byte-identical copy of full summary')
            historical.append(item)
        write_json(HERE / 'historical-hardware/source-index.json', historical)
        fixtures = [dict(path=str(B / ROOTS[0] / 'record.json'), consumer='check_reference_model.py stock4096 gate',
            byte_budget=1 << 20, next_review='Replace controller with canonical committed receipt gate before retiring raw record'),
            dict(path=str(B / 'main-pre-Xe-1eb-build-root-v2'), consumer='old source comparison and further component qualification',
                 byte_budget=4 << 30, next_review='When reference/component comparisons close'),
            dict(path=str(P / '.tools/pre-xe-1eb-source-v1'), consumer='old source SHA closure and component porting',
                 byte_budget=64 << 20, next_review='When reference/component comparisons close')]
        write_json(HERE / 'retention-plan.json', dict(raw_text_budget_per_job=64 << 20,
            fixtures=fixtures, status='Evidence closed; retirement follows verified commit',
            no_external_resume_checkpoint=True))
        print(json.dumps(dict(passed=True, closed_roots=ROOTS, archives=len(archive_index),
            reports=len(reports), GPU_executed=False)))


if __name__ == '__main__':
    main()
