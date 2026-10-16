#!/usr/bin/env python3
"""Bind selected public session messages to immutable Git measurement evidence."""
import hashlib
import gzip
import json
import shutil
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
D = Path(__file__).parent
R = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/research-20261009')
REV = '16b1b04ffd77101c4b49543507c06fc71fbf2368'
BASE = 'bench/results/2026-10-05-sycl-prefill-scaling/'


def git(*argv):
    return subprocess.check_output(['git', *argv], cwd=P)


def write(name, value):
    (D/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    index = json.loads((D/'historical-source-index.json').read_text())
    needed = {
        'layer-major/README.md': 'historical-layer-major/README.md',
        'layer-major/paired-normal-wall/summary.json': 'historical-layer-major/summary.json',
        'cache-release/README.md': 'historical-cache-release/README.md',
        'upstream-v0.1.41-20261009/default32k/checks/integrated-build.json':
            'historical-v0141/integrated-build.json',
    }
    for arm in ('baseline', 'layer-major'):
        for i in range(1, 4):
            needed[f'layer-major/paired-normal-wall/{arm}-p{i}/run.json'] = \
                f'historical-layer-major/{arm}-p{i}.json'
    for original, retained in needed.items():
        source_path = BASE+original
        raw = git('show', f'{REV}:{source_path}')
        destination = D/retained
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        item = dict(commit=REV, source_path=source_path,
                    git_blob=git('rev-parse', f'{REV}:{source_path}').decode().strip(),
                    sha256=hashlib.sha256(raw).hexdigest(), retained=retained)
        index = [x for x in index if x['retained'] != retained]+[item]
    native_rev = '23268953314d12588fd3f496414a46bd426a7306'
    native_base = BASE+'native-expert-copy-v0141-20261009/'
    for original in ('quiet32k/README.md', 'quiet32k/summary.json',
                     'decode-repeat/README.md', 'decode-repeat/summary.json',
                     'decode-repeat/measurements.csv', 'full256k/README.md'):
        raw = git('show', f'{native_rev}:{native_base+original}')
        compressed = original.endswith('.csv')
        retained = 'historical-copy-candidate/'+original+('.gz' if compressed else '')
        destination = D/retained
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
        item = dict(commit=native_rev, source_path=native_base+original,
                    git_blob=git('rev-parse', f'{native_rev}:{native_base+original}').decode().strip(),
                    sha256=hashlib.sha256(raw).hexdigest(), retained=retained)
        if compressed:
            item.update(retained_encoding='gzip', decoded_bytes=len(raw),
                        archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        index = [x for x in index if x['source_path'] != item['source_path']]+[item]
    affinity_rev = 'aaee9775d0ba4425726ea7b33e2024ccddc57c51'
    affinity_path = 'bench/results/2026-10-09-sycl-stager-affinity/README.md'
    raw = git('show', f'{affinity_rev}:{affinity_path}')
    (D/'historical-stager-affinity.md').write_bytes(raw)
    retained = 'historical-stager-affinity.md'
    index = [x for x in index if x['retained'] != retained]+[dict(
        commit=affinity_rev, source_path=affinity_path,
        git_blob=git('rev-parse', f'{affinity_rev}:{affinity_path}').decode().strip(),
        sha256=hashlib.sha256(raw).hexdigest(), retained=retained)]
    write('historical-source-index.json', index)

    claims = json.loads((D/'session-performance-claims.json').read_text())
    offsets = [180227242, 185554690, 318209137, 338648075, 383038538,
               392698353, 404125838, 501758585, 540865064, 840141065,
               842825888, 853436774, 872899030]
    messages = []
    with Path(claims['original_path']).open('rb') as stream:
        for offset in offsets:
            stream.seek(offset)
            row = json.loads(stream.readline())
            payload = row['payload']
            assert row['type'] == 'response_item' and payload['type'] == 'message'
            assert payload['role'] in ('user', 'assistant')
            body = '\n'.join(x.get('text', '') for x in payload['content']
                             if x['type'] in ('input_text', 'output_text'))
            assert not any(s in body for s in ('HF_TOKEN', 'Authorization:', 'Bearer '))
            messages.append(dict(timestamp=row['timestamp'], byte_offset=offset,
                role=payload['role'], phase=payload.get('phase') or payload.get('channel'),
                message_sha256=hashlib.sha256(body.encode()).hexdigest(), text=body))
    write('session-decisions.json', dict(original_snapshot=claims['scanned_prefix_sha256'],
          first_Xe_request=claims['first_Xe_user_request'], messages=messages,
          scope='Selected public statements needed to reconcile measurements, adoption and active branch. No tool outputs, hidden reasoning or full session copy.'))

    registry = D/'research-index.json'
    research = [x for x in json.loads(registry.read_text()).get('reports', [])
                if x.get('round') not in (490, 491, 492, 493)] if registry.exists() else []
    for number, filename in (
        (490, 'research-primary-v0141-to-current-component-regression-round490-v1.txt'),
        (491, 'research-primary-pre-Xe-configuration-regression-round491-v1.txt'),
        (492, 'research-primary-pre-Xe-replay-safety-round492-v1.txt'),
        (493, 'research-primary-pre-Xe-build-replay-round493-v1.txt')):
        dest = D/'research'/filename
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(R/filename, dest)
        research.append(dict(round=number, model='gpt-6-luna', read_only=True,
            original=str(R/filename), retained=str(dest.relative_to(D)),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    write('research-index.json', dict(reports=research, no_intervention_until_final=True,
        main_work_in_parallel='Exact-session and original Git measurement/source reconciliation; no new model or GPU runs.',
        main_corrections=[
            'R490/R492 describe the current comparison as quiet GEN64. The actual current accepted speed baseline is unchanged Niko bench.sh/stock client, actual32049 input and24 outputs. No current quiet GEN64 speed baseline was run.',
            'R491 says IQ_SINGLE_EXACT is not pinned in its summary. The canonical current stock environment explicitly pins STRATA_IQ_SINGLE_EXACT=1.',
            'R491 methodological caution means causal attribution is unresolved; it does not erase the previous much faster Niko-derived own-fork measurements or their absent selected source paths.',
            'R490/R491 focus on 1eb/449.7. Main also recovered earlier own-fork layer-major645.85, preceding Xe, with a different65538 context and no MTP in timed CLI. It is not a qualified262144/MTP speed result.',
            'R492 Windows detached-thread finding alone is irrelevant to this Linux replay. Audit actual Linux observer construction/destruction order and retain a scoped join before observed object destruction; do not impose unrelated Windows implementation work.',
            'R493 missing effective compile/link details limit byte-exact reproduction. Source-defined flags and a clean source/binary build receipt are retained; missing data remains unknown.',
            'R489 prior scope was insufficient: it only compared the Oct4 root-native lineage and omitted the later pre-Xe migrated overlay and its measured controls. Its no-measured-loss implication is superseded.'
        ]))

    old = json.loads((D/'historical-v0141/tuned32k/summary.json').read_text())
    lm = json.loads((D/'historical-layer-major/summary.json').read_text())
    repeat = json.loads((D/'historical-copy-candidate/decode-repeat/summary.json').read_text())
    copy_quiet = json.loads((D/'historical-copy-candidate/quiet32k/summary.json').read_text())
    current = json.loads((P/'bench/results/2026-10-14-primary-model-baseline/stock-baseline.json').read_text())
    ancestry = {s:subprocess.run(['git', 'merge-base', '--is-ancestor', s, 'HEAD'], cwd=P).returncode == 0
                for s in (REV, native_rev, affinity_rev, old['engine_commits']['integrated'])}
    assert all(ancestry.values())
    write('comparison.json', dict(
        pre_Xe_first_request=claims['first_Xe_user_request'],
        old_layer_major=dict(binary_sha256=lm['binary_sha256'], actual_input=lm['tokens'],
            context=lm['max_context'], chunk=lm['chunk'], kv=lm['kv'],
            original_individual_samples=lm['pairs'], summary=lm['median_tokens_per_second'],
            method='Quiet CLI1-output, no MTP, all KV and32K residual rows resident; warm PLE; three independent alternating pairs.',
            capacity_limit='Only allocation+two-token execution at262146 was shown separately; no full physical262144 qualification for this timed configuration.'),
        old_v0141=dict(engine_commit=old['engine_commits']['integrated'],
            binary_sha256=old['binary_sha256']['integrated'],
            configuration_argv=old['configuration_argv_without_executable'],
            environment=old['actual_target_environment'],
            individual_samples=old['reads'], summary=old['summary'],
            method='Two processes/build; A/B/A/B32768 fresh inputs64-output. Separate first2 and later6.',
            matched_prior_control_conclusion=old['integrated_percent_change'],
            original_full_capacity_receipt_sha256=old['physical_gate_receipt_sha256'],
            capacity_scope='262140 input+4 output reaches262143;262144 input+1 output rejected; all saved262144 cells andRESTORE verified. Exact262144 input withzero outputs is not demonstrated by this retained gate.'),
        copy_queue_candidate=dict(quiet_comparison=copy_quiet,
            independent_decode_comparison=repeat,
            decision='Unadopted: fresh32K post-prefill decode regresses7.985%, paired95%CI[-12.806,-2.897] across6 independent processes per condition, despite541.376 PP and separate own physical capacity/RESTORE pass.',
            stale_message_correction='Oct10 statement that this candidate was waiting for256K is stale: own physical gate had completedOct9; the decode regression remained the adoption blocker.'),
        current_stock=current,
        historical_objects_reachable_from_primary=ancestry,
        observed_source_boundary=dict(old='1eb89482a4afd20277ae0405780ed4f8eb98eb20',
            old_resolver='root CMake returns into sycl/CMakeLists; sycl/src overlay precedes root',
            current_engine_base='4fa478f50f8da396e651e3ae862f8b1849d49b54',
            current_resolver='cmake/sycl_backend.cmake compiles root src/kernels/sycl and src/prefill/sycl',
            source_change_date_note='4fa is datedOct9 but its prior product boundary22e61b95 isOct4. Later qualified source is retained in history, not selected by current build.',
            history_preserving_merge='4526800257fe49c0db02a414f21b38dac2586e54'),
        causal_regression_attribution='Unresolved: source family, compiler/runtime, chunk/cache geometry, split grouping, inputs/output lengths and methods differ. Restore/evaluate the prior qualified components; do not dismiss the historical gap.',
        new_GPU_or_model_runs_executed=False))
    print(json.dumps(dict(historical_files=len(index), public_decisions=len(messages),
          research_reports=len(research), old_layer_major_median=lm['median_tokens_per_second'],
          old_v0141_later=old['summary']['integrated']['later_fresh_reads']), indent=2))


if __name__ == '__main__':
    main()
