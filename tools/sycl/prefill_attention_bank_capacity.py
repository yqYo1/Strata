#!/usr/bin/env python3
"""Check the matched physical-context extent of the attention bank store."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_attention_bank_build as build
import ur_validation as validation

P, B, RESULT, basis = build.P, build.B, build.RESULT, build.basis
require, sha, signature = build.require, build.sha, build.signature
ROOT = B / 'prefill-attention-bank-capacity-v2'
KEY = 'prefill_attention_bank_capacity_v2'
PREVIOUS = 'prefill_attention_bank_capacity_v1'
THIS = 'tools/sycl/prefill_attention_bank_capacity.py'
SELECTOR = 'STRATA_SYCL_PREFILL_DIRECT_ATTENTION_STORE'
BUILD_EVIDENCE = 'a3ad77136'
SPEED_EVIDENCE = 'b973b13dd1408aeae56ef17e7ff5536d57bf6654'
ORDER = ('short-copy', 'short-direct', 'full-copy', 'full-direct')
CAPACITY, CHUNK, LAYERS, ROW_BYTES = 262144, 8192, 48, 10240 * 2
BUDGET = dict(cpu=18000, rss_cap=104 << 30, text_cap=16 << 20,
              file_cap=32 << 20, total_cap=256 << 20)
owner_module = basis.comparison.build.supervision


def retained_build():
    row = build.read(build.ROOT / 'record.json')
    original = build.historic(BUILD_EVIDENCE, build.KEY)
    require(validation.committed(RESULT, (build.KEY,)) == original == build.builder.compact(row)
            and row['stage'] == 'built' and row['ownership_closed'] and not row['active']
            and row['passed'] and row['host_control']['passed'], 'closed measured native build required')
    native, proof, speed = basis.basis(expected=row['foundation'])
    hashes, _, recipe, history = build.inputs(native)
    require(row['source_files'] == hashes and row['recipe'] == recipe and row['history'] == history
            and proof == row['foundation'] and row['driver'] == build.builder.guards.driver_guard()
            and row['retained_decode_reference'] == build.builder.decode_reference(),
            'frozen native source, recipe, driver or decode reference differs')
    # The new diagnostic is a caller of the old executable. Recover each old
    # build control at its original revision and inspect existing live files.
    controls = row['controller_sources']
    recovered = build.stage.hashed(dict(build.stage.source.blobs(P, row['source_commit'], controls)))
    require(recovered == controls and all(sha(P / p) == h for p, h in controls.items()),
            'an original build control changed')
    require(all(sha(build.ROOT / 'source' / p) == h for p, h in hashes.items())
            and all(sha(Path(dep['root']) / p) == h for dep in native['dependencies']
                    for p, h in dep['source_sha256'].items())
            and all(sha(p) == h for p, h in row['configured_files'].items())
            and row['actual_inputs']['uncovered'] == 0
            and all(sha(p) == h for group in build.ACTUAL for p, h in row['actual_inputs'][group].items())
            and all(sha(v['path']) == v['sha256'] for v in row['binaries'].values()),
            'compiled source, dependencies, build configuration or ELF differs')
    timing = build.historic(SPEED_EVIDENCE, 'prefill_attention_bank_timing_v1')
    require(timing['binary'] == row['binaries']['strata'] and timing['native_source_commit'] == row['source_commit']
            and timing['stage'] == 'closed' and timing['ownership_closed'] and timing['decision']['all_warmed_pairs_faster'],
            'exact successful paired 64K evidence required')
    return row, speed


def source_pins(revision):
    pins = basis.comparison.build.controls(revision, controller=P / THIS)
    pins.update(build.stage.source.pin(P, list(basis.POLICIES))['source_sha256'])
    return pins


def environment(row, case):
    root = case['root']
    env = basis.comparison.environment(row, dict(root=root, heads=1))
    probe = row['native_validation_probe']
    # This standalone entrypoint cannot borrow the server's mixed scratch.
    # Retain the coordinated three-switch capacity adaptation from the
    # previous successful physical fill.
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE='0', STRATA_TRACE='1',
               STRATA_SYCL_PREFILL_MIXED_SCRATCH='0', STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='0',
               STRATA_SYCL_PREFILL_MOE_CENSUS='0', STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
               TMPDIR=root + '/tmp', XDG_CACHE_HOME=root + '/cache',
               LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    env[SELECTOR] = str(int(case['direct']))
    env.pop('UR_LOG_TRACING', None)
    for key in ('UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO'):
        env[key] = 'level:warning;flush:warning;output:stderr'
    require(env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS'] == '384'
            and env['STRATA_SYCL_PREFILL_QSA_MATH'] == 'bf16-prefix-ordered-exp', 'reference BF16 math/placement changed')
    return env


def one(text, pattern, label):
    entries = re.findall(pattern, text, re.M)
    require(len(entries) == 1, 'missing or duplicate ' + label)
    return entries[0]


def expected_store(rows, gpu_rows, maximum):
    """Count actual two-piece windows, including residents left outside them."""
    start, pieces, copied_rows = 0, 0, 0
    while start < rows:
        next_piece = start + CHUNK
        if next_piece < rows and min(next_piece + CHUNK, rows) <= gpu_rows:
            end = min(rows, gpu_rows, start + maximum)
            require(end > next_piece and (end == rows or end % CHUNK == 0),
                    'unexpected resident window shape')
            pieces += (end - start + CHUNK - 1) // CHUNK
            copied_rows += end - start
            start = end
        else:
            start = next_piece
    return pieces * LAYERS, copied_rows * LAYERS * ROW_BYTES


def evidence(stdout, stderr, row, case, pid):
    proof = validation.preload_evidence(stderr, pid)
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', stderr),
            'provider activation or runtime validation failed')
    values = one(stderr, r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=bf16 '
                 r'residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) '
                 r'residual_host=(\d+) residual_staging=(\d+) layer_cache=(\d+);', 'BF16 capacity ledger')
    n, chunk, total, gpu, gpu_bytes, host_bytes, staging, bank_bytes = map(int, values)
    require(n == case['tokens'] - 1 and chunk == CHUNK and 0 < gpu <= n and total == n * ROW_BYTES
            and gpu_bytes == gpu * ROW_BYTES and host_bytes == (n - gpu) * ROW_BYTES and staging == 0,
            'actual BF16 row/byte conservation failed')
    requested, slots, staged, slab, size, bank_gpu, limit = map(int, one(stderr,
        r'^strata MoE expert bank: requested_slots=(\d+) effective_slots=(\d+) staged=([01]) host_slab_bytes=(\d+) '
        r'device_bytes=(\d+) residual_gpu_rows=(\d+) allocation_limit_bytes=(\d+)', 'standalone expert bank'))
    require((requested, slots, staged, slab, size, bank_gpu) == (384, 512, 0, bank_bytes, bank_bytes, gpu)
            and bank_bytes == 512 * 2662400 and max(gpu_bytes, bank_bytes) <= limit,
            'standalone bank or allocation bound differs')
    mode, maximum, mixed = map(int, one(stderr,
        r'^strata MoE window configuration: requested_mode=10 effective_mode=(2|10) '
        r'maximum_rows=(\d+) mixed_fp16_bytes=(\d+)$', 'resident window'))
    require((mode == 2 and maximum == 16384 and mixed == 0)
            or (mode == 10 and maximum == min(81920, gpu) and mixed == maximum * 2560 * 2 <= limit),
            'unexpected actual standalone window geometry')
    pieces, bytes_ = expected_store(n, gpu, maximum)
    direct = int(case['direct'])
    receipt = tuple(map(int, one(stderr,
        r'^strata prefill attention bank: requested=([01]) pieces=(\d+) direct=(\d+) requested_D2D_bytes=(\d+)$',
        'attention store route')))
    require(pieces > 0 and receipt == (direct, pieces, pieces * direct, 0 if direct else bytes_),
            'actual destination, piece count or D2D volume differs')
    require(host_bytes > 0 if case['full'] else gpu == n,
            'physical case must exercise a host-backed tail and short case an all-GPU residual')
    prefix = int(one(stderr, r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
        r'KV=active-logical-prefix transient_bytes=(\d+); source=resident-Mode0 score_pitch=33 '
        r'operand_layout=row-major softmax=parallel-exp-ascending-sum$', 'BF16 QSA route'))
    require(prefix == ((n + 15) // 16) * 16 * 2056 <= limit
            and 'strata prefill: native16 BF16 HC residual/projection IO and MoE outputs;' in stderr,
            'typed prefix shape or native16 route differs')
    released = int(one(stderr, r'^strata trace: (\d+) MiB free after releasing the standalone prompt buffers$',
                       'retired standalone buffers'))
    count, observed_ms = one(stdout, r'^prefill\s+(\d+) tokens in ([0-9.]+) ms', 'actual batched extent')
    ids = [int(t) for t in one(stdout, r'^output\s*:(.*)$', 'output IDs').split()]
    markers = [s for s in stdout.splitlines() if s.startswith('strata FULL_CONTEXT ')]
    require(int(count) == n and float(observed_ms) > 0 and len(ids) == case['outputs']
            and all(0 <= t < row['vocab_size'] for t in ids)
            and markers == ([basis.MARKER] if case['full'] else []), 'output count or checked final slot differs')
    return dict(validation=proof, batched_rows=n, generated_ids=len(ids), output_ids_sha256=signature(ids),
                final_slot_consumed=bool(markers), gpu_rows=gpu, host_rows=n-gpu, gpu_bytes=gpu_bytes,
                host_bytes=host_bytes, bank_bytes=bank_bytes, allocation_limit_bytes=limit, qsa_bytes=prefix,
                window=dict(mode=mode, maximum_rows=maximum, mixed_fp16_bytes=mixed),
                store=dict(requested=direct, pieces=pieces, direct=receipt[2], requested_D2D_bytes=receipt[3]),
                post_release_free_mib=released, observed_prefill_ms=float(observed_ms), speed_eligible=False)


def check_parser():
    require(expected_store(262143, 114688, 16384) == (672, 112742891520), 'physical fixture ledger')
    require(expected_store(16384, 16384, 16384) == (96, 16106127360), 'short fixture ledger')
    require(expected_store(16385, 16385, 16384) == (96, 16106127360), 'isolated resident tail')
    require(expected_store(8193, 8193, 16384) == (96, 8054046720), 'in-window partial tail')


def token_bytes(n):
    return ''.join(str(i % 4096) + '\n' for i in range(n)).encode()


def make_case(row, name):
    full = name.startswith('full-')
    n, outputs = (CAPACITY, 0) if full else (16385, 4)
    root = ROOT / name
    path = ROOT / ('tokens-full.txt' if full else 'tokens-short.txt')
    args = row['configuration']['args']
    require(all(flag not in args for flag in ('--serve', '--tokens-file', '--max-new', '--consume-prompt')),
            'standalone arguments differ')
    for flag, value in (('--max-context', '262144'), ('--kv-resident', '20480'), ('--prefill', '8192'),
                        ('--expert-cache', '1256'), ('--spec', '4')):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == value, 'invalid fixed argument ' + flag)
    argv = [row['binary']['path'], *args, '--tokens-file', str(path), '--max-new', str(outputs)]
    if full:
        argv.append('--consume-prompt')
    case = dict(name=name, full=full, direct=name.endswith('-direct'), tokens=n, outputs=outputs, root=str(root),
                wall=2400 if full else 900, fixture=dict(path=str(path), sha256=sha(path)), command=argv)
    case['environment_sha256'] = signature(environment(row, case))
    return case


def save(row):
    build.stage.write(ROOT / 'record.json', row)
    result = build.read(RESULT)
    result[KEY] = basis.compact(row)
    build.stage.write(RESULT, result)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh committed campaign only')
    check_parser()
    previous = validation.committed(RESULT, (PREVIOUS,))
    require(previous['stage'] == 'failed' and not previous['GPU_executed']
            and previous['prepare_failure']['owners_started'] == 0, 'retain the original pre-launch refusal')
    native, old = retained_build()
    prior = validation.committed(RESULT, ('coherent_context_model_v2',))
    require(prior['outcomes']['full']['passed'] and prior['outcomes']['full']['final_slot_consumed']
            and prior['configuration'] == old['configuration'], 'matching previous standalone setup required')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    vocab = json.loads((Path(old['configuration']['tokenizer']) / 'vocab.json').read_text())
    require(set(range(4096)) <= set(vocab.values()), 'fixture vocabulary outside model')
    row = {k: old[k] for k in ('configuration', 'environment_template', 'model_assets', 'native_validation_probe',
                               'driver', 'boot', 'runtime_manifest')}
    row.update(stage='prepared', source_commit=revision, sources=source_pins(revision),
               native_source_commit=native['source_commit'], binary=native['binaries']['strata_full_context_diagnostic'],
               host_binary=native['binaries']['full_context_diagnostic_host_test'], vocab_size=max(vocab.values())+1,
               native_build_sha256=signature(build.builder.compact(native)), cases=[], outcomes={},
               active=False, ownership_closed=True, passed=False, complete=False, user_route_adopted=False,
               budgets=BUDGET, plan=dict(unit='Same diagnostic ELF and fixed standalone allocation, switch only the attention bank store.',
               gates='Host parser and existing same-build physical bounds, then both 16385-input/4-output cases '
                     'with identical greedy fixture IDs. Then both 262144-input/zero-output cases with '
                     'matched mixed-residency geometry and checked final token/refill marker.',
               stopping='Stop on first missing route, extent, changed input, validation issue, driver fault or failed ownership. No retries.',
               scope='Standalone capacities only. Server RESTORE, sustained decode and adoption remain separate.'),
               previous_standalone_evidence_sha256=signature(prior))
    ROOT.mkdir(mode=0o700)
    (ROOT / 'host').mkdir(mode=0o700)
    for name, n in (('short', 16385), ('full', CAPACITY)):
        (ROOT / ('tokens-' + name + '.txt')).write_bytes(token_bytes(n))
    for name in ORDER:
        case = make_case(row, name)
        for folder in ('tmp', 'cache'):
            (Path(case['root']) / folder).mkdir(parents=True, mode=0o700)
        row['cases'].append(case)
    for i in (0, 2):
        common = []
        for case in row['cases'][i:i+2]:
            common.append({k: v.replace(case['root'], '<private-case>')
                           for k, v in environment(row, case).items() if k != SELECTOR})
        require(common[0] == common[1], 'another process setting changed between arms')
    save(row)


def current(row):
    require(validation.committed(RESULT, (KEY,)) == basis.compact(row)
            and signature(validation.committed(RESULT, ('coherent_context_model_v2',)))
                == row['previous_standalone_evidence_sha256']
            and source_pins(row['source_commit']) == row['sources'], 'commit exact source and receipt')
    native, old = retained_build()
    require(row['native_build_sha256'] == signature(build.builder.compact(native))
            and row['binary'] == native['binaries']['strata_full_context_diagnostic']
            and row['host_binary'] == native['binaries']['full_context_diagnostic_host_test']
            and row['native_source_commit'] == native['source_commit']
            and all(row[k] == old[k] for k in ('configuration', 'environment_template', 'model_assets',
                                              'native_validation_probe', 'driver', 'boot', 'runtime_manifest')),
            'exact executable, platform or model identity differs')
    for name, case in zip(ORDER, row['cases'], strict=True):
        require(case == make_case(row, name)
                and Path(case['fixture']['path']).read_bytes() == token_bytes(case['tokens']),
                'frozen workload changed')


def run(name):
    row = build.read(ROOT / 'record.json')
    current(row)
    order = ('host', *ORDER)
    require(row['ownership_closed'] and not row['active'] and row['stage'] not in ('failed', 'closed')
            and list(row['outcomes']) == list(order[:order.index(name)])
            and all(o['passed'] for o in row['outcomes'].values()), 'earlier capacity gate incomplete')
    root = ROOT / name
    require(not (root / 'execution.json').exists(), 'inspect existing owner receipt')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
    outcome = dict(complete=False, GPU_executed=False, host_probe_accepted=False)
    owner = None

    def persist():
        build.stage.write(root / 'execution.json',
                          dict(outcome=outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))

    owner = owner_module.Owner(root / 'owned', persist)
    try:
        if name == 'host':
            check_parser()
            env = dict(row['environment_template'])
            env.pop('LD_PRELOAD', None)
            entry, out, _ = owner.run('bounds', [row['host_binary']['path']], env, P, wall=30, cpu=20,
                rss_cap=256 << 20, text_cap=64 << 10, file_cap=1 << 20, total_cap=4 << 20)
            require(owner_module.closed(entry) and entry['exit_code'] == 0
                    and 'full_context_request_and_terminal_bounds=PASS' in out.read_text()
                    and 'byte_order_and_bounds=PASS' in out.read_text(), 'same-build host controls failed')
        else:
            case = next(c for c in row['cases'] if c['name'] == name)
            outcome['native_execution_attempted'] = True
            entry, out, err = owner.run(name, case['command'], environment(row, case), P,
                                       wall=case['wall'], **row['budgets'])
            outcome.update(native_exit_code=entry['exit_code'], stdout_sha256=sha(out), stderr_sha256=sha(err))
            require(owner_module.closed(entry) and entry['exit_code'] == 0, 'native capacity exit or ownership failed')
            outcome.update(evidence(out.read_text(), err.read_text(), row, case, entry['pid']), GPU_executed=True)
            if case['direct']:
                previous = row['outcomes'][name.replace('-direct', '-copy')]
                for key in ('batched_rows', 'generated_ids', 'output_ids_sha256', 'final_slot_consumed',
                            'gpu_rows', 'host_rows', 'window'):
                    require(outcome[key] == previous[key], 'matched standalone work differs: ' + key)
        current(row)
        if name == ORDER[-1]:
            basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, row['boot'])
        passed = outcome['passed'] and outcome['complete'] and outcome.get('driver_after') == row['driver']
        outcome['passed'] = bool(passed)
        row['outcomes'][name] = outcome
        done = bool(passed and name == order[-1])
        row.update(stage='closed' if done else 'running' if passed else 'failed', complete=done, passed=done,
                   ownership_closed=outcome['ownership_closed'], active=bool(owner.active))
        persist()
        save(row)
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'complete', 'final_slot_consumed',
        'fresh_faults', 'ownership_closed', 'error')}), flush=True)
    require(passed, 'capacity gate stopped; inspect the recorded result')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'host', *ORDER), 'unplanned capacity action')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            run(action)
