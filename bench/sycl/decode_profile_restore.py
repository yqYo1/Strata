"""Check the shipped Intel expert placement across prefill, decode and live conversation restore."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_restore as state
import cpu_work_census as census
import decode_footer as footer

model, bank, restore, require = state.model, state.bank, state.restore, state.require
ROOT = bank.B / 'decode-expert-profile-restore-v1'
KEY = 'decode_expert_profile_restore_v1'
PROFILES = {'stock': 'data/expert-profile.bin', 'intel': 'data/expert-profile-sycl.bin'}
ORDER = ('short-stock', 'short-intel', 'near-intel', 'near-stock')
SOURCES = [str(Path(__file__).resolve().relative_to(bank.P)), *state.SCRIPTS, *PROFILES.values()]


def same_build_controls():
    row = bank.basis.validation.committed(bank.RESULT, (state.KEY,))
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and list(row['outcomes']) == list(state.ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in row['outcomes'].values()),
            'the coherent server must first pass its original state and RESTORE controls')
    return row


def inspect_profile(log, row, case, requests):
    path = bank.P / PROFILES[case['arm']]
    startup = f'strata generate: profile {path}: 24576 ranked pairs, built for 24576 slots'
    require(log.splitlines().count(startup) == 1 and log.count('strata generate: profile ') == 1
            and log.count('pre-filled 1256 of 1256 slots from the profile; slot 0 verified') == 1
            and log.count('R4.2g PER-LAYER: each layer owns 26 slots (0..25).') == 1,
            'actual profile and fixed per-layer admission required')
    loans = re.findall(r'^strata loan audit: phase=layout chunk=(\d+) first=(\d+) slots=(\d+) '
                       r'bytes=(\d+) need=(\d+) lent=(\d+) mask_match=1$', log, re.M)
    expected = (8192, 132, 1256, 2321664000, 2319958528, 1124)
    require([tuple(map(int, loan)) for loan in loans] == [expected, expected]
            and [r['decode_cache_lent'] for r in requests] == [1124, 0, 1124],
            'actual fresh/restored loan layout or short cache route differs')
    headers = re.findall(footer.HEADER, log, re.M)
    clocks = re.findall(footer.TIMING, log, re.M)
    copies = re.findall(footer.COPY, log, re.M)
    positions = re.findall(r'^strata trace: window (\d+) (\d+)$', log, re.M)
    require([(int(pos), int(width)) for pos, width in positions] == [(r['prompt'] - 1, 1) for r in requests],
            'verifier must resume at the absolute final prompt position, including the restored suffix')
    bindings = []
    for req, header, clock, copy in zip(requests, headers, clocks, copies, strict=True):
        require(tuple(map(int, header)) == (req['prompt'], req['reused'], req['read'], 1, 0, 0)
                and int(clock[0]) == 1 and not any(map(int, copy[:6])), 'wrong verifier work or expert copies')
        ms = dict(zip(footer.LABELS, map(float, clock[1:]), strict=True))
        require(ms['average_T'] == ms['tokens_per_window'] == 1 and ms['PCIe'] == 0, 'expected one T1 window')
        bindings.append(dict(prompt_tokens=req['prompt'], windows=1, rows=1, cpu_jobs=int(copy[-2]),
                             cpu_entries=int(copy[-1]), pool_ms=ms['CPU_pool_run']))
    args = row['configuration']['args']
    formats = census.manifest_formats(Path(args[args.index('--pack') + 1]) / 'native_experts.txt')
    observed = census.extract(log, bindings, formats, census.expected_membership(path))
    return [{k: s[k] for k in ('request', 'membership_sha256', 'cpu_jobs', 'cpu_entries', 'vram_entries',
                                 'logical_job_blob_bytes')} for s in observed]


def experiment():
    ctx = bank.basis.core.source.device.load_module('decode_profile_state', model.__file__)
    ctx.KEY, ctx.ROOT, ctx.ORDER = KEY, ROOT, ORDER
    original_save, original_current = ctx.save, ctx.current

    def inputs():
        native, speed, build, fixture = state.inputs()
        gate = same_build_controls()
        require(gate['binary'] == native['binaries']['strata']
                and gate['native_source_commit'] == native['native_source_commit']
                and gate['build_sha256'] == build and gate['device_sha256'] == fixture,
                'state controls and their sources must belong to this executable')
        return native, speed, build, fixture

    def environment(row, case):
        return dict(model.environment(row, case), STRATA_CPU_WORK_CENSUS='1',
                    **{state.KV: '0', state.GATHER: '0'})

    def plan(row, name):
        size, arm = name.split('-')
        command = restore.argv(row)
        require(command.count('--expert-profile') == command.count('--expert-cache-per-layer') == 1
                and command[command.index('--expert-profile')+1] == str(bank.P / PROFILES['stock'])
                and all(command.count(k) == 1 and command[command.index(k)+1] == '0'
                        for k in ('--pcie-frac', '--adapt-swaps')), 'expected the fixed stock decode configuration')
        command[command.index('--expert-profile')+1] = str(bank.P / PROFILES[arm])
        case = dict(name=name, arm=arm, root=str(ROOT / name), near=size == 'near', borrow=False,
                    direct=False, command=command, requests=restore.input_plan(size == 'near'))
        case['environment_sha256'] = bank.signature(environment(row, case))
        return case

    def inspect(log, row, case, requests, pid):
        out = model.inspect(log, row, case, requests, pid)
        out['decode_placement'] = inspect_profile(log, row, case, requests)
        control = same_build_controls()['outcomes'][case['name'].split('-')[0] + '-base']
        for sample, ref in zip(requests, control['requests'], strict=True):
            keys = ('prompt', 'read', 'reused', 'publications', 'attention_store', 'attention_input',
                    'expert_bank', 'decode_cache_lent', 'prefill_route')
            require(all(sample.get(k) == ref.get(k) for k in keys), 'prefill work, placement or handoff changed')
        if case['arm'] == 'stock':
            restore.compare(out, control)
        return out

    def save(row):
        if row['stage'] == 'prepared':
            row['sources'].update(bank.stage.source.pin(bank.P, SOURCES)['source_sha256'])
            row['plan'] = dict(order=ORDER, profile_paths=PROFILES, state_sha256=bank.signature(same_build_controls()),
                scope='Diagnostic static placement. Check all 48 actual decode membership masks on each request, '
                      'all 1256 restored payloads on long requests, unchanged loan geometry and live restore. '
                      'Different profiles may produce different tokens. No timing or task-quality adoption.')
        original_save(row)

    def current(row):
        require(set(SOURCES) <= row['sources'].keys() and row['plan']['order'] == list(ORDER)
                and row['plan']['profile_paths'] == PROFILES
                and row['plan']['state_sha256'] == bank.signature(same_build_controls()),
                'missing profile inputs or changed state evidence')
        original_current(row)
        bank.basis.comparison.shared.models.model_pins(row['configuration'], row['model_assets'])

    ctx.inputs, ctx.environment, ctx.plan = inputs, environment, plan
    ctx.inspect, ctx.save, ctx.current = inspect, save, current
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and a planned stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        else:
            ctx.run(action)
