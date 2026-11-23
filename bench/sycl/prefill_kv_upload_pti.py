"""Check the retained non-counter PTI collector on the V2 expert-bank fixture."""

from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import secrets
import sys
import time

sys.dont_write_bytecode = True
import prefill_kv_upload_capacity as capacity
import current_pti_admission as pti
import profile_split as profile
from engine_once import native_environment, profile_command
from run_q6_columns import validation_log

bank, basis, require = capacity.bank, capacity.basis, capacity.require
owner_module = capacity.capacity.owner_module
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'prefill_kv_upload_unitrace_fixture_v2'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('control', 'trace')
BUDGET = dict(wall=90, cpu=90, rss_cap=4 << 30, text_cap=4 << 20,
              file_cap=16 << 20, total_cap=64 << 20)
PLAN = dict(order=list(ORDER), fixture_arg='--split-upload-smoke',
            previous_failed_attempt='60a23f1c94818dc522167d14eb8f17d1f29774cf',
            acceptance='Match all three data, guard and cancellation cases. Verify actual providers, '
                       'RAW clock, the M2D/D2D/D2M copy ledger and three fills. Stop at the first failure.',
            scope='Current-build copy-path collector admission. Model coverage and overhead have separate gates.')


def inputs(revision):
    comparison = capacity.upload.experiment('store-time')
    native, speed, _, proof = capacity.prerequisites(comparison)
    full = basis.validation.committed(bank.RESULT, (capacity.KEY,))
    require(capacity.upload.gather.closed(full) and full['native_source_commit'] == native['native_source_commit']
            and full['binary'] == native['binaries']['strata_full_context_diagnostic']
            and full['outcomes']['full-copy']['final_slot_consumed']
            and full['outcomes']['full-direct']['final_slot_consumed'],
            'recover both closed physical-context controls on this native build')
    tools = pti.verify_tools(pti.committed_json(pti.RESULT, pti.CHECKOUT))
    pins = comparison.pins(revision) | bank.stage.source.pin(bank.P, [THIS, capacity.THIS])['source_sha256']
    return dict(source_commit=revision, sources_sha256=bank.signature(pins), sources_count=len(pins),
                native_source_commit=native['native_source_commit'], native_build_sha256=proof['build_sha256'],
                fixture=native['binaries']['sycl_expert_bank_test'], engine=native['binaries']['strata'],
                tools=tools, runtime_manifest=speed['runtime_manifest'], boot=speed['boot'], driver=speed['driver'],
                native_validation_probe=speed['native_validation_probe'],
                environment_template=speed['environment_template'])


def launch(row, label):
    root = ROOT / label
    template = row['inputs']['environment_template']
    env = {k: v.replace(template['STRATA_MEASUREMENT_ROOT'], str(root)) for k, v in template.items()}
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'), LD_DEBUG='libs',
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr')
    require(env['ZES_ENABLE_SYSMAN'] == env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == env['NEOReadDebugKeys'] == '1'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and not any(k.startswith(('UNITRACE_', 'ZET_')) for k in env),
            'preserve the qualified safety envelope')
    env = native_environment(row['inputs'], env)
    argv = [row['inputs']['fixture']['path'], PLAN['fixture_arg']]
    if label == 'trace':
        assets = row['inputs']['tools']['artifacts']
        config = dict(launcher=assets['unitrace'], mode='aggregate', session=row['session'])
        env.update(STRATA_UNITRACE_TARGET_PRELOAD=env.pop('LD_PRELOAD'),
                   UNITRACE_SystemTime='1', UNITRACE_FollowChildProcess='1',
                   UNITRACE_TraceOutputDir=str(root / 'timeline'), UNITRACE_ChromeKernelLogging='1',
                   UNITRACE_ChromeEventBufferSize='128')
        argv = profile_command(config, argv)
    return argv, env


def compact(row):
    value = dict(row['inputs'])
    value['environment_template_sha256'] = bank.signature(value.pop('environment_template'))
    return dict(row, inputs=value)


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    document = bank.previous.read(bank.RESULT)
    document[KEY] = compact(row)
    bank.stage.write(bank.RESULT, document)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh PTI probe only')
    revision = bank.stage.source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', inputs=inputs(revision), plan=PLAN, budget=BUDGET,
               session='StrataV2Bank' + secrets.token_hex(12), outcomes={},
               active=False, passed=False, complete=False, ownership_closed=True)
    row['launches'] = {name: launch_identity(row, name) for name in ORDER}
    ROOT.mkdir(mode=0o700)
    for label in ORDER:
        for suffix in ('tmp', 'cache', 'timeline'):
            (ROOT / label / suffix).mkdir(mode=0o700, parents=True)
    save(row)


def launch_identity(row, label):
    argv, env = launch(row, label)
    return dict(command=argv, environment_sha256=bank.signature(env))


def current(row):
    require(basis.validation.committed(bank.RESULT, (KEY,)) == compact(row)
            and row['inputs'] == inputs(row['inputs']['source_commit'])
            and row['plan'] == PLAN and row['budget'] == BUDGET
            and row['launches'] == {name: launch_identity(row, name) for name in ORDER},
            'commit and verify all fixture, provider and launch inputs')


def numerical(text):
    prefix = 'split_prefill_upload_fixture PASS stride=4096 '
    rows = re.findall('^' + prefix + r'coalesced=([01]) cancellation=([01]) groups=(\d+) experts=(\d+) '
                      r'copy_commands=(\d+) payload_bytes=(\d+) guards=PASS checked_drain=PASS$', text, re.M)
    values = [list(map(int, row)) for row in rows]
    require(values == [[0, 0, 5, 27, 27, 110592], [1, 0, 5, 27, 5, 110592], [1, 1, 1, 16, 1, 65536]]
            and text.count(prefix) == 3, 'unexpected copy, guard or cancellation fixture result')
    return values


def providers(log, row, env, traced):
    allowed = dict(bank.previous.read(Path(row['inputs']['runtime_manifest']['path'])))
    probe = row['inputs']['native_validation_probe']
    required = {probe['library']['path'], probe['loader']['path'],
                env['UR_ADAPTERS_FORCE_LOAD'], env['ZE_ENABLE_ALT_DRIVERS']}
    allowed[probe['library']['path']] = probe['library']['sha256']
    if traced:
        collector = row['inputs']['tools']['artifacts']['libunitrace_tool.so']
        required.add(collector['path'])
        allowed[collector['path']] = collector['sha256']
    allowed = {str(Path(p).resolve()): h for p, h in allowed.items()}
    loaded = {str(Path(s.strip()).resolve()) for s in
              re.findall(r'^\s*\d+:\s+calling init:\s*(/[^\n]+)$', log, re.M)}
    require({str(Path(p).resolve()) for p in required} <= loaded
            and any('libsycl.so' in p for p in loaded)
            and all(p in allowed and bank.sha(p) == allowed[p] for p in loaded),
            'initialized PTI/SYCL/Level Zero runtime providers differ')
    return dict(count=len(loaded), sha256=bank.signature({p: allowed[p] for p in sorted(loaded)}))


def trace_evidence(path, pid, begin, end):
    require(0 < path.stat().st_size <= BUDGET['file_cap'], 'bounded, flushed native timeline required')
    events = json.loads(path.read_text())['traceEvents']
    summary = profile.trace_summary(events, pid, begin, end)
    sizes = [65536, 20480, 4096, 16384, 4096]
    readers = 2 * sizes + sizes[:1]
    # PTI uses M for the fixture's std::vector storage (ZE_MEMORY_TYPE_UNKNOWN).
    expected = {'M2D': Counter([4096] * 27 + sizes + sizes[:1]), 'D2D': Counter(readers),
                'D2M': Counter([110592, 110592, 65536] + [256] * 6)}
    copies = {k: Counter() for k in expected}
    fills = []
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') != 'gpu_op':
            continue
        name = event['name']
        if name.startswith('zeCommandListAppendMemoryCopy'):
            match = re.search(r'\((M2D|D2D|D2M)\)\s*\[(\d+)\]$', name)
            require(match, 'copy direction or bytes absent')
            copies[match[1]][int(match[2])] += 1
        if name.startswith('zeCommandListAppendMemoryFill'):
            fills.append(int(re.search(r'\[(\d+)\]$', name)[1]))
    require(copies == expected and fills == [66048] * 3 and summary['gpu_events'] == 56,
            'GPU copy/fill inventory differs from the fixture')
    return dict(timeline_sha256=bank.sha(path), timeline_bytes=path.stat().st_size,
                copy_commands={k: sum(v.values()) for k, v in copies.items()},
                requested_copy_bytes={k: sum(n * count for n, count in v.items()) for k, v in copies.items()},
                fills=len(fills), gpu_events=summary['gpu_events'], host_events=summary['host_events'],
                gpu_interval_union_us=summary['gpu_union_us'], joined_native_events=True,
                activated_from_exec=True)


def run(label):
    row = bank.previous.read(ROOT / 'record.json')
    current(row)
    root = ROOT / label
    require(row['stage'] in ('prepared', 'running') and row['ownership_closed'] and not row['active']
            and list(row['outcomes']) == list(ORDER[:ORDER.index(label)])
            and not (root / 'execution.json').exists(), 'run each frozen case once in order')
    traced = label == 'trace'
    session = profile.session_path(row['session'])
    require(not session.exists(), 'existing collector session')
    cursor = profile.cursor()
    argv, env = launch(row, label)
    outcome = dict(complete=False, host_probe_accepted=False)
    owner = owner_module.Owner(root / 'owned',
        lambda: bank.stage.write(root / 'execution.json', dict(outcome=outcome, commands=owner.commands,
                                                                active=bool(owner.active))))
    clock = lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    try:
        begin = clock()
        entry, out, err = owner.run(label, argv, env, bank.P, **row['budget'])
        end = clock()
        outcome['native_exit_code'] = entry['exit_code']
        require(owner_module.closed(entry) and entry['exit_code'] == 0, 'fixture execution failed')
        evidence = numerical(out.read_text())
        log = err.read_text()
        probe = row['inputs']['native_validation_probe']
        activation = validation_log(log, entry['pid'], probe['loader']['path'], traced=True)
        outcome.update(fixtures=evidence, validation=activation, providers=providers(log, row, env, traced))
        if traced:
            require(evidence == row['outcomes']['control']['fixtures'], 'instrumented and control fixtures differ')
            trace, = (root / 'timeline').glob('*.json')
            outcome['trace'] = trace_evidence(trace, entry['pid'], begin, end)
        current(row)
        outcome.update(complete=True, host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if traced and owner.active is None and session.exists():
            try:
                tools = row['inputs']['tools']['artifacts']
                command, _, _ = owner.run('stop', [tools['unitrace']['path'], '--stop', row['session']],
                                          row['inputs']['environment_template'], bank.P, **row['budget'])
                require(owner_module.closed(command) and command['exit_code'] == 0 and not session.exists(),
                        'named collector session did not close')
            except BaseException as error:
                outcome['control_error'] = repr(error)
        basis.validation.finish(outcome, owner, cursor, row['inputs']['boot'])
        passed = outcome['passed'] = (outcome['passed'] and outcome.get('driver_after') == row['inputs']['driver']
                                     and 'control_error' not in outcome and not session.exists())
        row['outcomes'][label] = outcome
        done = passed and traced
        row.update(stage='closed' if done else 'running' if passed else 'failed', complete=done, passed=done,
                   ownership_closed=outcome['ownership_closed'], active=bool(owner.active))
        owner.persist()
        save(row)
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'trace', 'fresh_faults', 'ownership_closed', 'error')}),
          flush=True)
    require(passed, 'stop at the original PTI fixture outcome')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'select a frozen stage with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare() if action == 'prepare' else run(action)
