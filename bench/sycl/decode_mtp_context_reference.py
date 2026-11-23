"""Verify the retained CLI's exact-fit refusal before building the MTP fix."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_kv_capacity as previous

bank, basis, require = previous.bank, previous.basis, previous.require
capacity, model = previous.capacity, previous.model
ROOT = bank.B / 'decode-mtp-context-reference-v1'
KEY = 'decode_mtp_context_reference_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
SOURCE = 'fde0cdd11a72a50aaa0e2105de24996589aef52c'
CONTEXT, TOKENS, OUTPUTS = 16387, 16385, 2
INPUT = ROOT / 'tokens.txt'
PROCESS = ROOT / 'reference'
PLAN = dict(context=CONTEXT, input_tokens=TOKENS, max_new=OUTPUTS, spec=4,
    expected_exit=2, first_verifier_position=TOKENS - 1, expected_refusal_position=TOKENS,
    scope='The original native-pack first window is width one. Its width-four draft writes at '
          'most through the last valid cell; the CLI refuses the next width-four verifier '
          'window before dispatch. Use this same admitted request on the fixed CLI.')
LIMITS = dict(wall=420, cpu=2400, rss_cap=96 << 30, text_cap=4 << 20,
              file_cap=8 << 20, total_cap=48 << 20)


def inputs():
    native, speed, _, proof = previous.prerequisites()
    require(native['native_source_commit'] == SOURCE and native['ownership_closed']
            and not native['active'] and native['passed'] and native['complete'],
            'retain the exact measured prefill and MTP executable')
    return dict({k: speed[k] for k in model.FIELDS}, native_source_commit=SOURCE,
                binary=native['binaries']['strata'], qualification=proof)


def command(row):
    args = list(row['configuration']['args'])
    for arg, expected in (('--max-context', '262144'), ('--kv-resident', '20480'), ('--spec', '4'),
                          ('--suffix-draft', '0'), ('--prefill', '8192')):
        require(args.count(arg) == 1 and args[args.index(arg) + 1] == expected,
                'unexpected controlled argument ' + arg)
    require(all(args.count(arg) == 1 for arg in ('--native', '--pack', '--mtp', '--greedy', '--no-coupled-draft'))
            and not any(arg in args for arg in ('--serve', '--tokens-file', '--max-new', '--consume-prompt',
                                              '--spec-min-p', '--mtp-max-t')), 'require native fixed-width MTP')
    args[args.index('--max-context') + 1] = str(CONTEXT)
    return [row['binary']['path'], *args, '--tokens-file', str(INPUT), '--max-new', str(OUTPUTS)]


def environment(row):
    env = previous.experiment().environment(row, dict(root=str(PROCESS), name='short-reuse', direct=False))
    require('STRATA_SYCL_MTP_DECODE_COMMITTED_PREFIX' not in env, 'require the original drafting route')
    return env


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = basis.compact(row)
    bank.stage.write(bank.RESULT, result)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'one fresh reference plan')
    row = inputs()
    pins = bank.stage.source.pin(bank.P, [THIS, *previous.SOURCES])
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (PROCESS / name).mkdir(parents=True, mode=0o700)
    INPUT.write_bytes(capacity.token_bytes(TOKENS))
    row.update(stage='prepared', source_commit=pins['revision'], sources=pins['source_sha256'],
               plan=PLAN, limits=LIMITS, command=command(row), environment_sha256=bank.signature(environment(row)),
               input_sha256=bank.sha(INPUT), active=False, ownership_closed=True, complete=False, passed=False)
    save(row)


def current(row):
    require(basis.validation.committed(bank.RESULT, (KEY,)) == basis.compact(row)
            and row['sources'] == bank.stage.source.pin(bank.P, list(row['sources']))['source_sha256']
            and all(row[k] == v for k, v in inputs().items()) and row['plan'] == PLAN and row['limits'] == LIMITS
            and row['command'] == command(row) and row['environment_sha256'] == bank.signature(environment(row))
            and INPUT.read_bytes() == capacity.token_bytes(TOKENS) and row['input_sha256'] == bank.sha(INPUT),
            'frozen CLI, workload, source or runtime differs')


def run():
    row = bank.previous.read(ROOT / 'record.json')
    current(row)
    execution = PROCESS / 'execution.json'
    require(row['stage'] == 'prepared' and not row['active'] and row['ownership_closed']
            and not execution.exists(), 'inspect the original reference attempt')
    cursor, = (line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
    outcome = dict(host_probe_accepted=False)
    owner = bank.builder.ownership.Owner(PROCESS / 'owned', lambda: bank.stage.write(execution, owner.commands))
    try:
        entry, _, err = owner.run('exact-fit', row['command'], environment(row), bank.P, **LIMITS)
        outcome['native_exit_code'] = entry['exit_code']
        log = err.read_text()
        proof = basis.validation.preload_evidence(log, entry['pid'])
        prefix = capacity.one(log, r'^strata generate: prefill (\d+) tokens in \d+ chunks,', 'actual prefill extent')
        refusal = capacity.one(log, r'^strata generate: ran out of context at position (\d+)$', 'CLI context refusal')
        require(bank.builder.ownership.closed(entry) and entry['exit_code'] == PLAN['expected_exit']
                and proof['loader'] == row['native_validation_probe']['loader']['path']
                and int(prefix) == TOKENS - 1 and int(refusal) == TOKENS
                and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', log),
                'expected only the source-proven pre-dispatch refusal')
        current(row)
        outcome.update(prefill_tokens=int(prefix), refusal_position=int(refusal), host_probe_accepted=True)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, row['boot'])
        passed = outcome['passed'] and outcome.get('driver_after') == row['driver']
        row.update(stage='closed' if passed else 'failed', outcome=outcome, active=bool(owner.active),
                   ownership_closed=outcome['ownership_closed'], complete=bool(passed), passed=bool(passed))
        save(row)
    print(dict(stage=row['stage'], **outcome), flush=True)
    require(passed, 'the original CLI reference did not meet the diagnostic gate')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'run') and sys.executable == basis.reference.model.PYTHON,
            'use the pinned interpreter and prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        prepare() if action == 'prepare' else run()
