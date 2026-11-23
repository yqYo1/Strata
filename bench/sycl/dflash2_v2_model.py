"""Qualify the default DFlash2 selection on the retained V2 B570 server."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_v2_generation as v2
from check_dflash2_model import request_ids, wire
from engine_once import native_environment
from native_request_protocol import RequestProtocol

bank, basis, require = v2.bank, v2.basis, v2.require
sys.path.insert(0, str(bank.P))
from serve import server

restore = v2.clock.model.restore
owner_module = restore.owner_module
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_v2_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
PACK_SHA = 'cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1'
LIMITS = dict(wall=900, cpu=2400, rss_cap=104 << 30, text_cap=16 << 20,
              file_cap=32 << 20, total_cap=64 << 20)
INFO = dict(context='262144', spec='4', kv='int8', kv_resident='20480', expert_slots='1256')
SOURCES = [THIS, 'tools/sycl/check_dflash2_model.py', 'tools/sycl/native_request_protocol.py',
           'tools/sycl/engine_once.py', 'tools/strata_tokenizer.py', 'serve/server.py',
           'strata_control/drafting.py', 'strata_control/vram_fit.py', 'strata_control/user_args.py']


def inputs(revision):
    base = v2.baseline()
    cfg = dict(base['configuration'], user_args=['--spec', '4', '--prompt-cache', '1', '--prompt-cache-every', '0'])
    require('drafting' not in cfg, 'test the normal default resolver')
    args = server.engine_args(cfg, exe=cfg['exe'], announce=False)
    flags = {'--draft': 'dflash2', '--spec': '4', '--draft-conf': '0.6', '--draft-fallback': 'none',
             '--prefill': '8192', '--max-context': '262144', '--prompt-cache': '1', '--prompt-cache-every': '0'}
    for name, value in flags.items():
        require(args.count(name) == 1 and args[args.index(name) + 1] == value, 'unexpected launch flag: ' + name)
    require('--mtp' not in args and '--no-token-graph' in args and '--no-prefill-borrow' not in args,
            'expected DFlash2 alone with spec4 lease')
    pack, vocab = (Path(args[args.index(flag) + 1]) for flag in ('--draft-model', '--mtp-draft-vocab'))
    assets = {str(path): bank.sha(path) for path in (pack, vocab)}
    require(pack.stat().st_size == 522458112 and assets[str(pack)] == PACK_SHA
            and vocab.stat().st_size == 425196, 'the verified Q4 draft pack and subset vocabulary are required')
    plan = dict(root=str(ROOT), index=1, arm='candidate')
    env = v2.launch_environment(base, plan, split=False)
    env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='1', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0',
               STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_STATE_HASH='1')
    env = native_environment(base, env)
    short = request_ids(Path(cfg['tokenizer']))['ids']
    long = [i % 4096 for i in range(32769 - len(short))] + short
    requests = [(short, 32), (long, 1), (short, 32)]
    request_plan = [dict(prompt=len(ids), output=out, sha256=restore.signature_bytes(wire(ids, out)))
                    for ids, out in requests]
    pins = v2.experiment('split').pins(revision)
    pins.update(bank.stage.source.pin(bank.P, SOURCES)['source_sha256'])
    for name in SOURCES:
        original = bank.stage.source.git(bank.P, 'show', revision + ':' + name)
        require((bank.P / name).read_bytes() == original, 'changed DFlash2 input: ' + name)
    frozen = dict(source_commit=revision, source_inputs=dict(files=len(pins), sha256=bank.signature(pins)),
        native_source_commit=base['native_source_commit'], binary=base['binary'], build_sha256=base['native_build_sha256'],
        boot=base['boot'], driver=base['driver'], runtime_manifest=base['runtime_manifest'],
        drafter_assets=assets, command=[base['binary']['path'], '--serve', *args],
        environment_sha256=bank.signature(env), requests=request_plan, limits=LIMITS,
        scope='DFlash2 alone on a canonical DPAS-1 target head. Verify actual short/long/short routing, '
              'the checked 8192-token cache lease and exact short state after reset. Diagnostic clocks only.')
    return frozen, env, requests, base


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def request(io, ids, output, *, reused_tokens=0):
    data = wire(ids, output)
    for offset in range(0, len(data), 256 << 10):
        io.send(data[offset:offset + (256 << 10)])
    parser = RequestProtocol(len(ids), output, reused_tokens=reused_tokens)
    while parser.done is None:
        parser.feed(io.line(480))
    return dict(prompt=len(ids), generated=len(parser.tokens), output_sha256=bank.signature(parser.tokens),
                accepted=int(parser.done[6]), offered=int(parser.done[7]), finish=parser.done[5])


def inspect(text, pid, base, requests):
    proof = basis.validation.preload_evidence(text, pid)
    require(proof['loader'] == base['native_validation_probe']['loader']['path']
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', text),
            'unexpected MTP or runtime validation')
    dflash2 = [dict(tap_rows=int(rows), cold_prefix=int(cold), vram_bytes=int(size))
               for rows, cold, size in re.findall(
                   r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=([01]) vram_bytes=(\d+)$', text, re.M)]
    require(len(dflash2) == len(requests) and all(x['tap_rows'] > 0 and x['vram_bytes'] > 0 for x in dflash2)
            and text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1,
            'actual DFlash2 or target-head dispatch differs')
    states = re.findall(r'^strata serve: STATE_HASH L=.*$', text, re.M)
    require(len(states) == len(requests), 'missing state publications')
    active = [{k: v for k, v in re.findall(r'(\w+)=([0-9a-f,-]+)', line) if k in restore.ACTIVE_STATE}
              for line in states]
    require(all(set(s) == set(restore.ACTIVE_STATE) for s in active)
            and active[0] == active[2] and requests[0] == requests[2] and requests[0]['offered'] > 0
            and int(active[1]['L']) == 32769, 'invalid short reset or long handoff')
    store = restore.require_store(text, dict(direct=False), dict(read=32769))
    require(store['gpu_rows'] == 32768 and store['host_rows'] == 0
            and text.count('strata prefill policy: requested=throughput route=cache-window') == 2,
            'unexpected actual prefill placement')
    masks = re.findall(r'^strata cache loan: phase=masked generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready=0$', text, re.M)
    ready = re.findall(r'^strata cache loan: phase=ready generation=(\d+) chunk=8192 first=(\d+) lent=(\d+) decode_ready=1$', text, re.M)
    require(len(masks) == 1 and masks == ready and int(ready[0][2]) > 0
            and sum(map(int, ready[0][1:])) == 1256 and 'payload_match=1' in text, 'DFlash2 cache refill differs')
    return dict(validation=proof, states=active, placement=store, loan=list(map(int, ready[0])), dflash2=dflash2)


def run(row, env, requests, base):
    plan = row['plan']
    cursor, = (s.removeprefix('-- cursor: ') for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if s.startswith('-- cursor: '))
    owner = owner_module.Owner(ROOT / 'owned', lambda: bank.stage.write(
        ROOT / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    outcome = dict(complete=False, host_probe_accepted=False, requests=[])

    def interact(io, _entry):
        info = {}
        while True:
            line = io.line(600)
            if not line.startswith('INFO '):
                require(line == 'READY 262144 stop', 'unexpected startup or context')
                break
            info.update(x.split('=', 1) for x in line.split()[1:] if '=' in x)
        require(all(info.get(k) == v for k, v in INFO.items()), 'native drafter and placement differ')
        outcome['info'] = INFO
        for ids, output in requests:
            done = request(io, ids, output)
            outcome['requests'].append(done)
            print('DFlash2 completed input', len(ids), 'generated', done['generated'], 'drafts', done['offered'], flush=True)
        io.send(b'QUIT\n')
        io.exit(90)

    try:
        entry, _, stderr = owner.run('model', plan['command'], env, bank.P, interaction=interact, **LIMITS)
        outcome['native_exit_code'] = entry['exit_code']
        require(owner_module.closed(entry) and entry['exit_code'] == 0, 'native owner or exit differs')
        outcome.update(inspect(stderr.read_text(), entry['pid'], base, outcome['requests']))
        require(all(bank.sha(path) == expected for path, expected in plan['drafter_assets'].items()), 'drafter assets changed')
        outcome.update(complete=True, host_probe_accepted=True)
    except Exception as error:
        outcome['error'] = repr(error)
    finally:
        basis.validation.finish(outcome, owner, cursor, plan['boot'])
        outcome['passed'] = bool(outcome['passed'] and outcome['driver_after'] == plan['driver'])
        row.update(stage='closed' if outcome['passed'] else 'failed', outcome=outcome,
                   complete=outcome['complete'], passed=outcome['passed'], active=False,
                   ownership_closed=outcome['ownership_closed'])
        save(row)
    print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
    require(outcome['passed'], outcome.get('error', 'DFlash2 model check failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'run') and sys.executable == basis.reference.model.PYTHON,
            'use the pinned Python and select prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh DFlash2 plan required')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _, _, _ = inputs(revision)
            for folder in ('cache', 'tmp'):
                (ROOT / folder).mkdir(parents=True, mode=0o700)
            save(dict(plan=plan, stage='prepared', complete=False, passed=False, active=False, ownership_closed=True))
        else:
            row = bank.previous.read(ROOT / 'record.json')
            plan, env, requests, base = inputs(row['plan']['source_commit'])
            require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['plan'] == plan
                    and row['stage'] == 'prepared' and not (ROOT / 'execution.json').exists(),
                    'the committed DFlash2 plan or ownership differs')
            run(row, env, requests, base)
