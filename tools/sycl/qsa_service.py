#!/usr/bin/env python3
"""Bounded selected-QSA device-service diagnostic on the retained engine.

Repository LICENSE. PTI aggregates are diagnostics, not an independent UR
execution ledger, clean speed, physical traffic or hardware saturation proof.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervision_model as reference
import profile_aggregate

P, B, A = reference.P, reference.B, reference.A
ROOT = B / 'selected-qsa-service-v1'
RESULT = A / 'prefill-resources-current.json'
KEY = 'selected_qsa_service'
require, digest = reference.require, reference.digest
FILTER = 'prefix_batch_impl'
SOURCES = ('src/kernels/sycl/qsa_bf16_prefix.cpp',
    'include/strata/kernels/qsa_bf16_prefix.hpp', 'src/prefill/sycl/qsa_native16.cpp',
    'src/prefill/layer_major.inc', 'src/prefill/prefill.cpp', 'src/sycl/runtime.cpp')


def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def save(row):
    """Change only this result subtree; keep the original evidence verbatim."""
    raw = RESULT.read_text(); desired = json.loads(raw)
    value = {k: v for k, v in row.items() if k not in ('inputs', 'environment', 'configuration')}
    literal = json.dumps(KEY) + ':'
    text = json.dumps(value, indent=2).replace('\n', '\n    ')
    if KEY in desired['event_reconciliation']:
        require(raw.count(literal) == 1, 'ambiguous QSA diagnostic scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, count = json.JSONDecoder().raw_decode(raw[start:])
        raw = raw[:start] + text + raw[start + count:]
    else:
        anchor = '"event_reconciliation": {'
        require(raw.count(anchor) == 1, 'ambiguous event scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + text + ',', 1)
    desired['event_reconciliation'][KEY] = value
    require(json.loads(raw) == desired, 'QSA update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def inputs(revision):
    pins = reference.gen.benchmark_pins(revision)
    for name in (*SOURCES, 'tests/core/test_qsa_service.py'):
        raw = reference.gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact QSA input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def assets():
    events = reference.runtime.validation.committed(RESULT, ('event_reconciliation',))
    known = events['native_observation']['assets']
    result = {key: known[key] for key in ('launcher', 'collector')}
    for item in result.values(): require(digest(item['path']) == item['sha256'], 'collector artifact changed')
    require(events['ur_metadata_gpu']['passed'] and events['ur_subscriber_gpu']['passed'],
        'preserve existing direct fixture qualification')
    return result


def summarize(text, tokens):
    """Conserve the filtered PTI table and independently expected source calls."""
    require(tokens == 39260, 'unexpected actual input extent')
    table = profile_aggregate.device_summary(text)
    families = {'attention': [], 'pack': []}
    selected = 'prefix_batch_impl<33, false, false, true, true, true, true, false>'
    for row in table['rows']:
        name = row['name']
        require(selected in name, 'unexpected QSA template or unfiltered operation')
        geometry = re.search(r'\[SIMD(\d+) \{(\d+); 1; 1\} \{(\d+); 1; 1\}\]', name)
        require(geometry is not None, 'missing PTI kernel geometry')
        simd, groups, local = map(int, geometry.groups())
        require(simd in (16, 32) and groups > 0, 'invalid subgroup/geometry')
        if local == 128:
            require(simd == 16 and groups in (2 * 8192, 2 * 6491), 'attention extent differs')
            families['attention'].append(dict(row, query_rows=groups // 2))
        else:
            require(local == 64 and groups % 2 == 0, 'unexpected prefix pack extent')
            families['pack'].append(row)
    result = {}
    for family, rows in families.items():
        require(sum(r['calls'] for r in rows) == 60, family + ' source-call census differs')
        result[family] = dict(calls=60, service_ns=sum(r['total_ns'] for r in rows),
            rows=rows)
    require(sum(r['calls'] * r['query_rows'] for r in families['attention']) == 12 * (tokens - 1),
        'QSA query conservation differs')
    require(sum(r['calls'] for r in families['attention'] if r['query_rows'] == 8192) == 48 and
        sum(r['calls'] for r in families['attention'] if r['query_rows'] == 6491) == 12,
        'QSA chunk census differs')
    require(sum(v['service_ns'] for v in result.values()) == table['total_device_ns'], 'family service sum differs')
    return result


def original():
    row = json.loads((reference.ROOT / 'record.json').read_text())
    require(row['stage'] == 'closed' and row['passed'] and row['ownership_closed'] and not row['active'],
        'closed unchanged-engine reference required')
    reference.current(row)
    return row


def current(row):
    source = original()
    require(inputs(row['plan']['source_commit']) == row['inputs'] and assets() == row['assets'] and
        source['binary'] == row['binary'] and digest(reference.ROOT / 'record.json') == row['reference_sha256'],
        'source/native/collector/reference changed')
    require(reference.runtime.validation.committed(RESULT, ('event_reconciliation', KEY, 'plan')) == row['plan'],
        'commit immutable diagnostic plan')
    require(all(digest(ROOT / name) == pin for name, pin in row['plan']['generated'].items()), 'generated input changed')
    reference.models.model_pins(row['configuration'], source['model_assets'])
    return source


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text())['event_reconciliation'], 'single-use diagnostic')
    base = original(); source = reference.gen.git('rev-parse', 'HEAD').decode().strip()
    pinned = assets(); ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'trace'): (ROOT / name).mkdir()
    env = reference.environment(base, ROOT)
    env['UNITRACE_IncludeKernels'] = FILTER
    # Native preload remains target-only; engine_once's aggregate mode records
    # from exec and never enables full API logging or initializes a metric API.
    profile = dict(mode='aggregate', session='StrataQsa' + secrets.token_hex(12),
        **pinned, environment=dict(UNITRACE_SystemTime='1', UNITRACE_FollowChildProcess='1',
            UNITRACE_TraceOutputDir=str(ROOT / 'trace')))
    config = copy.deepcopy(base['configuration']); config['log'] = str(ROOT / 'engine.stderr')
    native_plan = dict(binary=base['binary'], native_validation_probe=base['probe'], native_profile=profile)
    write(ROOT / 'config.json', config); write(ROOT / 'plan.json', native_plan)
    with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
        '--prefill-only', '--prefill-tokens', '32768', '--think', '--runs', '1', '--profile-request-gates',
        '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(ROOT / 'bench')]
    row = dict(stage='prepared', binary=base['binary'], reference_sha256=digest(reference.ROOT / 'record.json'),
        inputs=inputs(source), configuration=config, environment=env, assets=pinned,
        plan=dict(source_commit=source, command=command,
            generated={name: digest(ROOT / name) for name in ('config.json', 'plan.json')},
            question='How much selected6e40 QSA attention and prefix-pack GPU service does one official32K+ request emit?',
            expected=dict(actual_input=39260, body_rows=39259, QSA_layers=12, chunks=[8192]*4+[6491],
                attention_calls=60, pack_calls=60, filter=FILTER),
            limits='Diagnostic PTI summed service only. No independent UR joins,complete graph/library ledger,lossless execution ledger,observer-cost qualification,clean speed,hardware saturation,physical bandwidth,decode/task/full-context or adoption claim. Existing UR subscriber logs every API with65536-record cap and is deliberately absent. Exact source-call and query counts,selected template/geometry,positive table clocks,owner/provider/pre-main/fault/journal/model checks remain required. Stop first failure;no retry.',
            budgets=dict(wall=1800, cpu=43200, rss_cap=104 << 30, text_cap=8 << 20, file_cap=32 << 20, total_cap=64 << 20),
            retention='One raw diagnostic<=64MiB consumed by service/count replay,then retire normal logs/JIT/generated plans. Keep source/configuration/binary/runtime/collector identities,compact family/shape samples,outcomes and limits in existing resource result. No new engine or collector binary.'),
        passed=False, complete=False, active=False, ownership_closed=True)
    save(row); print('Prepared selected-QSA aggregate diagnostic;commit plan before run.', flush=True)


def run(row):
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use committed plan required')
    base = current(row); native_plan = json.loads((ROOT / 'plan.json').read_text())
    session = native_plan['native_profile']['session']
    session_path = Path('/dev/shm/uctrl' + str(os.getuid()) + '_' + session)
    require(not session_path.exists(), 'preexisting collector session')
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    owner = control = None; outcome = dict(complete=False); rt = reference.runtime.validation.committed(reference.runtime.RESULT)
    host, manifest = reference.python_runtime.admitted(reference)
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist():
        row['active'] = bool((owner and owner.active) or (control and control.active))
        write(ROOT / 'execution.json', dict(outcome, active=row['active'], commands=owner.commands if owner else [],
            control_commands=control.commands if control else []))
    def interact(io, leader):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1200)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=0), 'unexpected request gate')
            delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            require(not (ROOT / 'validation.log').exists() or (ROOT / 'validation.log').stat().st_size == 0, 'parameter warning')
            require(reference.baseline.artifact_bytes(ROOT) <= 64 << 20, 'raw byte budget')
            claim = json.loads((ROOT / 'engine-start.json').read_text()); pid = claim['native']['pid']
            ident = reference.baseline.identity(pid)
            require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
                str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native identity differs')
            mapped = reference.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
            allowed = set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve()), str(Path(row['assets']['collector']['path']).resolve())}
            require(mapped <= allowed and {base['NEO'], row['assets']['collector']['path'], base['probe']['library']['path']} <= mapped,
                'unexpected native provider: ' + str(sorted(mapped - allowed)))
            actual = dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
            expected = reference.engine_once.native_environment(native_plan, row['environment'])
            # The qualified pre-exec launcher adds documented collection keys.
            require(all(actual.get(k.encode()) == v.encode() for k,v in expected.items()
                if k != 'LD_PRELOAD') and row['assets']['collector']['path'].encode() in actual.get(b'LD_PRELOAD',b'') and
                base['probe']['library']['path'].encode() in actual.get(b'LD_PRELOAD',b''), 'native preexec environment differs')
            wrapper = reference.baseline.identity(claim['wrapper']['pid']); require(wrapper and wrapper['sid'] == leader['pid'], 'wrapper ownership')
            server = reference.baseline.identity(wrapper['ppid']); require(server and server['sid'] == leader['pid'], 'server ownership')
            allowed_python = {v['realpath'] for v in manifest['inventory']['files'].values()}
            port = row['plan']['command'][row['plan']['command'].index('--port')+1]
            reference.python_runtime.check_running(leader['pid'], P/'tools/strata_bench.py',manifest['selected'],row['environment'],allowed_python,row['plan']['command'][2:])
            reference.python_runtime.check_running(server['pid'],P/'serve/server.py',manifest['selected'],row['environment'],allowed_python,
                ['--engine','strata','--config',str(ROOT/'config.json'),'--port',port])
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic()<deadline,'model teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage']='running'; owner=reference.baseline.Owner(ROOT/'owned',persist)
        control=reference.baseline.Owner(ROOT/'control-owned',persist)
        entry,_,_=owner.run('bench',row['plan']['command'],row['environment'],P,interaction=interact,**row['plan']['budgets'])
        require(reference.baseline.closed(entry) and entry['exit_code']==0,'model owner failed: '+str(entry.get('error')))
        start=json.loads((ROOT/'engine-start.json').read_text()); end=json.loads((ROOT/'engine-exit.json').read_text())
        require(end==dict(start,exit_code=0) and start['binary']==row['binary'] and start['native_profile']==native_plan['native_profile'] and
            start['args']==['--serve',*row['configuration']['args']] and start['boot_id']==base['boot'] and
            start['native']['sid']==start['wrapper']['sid']==entry['pid'],'native closure/recipe differs')
        result_file,=(ROOT/'bench').glob('*/results.json'); official=json.loads(result_file.read_text())
        samples=official['results']['prefill_only']['requests']; require(official['complete'] and not official['results']['failures'] and len(samples)==1,'official benchmark incomplete')
        s=samples[0]; require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and s['reasoning_observed'] and
            s['actual_read_tokens']==s['actual_prompt_tokens']==39260 and s['actual_reused_tokens']==0 and s['generated_tokens']==24,'workload differs')
        sys.path.insert(0,str(P/'tools')); from strata_bench import verify_request_journal
        verified=verify_request_journal(result_file.parent); require(verified['valid'] and verified['complete'],'request journal invalid')
        text=(ROOT/'engine.stderr').read_text(); reference.runtime.validation.preload_evidence(text,start['native']['pid'])
        require(text.count('strata prefill: layer-major candidate, tokens=39259 chunk=8192 residual_dtype=bf16')==1 and
            text.count('score_pitch=33 operand_layout=row-major softmax=parallel-exp-ascending-sum')==1 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10')==1 and
            not any(v in text for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),'selected route/runtime differs')
        families=summarize(text,39260)
        current(row); reference.models.model_pins(row['configuration'],base['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,diagnostic_counts_passed=True,families=families,
            request={k:s[k] for k in ('request_sha256','actual_read_tokens','actual_reused_tokens','generated_tokens','prompt_ms')},
            native=start['native'],model_hashes_matched=15,
            independent_UR_join_qualified=False,execution_ledger_qualified=False,observer_cost_qualified=False,
            hardware_saturation_qualified=False,performance_adopted=False)
    except BaseException as error: outcome['error']=repr(error)
    finally:
        if owner: reference.runtime.validation.finish(outcome,owner,cursor,base['boot'])
        if session_path.exists() and outcome.get('ownership_closed') and control is not None:
            try:
                stop,_,_=control.run('stop',[row['assets']['launcher']['path'],'--stop',session],row['environment'],P,
                    wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=64<<20)
                require(reference.baseline.closed(stop) and stop['exit_code']==0,'session stop failed')
            except BaseException as error: outcome['cleanup_error']=repr(error)
        outcome['session_removed']=not session_path.exists()
        row.update(outcome); row['active']=bool((owner and owner.active) or (control and control.active))
        row['ownership_closed']=bool(outcome.get('ownership_closed') and control and control.active is None and all(reference.baseline.closed(c) for c in control.commands))
        row['passed']=bool(row.get('passed') and row['complete'] and row['ownership_closed'] and row['session_removed'] and not row.get('cleanup_error'))
        row['stage']='diagnostic-counts-passed' if row['passed'] else 'failed'; persist(); save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','complete','error','session_removed')}),flush=True)
    return int(not row['passed'])


def main():
    require(sys.argv[1:] in (['prepare'],['run']), 'use qsa_service.py prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); reference.runtime.guards.no_model_owners([])
        if sys.argv[1]=='prepare': prepare(); return 0
        return run(json.loads((ROOT/'record.json').read_text()))


if __name__=='__main__': raise SystemExit(main())
