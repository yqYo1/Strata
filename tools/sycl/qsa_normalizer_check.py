#!/usr/bin/env python3
"""Owned lossless numerical gates, separate from normalizer speed and adoption.

Authored under repository LICENSE. No source edits, runtime environment changes,
unqualified division tolerance or automatic model advancement.
"""
import fcntl
from fractions import Fraction
import json
from pathlib import Path
import re
import select
import subprocess
import sys
sys.dont_write_bytecode = True
import qsa_head_math as m
import qsa_trace_interval as interval
n, q, r, P, B = m.n, m.q, m.r, m.P, m.B
ROOT = m.ROOT/'checks-v1'
KEY = 'lossless_numerics'
REVIEW = 'qsa_strict_fp32_target_contract_20261029'
INDEX = m.RESULT.with_name('research-index.json')
require, digest, write = m.require, m.digest, m.write


def sources(revision):
    return q.sources(revision, Path(__file__).resolve(),
                     [m.CPP, n.ORACLE, 'tools/test_qsa_trace_interval.py'])


def contract():
    review = r.validation.committed(INDEX, (REVIEW, 'native_review'))
    row = review['target_contract']
    require(row['source_qualified'] and row['FP32_RNE_gradual'] and row['BF16_RNE'],
            'qualified actual target arithmetic contract first')
    require(row['required_single_fp_config'] and
            isinstance(row['required_single_fp_config'], list), 'required FP32 capabilities')
    require(Fraction(row['division_absolute_ulp_bound']) > 0,
            'explicit source-justified division accuracy required')
    return row


def built():
    record, native, _, rt = m.native()
    row = json.loads((m.ROOT/'record.json').read_text())
    require(row['stage'] == 'built' and row['passed'] and row['ownership_closed'] and
            not row['active'] and m.compact(row) == record[n.KEY][m.KEY],
            'closed exact lossless fixture build required')
    m.admitted(row)
    return record, row, rt


def prepare(mode, page):
    require((mode, page) in [('tiny',4), ('component',4), ('component',16)], 'bounded gate order')
    record, fixture, rt = built()
    arithmetic = contract()
    gates = record[n.KEY].setdefault(KEY, {})
    name = mode+str(page)
    folder = ROOT/name
    require(name not in gates and not folder.exists(), 'fresh numerical gate only')
    if mode == 'component':
        previous = gates['tiny4' if page == 4 else 'component4']['outcome']
        require(previous['passed'] and previous['complete'] and previous['ownership_closed'] and
                previous['fresh_faults'] == [], 'prior numerical gate required')
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=P, text=True).strip()
    pins = sources(revision)
    folder.mkdir(mode=0o700, parents=True)
    for part in ['tmp','cache']:
        (folder/part).mkdir()
    env = dict(rt['environment'], TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'),
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        STRATA_TEST_MAIN_GATES='1',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
    if mode == 'component':
        env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    row = dict(stage='prepared', source_commit=revision, source_sha256=pins,
        fixture_source_commit=fixture['source_commit'], native_source_commit=fixture['native_source_commit'],
        binary=fixture['binaries'][str(page)], command=[fixture['binaries'][str(page)]['path'], '--'+mode],
        arithmetic=arithmetic, environment=env, boot=rt['boot'], mode=mode, page=page,
        budgets=dict(wall=600, cpu=1200, rss_cap=8<<30, text_cap=64<<20, file_cap=128<<20, total_cap=256<<20),
        criterion='Source-qualified RNE/gradual FP32 and BF16RNE;actual single_fp_config before enqueue. Same32actual p/BF16PV operands and exactFP32A,m/alpha/branch bits;serialascendingRNE fold,depth31/16positive-sum intervals and exactRNEonlineFMA,derived normalizer/ratio intervals with source-qualified targetFDivULP. Normal/capture fulloutput parity,independenthosttensororacle,7traceadmissionrefusals,guards/inputs/masks/invalidwidth/nonfinite branches,head/query/KV/case coverage. Tinyactual traced4kernelABI/WG/local extents. Allproviders/pre-mainvalidation/immutableenv/owners/faults mustpass. No empirical task tolerance or exacttext requirement.',
        stopping='One fresh process;stop first source/runtime/correctness/validation/fault/budget failure. No retries or exclusions;preserve failure. Component Page4requires tiny4;Page16requires component4.',
        limits='Component-only. Production2051cap=65tiles;component262144addresses do not fill fullmodelcontext. PubliccapINT_MAXmayhave67108864tiles:unchangedshape/bounds are sourcechecked,not allocatedhere. No modelquality,decode,statehandoff,physicalfullcontext or performance qualification.',
        retention='Keep compact counts/source/contract/decision only after gate verification;retire successful fulltrace/capture stdout and repeated provider evidence at candidate closure. Rawlosslesscapture<=64MiB remains only until parser/correspondence verification.',
        complete=False, passed=False, active=False, ownership_closed=True)
    write(folder/'plan.json', row)
    gates[name] = row
    write(m.RESULT, record)
    print('Lossless numerical plan prepared;commit before run.', flush=True)


def run(mode, page):
    record = r.validation.committed(m.RESULT)
    row = record[n.KEY][KEY][mode+str(page)]
    folder = ROOT/(mode+str(page))
    require(row['stage'] == 'prepared' and row == json.loads((folder/'plan.json').read_text()) and
            not (folder/'owned').exists(), 'fresh committed numerical plan')
    _, fixture, rt = built()
    require(sources(row['source_commit']) == row['source_sha256'] and row['arithmetic'] == contract() and
            row['binary'] == fixture['binaries'][str(page)], 'pinned numerical inputs differ')
    text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
                                  text=True, timeout=15)
    cursor = next(v.split(': ',1)[1] for v in text.splitlines() if v.startswith('-- cursor: '))
    outcome = dict(complete=False, passed=False, host_probe_accepted=False)
    owner = None
    def persist():
        write(folder/'record.json', dict(outcome, active=bool(owner and owner.active is not None),
                                       commands=owner.commands if owner else []))
    owner = r.supervision.Owner(folder/'owned', persist)
    try:
        def interact(io, entry):
            require(io.line(30) == 'QSA_BF16_READY before-runtime', 'entry gate differs')
            io.send(b'X')
            configs = []
            while (line := io.line(90)) != 'QSA_BF16_READY runtime-ready':
                if line.startswith('QSA_FP_CONFIG '):
                    configs.append(list(map(int, line.split()[1:])))
            require(len(configs) == 1 and set(row['arithmetic']['required_single_fp_config']) <= set(configs[0]),
                    'target FP32 capability missing before enqueue')
            outcome['single_fp_config'] = configs[0]
            q.observed_native(io.proc.pid, row['environment'], rt, row['binary'])
            io.send(b'X')
            while (line := io.line(550)) != 'QSA_BF16_READY after-cleanup':
                if line.startswith('case cells='):
                    print(line, flush=True)
            q.observed_native(io.proc.pid, row['environment'], rt, row['binary'])
            io.send(b'X')
            io.proc.stdin.close()
            require(io.line(20).startswith('QSA_NORM_DONE '), 'final completion marker')
            while not io.eof or not io.exit_watch.poll(0):
                io.poll()
                if select.select([io.proc.stdout], [], [], .02)[0]:
                    io.read();io.pending.clear()
            io.exit(10)
        entry, out, err = owner.run(mode+str(page), row['command'], row['environment'], P,
                                    interaction=interact, **row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                'numerical native process failed:'+err.read_text()[-2000:])
        proof = r.validation.preload_evidence(err.read_text(), entry['pid'])
        require(proof['loader'] == rt['preload']['loader'], 'pre-main provider differs')
        validation = folder/'validation.log'
        require(not validation.exists() or not validation.stat().st_size, 'parameter validation warning')
        require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|QSA_NORM_ERROR|terminate called|cleanup failed',
                              err.read_text()), 'runtime/numerical error')
        arithmetic = row['arithmetic']
        parsed = interval.check(out.read_text(), mode=mode, page=page, arithmetic_mode='RNE_gradual',
            division_ulps=Fraction(arithmetic['division_absolute_ulp_bound']), bf16_mode='RNE')
        if mode == 'tiny':
            props = record[n.KEY]['properties4']['outcome']['properties']
            outcome['dispatch'] = n.dispatches((folder/'api.trace').read_text(), props)
        m.admitted(fixture)
        require(sources(row['source_commit']) == row['source_sha256'] and arithmetic == contract(),
                'qualification input changed during execution')
        outcome.update(complete=True, host_probe_accepted=True, numerical=parsed,
                       parameter_validation=True, full_model_context_qualified=False,
                       task_qualified=False, performance_qualified=False, advance_to_model=False)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        r.validation.finish(outcome, owner, cursor, row['boot'])
        persist()
        row.update(stage='closed' if outcome['passed'] else 'failed', outcome=outcome)
        write(m.RESULT, record)
    print(json.dumps(outcome), flush=True)
    require(outcome['passed'], 'lossless numerical gate failed')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare, 'run':run}[sys.argv[1]](sys.argv[2], int(sys.argv[3]))
