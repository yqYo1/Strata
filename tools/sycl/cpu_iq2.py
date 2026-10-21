#!/usr/bin/env python3
"""Finite Main-owned fused direct IQ2_S NT1 component. CPU only; no model dispatch."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_roofline as base
import measurement_guards as guards
import measurement_owner as owner

P, B = base.P, base.B
ROOT = B/'cpu-iq2-pair-v1'
RESULT = base.A/'cpu-iq2-current.json'
PREFIXES = (*base.PREFIXES, 'bench/sycl/cpu_iq2', 'tools/sycl/cpu_iq2.py')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def gu_hash(row):
    path = Path(row['shard']); before = base.identity(path); h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        for e in range(96):
            for offset in (row['gate'], row['up']):
                start = offset+e*row['gu_bytes']
                base.need(start+row['gu_bytes'] <= before['bytes'], 'GU slice extent')
                f.seek(start); raw = f.read(row['gu_bytes']); base.need(len(raw) == row['gu_bytes'], 'short GU read')
                h.update(raw)
    base.need(base.identity(path) == before, 'fixture changed during hash')
    return h.hexdigest()


def prepare():
    base.need(not ROOT.exists(), 'single-use component root exists')
    previous = json.loads(RESULT.read_text())
    base.need(previous['stage']=='closed' and previous['ownership_closed'] and not previous['performance_gate_passed'] and previous.get('artifact_retirement'), 'close/retire previous comparison first')
    previous_commit = base.git(P,'log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip()
    revision = base.git(P, 'rev-parse', 'HEAD').decode().strip()
    row = next(r for r in base.formats() if (r['gu'], r['down']) == (22, 20))
    row['experts'] = 96; row['GU_sha256'] = gu_hash(row); row.pop('down_offset')
    config = ['/usr/bin/cmake', '-S', str(P/'bench/sycl/cpu_iq2'), '-B', str(ROOT/'build'),
        '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER='+str(base.LLVM/'bin/clang'),
        '-DCMAKE_CXX_COMPILER='+str(base.LLVM/'bin/clang++'), '-DSTRATA_GGML_DIR='+str(base.GGML)]
    plan = dict(source=base.source_pin(P, revision, PREFIXES),
        ggml=base.source_pin(base.GGML, base.GGML_REV, ('ggml',)),
        tools={str(Path(x).resolve()):base.digest(x) for x in
            (base.LLVM/'bin/clang', base.LLVM/'bin/clang++', '/usr/bin/cmake', '/usr/bin/ninja')},
        environment=base.environment(), machine=base.machine(), format=row,
        metadata_sha256=base.digest(base.METADATA), model_identity=base.identity(row['shard']),
        configure=config, build=['/usr/bin/cmake', '--build', str(ROOT/'build'), '--target', 'cpu_iq2_pair', '--parallel', '6'],
        protocol=dict(question='Does direct gate/up fusion reduce NT1 activation-load/loop cost without changing arithmetic?',
            methods=['linked production iq256_single_gu_rows(type22)', 'direct fused gate/up rolled8-part loop', 'direct fused gate/up fully expanded8 parts'],
            mechanism='Same packed IQ2_S codebook/sign/scale arithmetic. Gate and up share each Q8 activation load and loop control; two FP32 accumulators retain the original eight integer lanes, per-block FMA and final tree. Compare rolled and expanded loops for register/instruction tradeoffs. No LUT or expanded weights.',
            scope='Single pinned physical CPU, H2560/FF640, full GU/SwiGLU, NT1. Only GU; Down and production pool/model are later gates. No preprocessed activation table, weight conversion or hidden setup.',
            access='Hot6 distinct actual GU pairs6MiB; stream96 distinct actual GU pairs100761600B (>96MiB), no repeat within pass. Same original packed weights and direct codebooks.',
            numerical='First exhaustive4 valid activation patterns x1024 grids x256 signs, exact four-lane integer checks. Then1280 synthetic padded gate/up row dots match GGML bits for both variants, partial634 GU rows match current helper, zero/negative/subnormal finite scales and valid extreme Q8 codes. All96 real experts match complete GU/SwiGLU bits, finite outputs, immutable inputs and canaries before and after performance.',
            repetition='Three independent processes0,1,2; each six rounds hot then stream. Arm order rotates by round+process. Complete passes until0.35s; require>=0.25s; finite1000000-pass limit. No retries or duration changes after observation.',
            overhead='Per-pass steady-clock cost below0.1% of fastest warmed six-expert pass. Reference checks and model loading outside timing; all candidate activation setup inside.',
            performance_gate='For a candidate: all six process/access paired-median ratios<=1.0, and their geometric mean<=0.97. Ratios use wall per expert and within-round matching. Preserve all samples/spread. No isolated-dot speed substitute.',
            stop='First ownership/build/source/byte/numerical/timing failure closes the stage; no retry. Negative performance rejects this candidate, with no model run.',
            limits='Component only, no GPU execution, model throughput, hardware saturation, task quality or physical262144/RESTORE qualification. A winning method must next pass five-worker actual-batch production GU/SwiGLU and full-pool gates before any engine/model trial.'),
        previous_comparison=dict(commit=previous_commit,path=str(RESULT.relative_to(P)),decision='Subset methods rejected at4.66153x/5.24602x; raw/build retired.'),
        retention='Keep compact individual samples and current decision. Auxiliary CPU binary/build only for this exact comparison; retire closed raw/build after source/recipe recovery and result commit. Current known-good engine/ten fixtures stay untouched.')
    dump(RESULT, dict(schema='strata-cpu-iq2-v1', plan=plan, stage='prepared', passed=False, complete=False,
        active=False, GPU_executed=False, model_adoption_qualified=False))
    print('Prepared direct gate/up IQ2_S NT1 plan; commit before build.')


def load():
    base.need(RESULT.read_bytes() == base.git(P, 'show', 'HEAD:'+str(RESULT.relative_to(P))), 'commit current stage first')
    record = json.loads(RESULT.read_text()); plan = record['plan']
    base.need(base.source_pin(P, plan['source']['revision'], PREFIXES) == plan['source'], 'component source drift')
    base.need(base.source_pin(base.GGML, base.GGML_REV, ('ggml',)) == plan['ggml'], 'ggml drift')
    base.need(base.machine() == plan['machine'], 'machine drift')
    base.need(all(base.digest(p) == h for p,h in plan['tools'].items()), 'tool drift')
    base.need(base.digest(base.METADATA) == plan['metadata_sha256'] and
        base.identity(plan['format']['shard']) == plan['model_identity'] and
        gu_hash(plan['format']) == plan['format']['GU_sha256'], 'GU fixture drift')
    return record


def reduce_samples(samples):
    expected = {(p,r,m,a) for p in range(3) for r in range(6) for m in range(3) for a in range(2)}
    base.need(len(samples) == 108 and {(s['process'],s['round'],s['method'],s['mode']) for s in samples} == expected,
        'complete unique108-sample matrix')
    base.need(all(s['wall_s'] >= .25 and s['passes'] > 0 and s['experts'] == (96 if s['mode'] else 6) for s in samples), 'sample floor/count')
    lookup = {(s['process'],s['round'],s['method'],s['mode']):s['wall_s']/(s['passes']*s['experts']) for s in samples}
    decisions = []
    for method in (1,2):
        cells = []
        for process in range(3):
            for mode in range(2):
                ratios = [lookup[process,r,method,mode]/lookup[process,r,0,mode] for r in range(6)]
                cells.append(dict(process=process,mode=mode,paired_ratios=ratios,median=statistics.median(ratios)))
        gm = statistics.geometric_mean(c['median'] for c in cells)
        decisions.append(dict(method=method,cells=cells,geomean=gm,
            performance_gate_passed=gm <= .97 and all(c['median'] <= 1 for c in cells)))
    return decisions


def execute(stage):
    record = load();plan = record['plan'];previous = {'build':'prepared','check':'built','measure':'checked'}[stage]
    base.need(record['stage'] == previous and not record['active'], 'wrong stage; no retries')
    ROOT.mkdir(exist_ok=True);path = ROOT/(stage+'.json');base.need(not path.exists(), 'stage already attempted')
    state = dict(stage=stage,active=False,passed=False,complete=False,commands=[])
    def persist():
        state['active'] = own.active is not None
        dump(path, state)
    own = owner.Owner(ROOT/'owned', persist);state['commands'] = own.commands
    def run(label, argv, wall, cpu=600):
        entry,out,err = own.run(stage+'-'+label,argv,plan['environment'],P,wall=wall,cpu=cpu,
            rss_cap=3<<30,text_cap=64<<20,file_cap=64<<20,total_cap=1<<30)
        base.need(owner.closed(entry) and entry['exit_code'] == 0, 'closed successful child required: '+label)
        return out,err
    try:
        if stage == 'build':
            run('configure', plan['configure'],120);run('compile',plan['build'],600,3600)
            binary = ROOT/'build/cpu_iq2_pair';record['binary'] = dict(path=str(binary),sha256=base.digest(binary),bytes=binary.stat().st_size)
            links,_ = run('ldd',['/usr/bin/ldd',str(binary)],20)
            base.need(not any(x in links.read_text().lower() for x in ('sycl','level_zero','cuda','amdhip','libze')), 'GPU runtime linked')
        else:
            base.need(base.digest(record['binary']['path']) == record['binary']['sha256'], 'binary drift')
            row = plan['format'];all_samples=[];proof=[]
            for process in range(3 if stage == 'measure' else 1):
                argv = [record['binary']['path'],'perf' if stage == 'measure' else 'check',row['shard'],str(row['gate']),str(row['up']),str(row['gu_bytes']),str(process),str(plan['machine']['physical_cpus'][0])]
                out,_ = run(str(process),argv,120,120);rows = [json.loads(s) for s in out.read_text().splitlines()]
                base.need(rows[-1] == dict(kind='closed',passed=True), 'native completion marker')
                validation = [v for v in rows if v['kind'] == 'real_validation']
                base.need(len(validation) == 1 and validation[0]['experts'] == 96 and validation[0]['bitwise'], 'real numerical gate')
                if stage == 'check':
                    contracts = [v for v in rows if v['kind'] == 'contract']
                    base.need(len(contracts) == 1 and contracts[0]['passed'] and contracts[0]['integer_cases'] == 1048576 and contracts[0]['row_dots'] == 1280 and contracts[0]['valid_extreme_Q8_checked'], 'exhaustive numerical gate')
                    proof += contracts
                else:
                    clocks = [v for v in rows if v['kind'] == 'clock'];base.need(len(clocks) == 1 and clocks[0]['per_call_s'] < clocks[0]['fastest_warm_pass_s']*.001, 'clock gate')
                    proof.append(dict(process=process,clock=clocks[0],numerical=validation[0]))
                    all_samples += [v for v in rows if v['kind'] == 'sample']
            record[stage+'_evidence'] = proof
            if stage == 'measure':
                record['decisions'] = reduce_samples(all_samples)
                record['sample_columns'] = ['process','round','method','mode','passes','experts','wall_s','cpu_s']
                record['samples'] = [[s[k] for k in record['sample_columns']] for s in all_samples]
                record['performance_gate_passed'] = any(c['performance_gate_passed'] for c in record['decisions'])
        # Recheck exact inputs/fixture after execution; the committed preceding state remains original.
        load();state.update(passed=True,complete=True)
    except BaseException as error:
        state['error'] = str(error)
    finally:
        state.update(active=own.active is not None,ownership_closed=own.active is None and all(owner.closed(c) for c in own.commands));persist()
        record.update(active=state['active'],ownership_closed=state['ownership_closed'],passed=state['passed'],complete=state['complete'],stage={'build':'built','check':'checked','measure':'closed'}[stage] if state['passed'] else 'failed')
        if 'error' in state:record['error']=state['error']
        dump(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','performance_gate_passed','error')}))
    return 0 if state['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','build','check','measure'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal():continue
            try:exe=(proc/'exe').resolve(strict=True)
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            base.need(exe.name not in ('cpu_iq2_pair','cpu_iq2_subset','cpu_roofline'), 'existing CPU component owner '+proc.name)
        return prepare() if args.stage == 'prepare' else execute(args.stage)


if __name__ == '__main__':sys.exit(main())
