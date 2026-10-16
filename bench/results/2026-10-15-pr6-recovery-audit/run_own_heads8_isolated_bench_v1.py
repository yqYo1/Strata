#!/usr/bin/env python3
"""Clean independent-process component timings after the full correctness gate."""
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')


def main():
    module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gate = json.loads((A/'own-heads8-isolated-correctness-v2.json').read_text())
        build = json.loads((A/'own-heads8-isolated-build-v1.json').read_text())
        require(gate['passed'] and gate['complete'] and not gate['active'] and gate['full_full_dump_byteexact'] and not gate['fresh_faults'] and gate['boot_unchanged'] and not gate['driver_dump_exists'],'full oracle/byteexact/healthy gate')
        for c in gate['commands']:
            require(module.closed(c) and c['exit_code'] == 0,'gate owners closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'],'gate owner alive')
        exe = Path(build['binary'])
        require(sha(exe) == gate['binary_sha256'],'same isolated fixture')
        require(all(sha(P/n) == h for n,h in build['canonical_source_pins'].items()) and sha(Path(build['canonical_native'])) == build['canonical_native_sha256'],'native cohort sources/binary unchanged')
        require(all(sha(Path(v['path'])) == v['sha256'] for v in gate['runtime_files']),'runtime pins')
        root = B/'main-own-heads8-isolated-bench-root-v1'
        target = A/'own-heads8-isolated-bench-v1.json'
        require(not root.exists() and not target.exists(),'fresh clean component cohort')
        root.mkdir(mode=0o700)
        env = {k:v for k,v in gate['environment'].items() if not k.startswith('UR_LOG_') and k != 'UR_ENABLE_LAYERS'}
        env['XDG_CACHE_HOME'] = str(root/'cache')
        order = [(nq,arm) for arm in (0,1,1,0,0,1) for nq in (128,256)]
        record = dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],cases=[],boot_id=boot_id(),
                      recipe_sha256=sha(Path(__file__)),correctness_receipt_sha256=sha(A/'own-heads8-isolated-correctness-v2.json'),
                      binary_sha256=sha(exe),canonical_native_sha256=build['canonical_native_sha256'],
                      canonical_source_pins=build['canonical_source_pins'],runtime_files=gate['runtime_files'],environment=env,order=order,
                      independent_processes_per_arm_per_nq=3,queued_calls_per_timing=10,raw_tensor_budget_bytes=64 << 20,
                      scope='Bounded component host-clock submit-through-drain aggregate, cap2051/allcounts, nq128/256; three independent processes per arm/shape. Four oracle queries plus full finite/guards/immutable inputs per process; all full outputs compared externally. Not pure kernel time, model PP, quality, physicalcontext or native adoption.',
                      model_performance_qualified=False,native_adopted=False)
        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(root/'owned',persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,argv):
            c,out,err = owner.run(label,argv,hostenv,P,wall=60,cpu=120,rss_cap=2 << 30,text_cap=8 << 20,total_cap=1 << 30,file_cap=8 << 20)
            require(module.closed(c) and c['exit_code'] == 0,label+': '+err.read_text()[-2000:])
            return out.read_text()
        cursor = None
        try:
            require((DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists(),'GPU preflight')
            cursor = re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            reference = {}
            for sequence,(nq,arm) in enumerate(order):
                label = f'bench-{sequence}-nq{nq}-arm{arm}'
                dump = root/(label+'.bin')
                current = dict(env,STRATA_SYCL_PROMPT_HEADS8=str(arm))
                args = ['taskset','-c','0',str(exe),'--bench',str(nq),'10','--dump',str(dump)]
                record['GPU_executed'] = True
                c,out,err = owner.run(label,args,current,P,wall=300,cpu=2400,rss_cap=8 << 30,text_cap=64 << 20,total_cap=1 << 30,file_cap=8 << 20)
                require(module.closed(c) and c['exit_code'] == 0,label+': '+err.read_text()[-6000:])
                stdout,stderr = out.read_text(),err.read_text()
                match = re.search(r'bounded_component nq=(\d+) cap=2051 counts=all2051 limited_oracle_queries=0,1,2,last repeats=(\d+) total_ms=([0-9.eE+-]+)',stdout)
                require(match and (int(match[1]),int(match[2])) == (nq,10),'exact bounded timing shape')
                ms = float(match[3])
                require(math.isfinite(ms) and ms > 0 and 'benchmark oracle=FP64_queries_0_1_2_last' in stdout and 'full_finite=PASS prefix_suffix_guards=PASS inputs=PASS cross_arm_exact=EXTERNAL_REQUIRED' in stdout,'positive aggregate and finite/oracle/guards')
                require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',stderr),'runtime/native errors')
                require(dump.stat().st_size == (nq*24*256+32)*4,'full output bytes')
                data = dump.read_bytes()
                if nq not in reference:
                    reference[nq] = data
                require(data == reference[nq],'all independent process full outputs byteexact')
                record['cases'].append(dict(sequence=sequence,nq=nq,arm=arm,total_ms=ms,queued_calls=10,ms_per_call=ms/10,
                                             stdout=stdout,stderr_sha256=sha(err),dump_sha256=sha(dump),dump_bytes=len(data),full_output_byteexact=True,
                                             process_pid=c['pid'],normal_closed=True))
                require(sum(v['dump_bytes'] for v in record['cases']) <= record['raw_tensor_budget_bytes'],'whole cohort raw budget')
                persist()
            medians = {}
            for nq in (128,256):
                values = {str(arm):[v['ms_per_call'] for v in record['cases'] if v['arm'] == arm and v['nq'] == nq] for arm in (0,1)}
                require(all(len(x) == 3 for x in values.values()),'three independent samples per arm')
                medians[str(nq)] = dict(individual_ms_per_call=values,baseline_median_ms=statistics.median(values['0']),
                                       heads8_median_ms=statistics.median(values['1']),ratio_baseline_over_heads8=statistics.median(values['0'])/statistics.median(values['1']))
            require(all(sha(P/n) == h for n,h in build['canonical_source_pins'].items()) and sha(Path(build['canonical_native'])) == build['canonical_native_sha256'],'NLL source/native still unchanged')
            require(all(sha(Path(n)) == h for n,h in build['library_pins'].items()),'canonical archives unchanged')
            record.update(passed=True,complete=True,summary=medians,all_process_outputs_byteexact=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    rows = [json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE','')))],boot_unchanged=boot_id() == record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record,f,indent=2)
                    f.write('\n')
        print(json.dumps(dict(passed=record['passed'],summary=record['summary'],all_outputs_byteexact=True,native_unchanged=True)),flush=True)


if __name__ == '__main__':
    main()
