#!/usr/bin/env python3
"""Apply reviewed official-client prefill scope only after owned native closure."""
import fcntl
import json
from pathlib import Path
import re
import sys

A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    proposal=A/'bench-prefill-only-v1'
    prior_path=A/'selective-iq-clean-bench-full-1-v2.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=json.loads(prior_path.read_text())
        require(not prior['active'] and all(module.closed(c) for c in prior['commands']), 'old owner not retired')
        require(prior['result']['bench_exit_code']==-2 and prior['result']['server_exit_code']==0 and
                prior['native_exit']['exit_code']==0 and not prior['fresh_faults'] and prior['boot_unchanged'] and
                not prior['driver_dump_exists'],'expected deliberate benchmark interrupt / normal native closure')
        for c in prior['commands']:
            for old in c.get('owners',[]):
                now=module.identity(old['pid'])
                require(now is None or now['start_ticks']!=old['start_ticks'],'old identity still live')
        pins=json.loads((proposal/'pins.json').read_text())
        manifest=json.loads((proposal/'artifact-sha256.json').read_text())
        for n,v in manifest['artifacts'].items():
            require((proposal/n).stat().st_size==v['bytes'] and sha(proposal/n)==v['sha256'],'proposal artifact changed')
        for n,h in manifest['inputs'].items(): require(sha(P/n)==h,'proposal input changed: '+n)
        for n,v in pins.items():require(sha(P/n)==v['base_sha256'],'base changed: '+n)
        root=B/'main-bench-prefill-only-host-root-v1'
        receipt=A/'bench-prefill-only-host-v1.json'
        require(not root.exists() and not receipt.exists(),'fresh host root')
        root.mkdir(mode=0o700)
        record=dict(schema='main-bench-prefill-only-host-v1',active=True,passed=False,complete=False,
                    GPU_executed=False,commands=[],tests=[],source_applied=False,
                    recipe_sha256=sha(Path(__file__)),proposal_manifest_sha256=sha(proposal/'artifact-sha256.json'),
                    source_pins=pins,prior_interrupted_receipt_sha256=sha(prior_path),
                    scope='Host fake-HTTP, body/coverage/journal/CLI tests plus actual client patch. No native inference/speed/quality gate.')
        def persist(): (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands
        env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),LANG='C.UTF-8',LC_ALL='C.UTF-8',
                 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        def run(label,argv,test=False):
            c,out,err=owner.run(label,argv,env,P,wall=90,cpu=90,rss_cap=2<<30,
                                text_cap=8<<20,total_cap=64<<20,file_cap=16<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-4000:])
            if test:
                match=re.search(r'Ran (\d+) tests?',err.read_text())
                require(match is not None,label+' missing test count')
                record['tests'].append(dict(label=label,tests=int(match[1])))
        try:
            run('proposed-prefill-only',['/usr/bin/python3','-B',str(proposal/'test_prefill_only.py')],True)
            run('apply-reviewed-client',['git','apply',str(proposal/'source.diff')])
            record['source_applied']=True
            require(all(sha(P/n)==v['proposed_sha256'] for n,v in pins.items()),'actual output hash mismatch')
            for label,filename in [('benchmark-regression','tools/test_strata_bench.py'),
                                   ('benchmark-failfast','tools/test_strata_bench_failfast.py'),
                                   ('benchmark-evidence','tools/test_benchmark_evidence.py')]:
                run(label,['/usr/bin/python3','-B',filename],True)
            record.update(passed=True,complete=True,actual_client_sha256=sha(P/'tools/strata_bench.py'),
                          actual_wrapper_sha256=sha(P/'bench.sh'))
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            persist()
            with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],source_applied=record['source_applied'],tests=record['tests'])),flush=True)


if __name__=='__main__':main()
