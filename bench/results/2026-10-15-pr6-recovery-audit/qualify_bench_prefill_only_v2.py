#!/usr/bin/env python3
"""Close the actual evidence-fixture regression; preserve original failed receipt."""
import fcntl
import json
from pathlib import Path
import re
import sys

A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_path=A/'bench-prefill-only-host-v1.json';old=json.loads(old_path.read_text())
        require(not old['active'] and not old['passed'] and old['source_applied'] and
                all(module.closed(c) for c in old['commands']),'original failure not closed')
        require(old['tests']==[dict(label='proposed-prefill-only',tests=5),dict(label='benchmark-regression',tests=5),
                               dict(label='benchmark-failfast',tests=7)],'prior successful host evidence changed')
        require(old['commands'][-1]['label']=='benchmark-evidence' and old['commands'][-1]['exit_code']==1,
                'unexpected original failure')
        require(all(sha(P/n)==v['proposed_sha256'] for n,v in old['source_pins'].items()),'tested client changed')
        root=B/'main-bench-prefill-only-host-root-v2';receipt=A/'bench-prefill-only-host-v2.json'
        require(not root.exists() and not receipt.exists(),'fresh continuation root')
        root.mkdir(mode=0o700)
        record=dict(schema='main-bench-prefill-only-host-v2',active=True,passed=False,complete=False,GPU_executed=False,
                    commands=[],original_failed_receipt_sha256=sha(old_path),reused_passed_tests=old['tests'],
                    recipe_sha256=sha(Path(__file__)),source_pins=old['source_pins'],
                    evidence_fixture_sha256=sha(P/'tools/test_benchmark_evidence.py'),
                    reason='Old comparison fixture omitted actual nonce metadata required since fixed-workload patch; supply valid identity and also assert missing/different identities suppress comparison.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands
        env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        try:
            c,out,err=owner.run('benchmark-evidence',['/usr/bin/python3','-B','tools/test_benchmark_evidence.py'],env,P,
                                wall=90,cpu=90,rss_cap=2<<30,text_cap=8<<20,total_cap=64<<20,file_cap=16<<20)
            require(module.closed(c) and c['exit_code']==0,'evidence tests failed: '+err.read_text()[-4000:])
            count=re.search(r'Ran (\d+) tests?',err.read_text())
            require(count is not None and int(count[1])==14,'actual test count')
            require(sha(old_path)==record['original_failed_receipt_sha256'],'original result changed')
            record.update(passed=True,complete=True,new_tests=14,total_host_tests=31,
                          actual_client_sha256=sha(P/'tools/strata_bench.py'),actual_wrapper_sha256=sha(P/'bench.sh'))
        except BaseException as error:record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc());persist()
            with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],total_host_tests=31,GPU_executed=False)),flush=True)


if __name__=='__main__':main()
