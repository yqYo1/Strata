#!/usr/bin/env python3
"""Revalidate closed native evidence with the actual batched-prefix contract."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
BASE=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(HERE/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha
from validate_layermajor_census_v1 import validate_census


def main():
    module=load_owner(PROJECT/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_path=HERE/'selective-iq-model-v1.json'
        old=json.loads(old_path.read_text())
        require(not old['active'] and not old['passed'] and not old['complete'],'original failure preserved')
        require(old['error']=="RuntimeError('layer-major census: request geometry')",'unexpected original failure')
        require(all(module.closed(c) and c['exit_code']==0 for c in old['commands']),'native/ownership not normally closed')
        require(not old['fresh_faults'] and old['boot_unchanged'] and not old['driver_dump_exists'],'native fault gate')
        require(len(old['requests'])==1 and old['requests'][0]['complete'],'one complete model request')
        request=old['requests'][0]
        # generate.cpp:6370 uses n_prompt-1; the final prompt position is handled
        # by the token loop. PP protocol independently reports the same prefix.
        require(request['done']['prompt']==request['done']['read']==32089,'actual full prompt')
        pp=[line.split() for line in request['progress'] if line.startswith('PP ')]
        require(pp and int(pp[-1][1])==32088 and int(pp[-1][2])==32089,'independent PP prefix boundary')
        reference=json.loads((HERE/'owned-prefill-stall-model-v1.json').read_text())
        require(reference['passed'] and reference['complete'] and not reference['active'],'reference gate')
        require(request['generated_ids']==reference['requests'][0]['generated_ids'],'same-input generated IDs differ')
        log=BASE/'main-selective-iq-model-root-v1/native-owned/model.stderr'
        validation=validate_census(stderr_path=log,expected_tokens=32088)
        result=dict(schema='main-selective-iq-model-assessment-v2',passed=True,complete=True,active=False,
                    GPU_executed=False,native_GPU_run='selective-iq-model-v1.json; reused without replay',
                    original_receipt_sha256=sha(old_path),original_status_preserved=dict(passed=False,complete=False),
                    original_collector_error='Passed full prompt length to batched-prefill census; the native uses n_prompt-1.',
                    source_contract=dict(path='src/program/generate.cpp',sha256=sha(PROJECT/'src/program/generate.cpp'),line=6370),
                    recipe_sha256=sha(Path(__file__)),validator_sha256=sha(HERE/'validate_layermajor_census_v1.py'),
                    binary_sha256=old['binary_sha256'],request=request,census_validation=validation,
                    reference_receipt_sha256=sha(HERE/'owned-prefill-stall-model-v1.json'),
                    generated_ID_equality=True,fresh_faults=old['fresh_faults'],boot_unchanged=True,driver_dump_exists=False,
                    stderr_sha256=sha(log),performance_qualified=False,quality_qualified=False,
                    physical_262144_qualified=False,restore_qualified=False,
                    scope='Closed normal native, same24 IDs and independently validated actual48-layer census. No clean speed/task-quality/full-context claim.')
        output=HERE/'selective-iq-model-assessment-v2.json'
        require(not output.exists(),'do not overwrite assessment')
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,prefill_tokens=32088,request_prompt_tokens=32089,
                              by_role_format={k:{n:v for n,v in t.items() if n!='histogram'} for k,t in validation['by_role_format'].items()})),flush=True)


if __name__=='__main__':
    main()
