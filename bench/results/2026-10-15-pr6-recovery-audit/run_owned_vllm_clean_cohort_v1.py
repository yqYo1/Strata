#!/usr/bin/env python3
"""Continue the declared clean cohort serially after Main's first FP32 run."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first-fp32-sha256',required=True)
    opts=ap.parse_args()
    package=A/'own-vllm-xe2-qualification-v1'
    meta_path=package/'manifest.json'
    if sha(meta_path)!='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58':
        raise RuntimeError('reviewed manifest drift')
    meta=json.loads(meta_path.read_text())
    helper=package/'main_controller.py';owner_path=Path(meta['direct_owner']['path'])
    if sha(helper)!=meta['package_files']['main_controller.py']['sha256'] or sha(owner_path)!=meta['direct_owner']['sha256']:
        raise RuntimeError('reviewed helpers drift')
    h=imported('cohort_helpers',helper);o=imported('cohort_owner',owner_path)
    recipe=A/'run_owned_vllm_long_v1.py';recipe_sha=sha(recipe)
    result=A/'own-vllm-clean-cohort-v1.json'
    h.require(not result.exists(),'fresh aggregate only; no automatic retries')
    prior=A/'own-vllm-clean-fp32-r0-v1.json';prior_sha=opts.first_fp32_sha256
    first,gate=h.receipt_gate(o,prior,prior_sha,h.boot(),qualification=True)
    h.require(first['schema']=='strata-vllm-long-v1' and first['phase']=='clean' and
              first['arm']=='fp32' and first['repetition']==0,'first clean FP32 exact receipt')
    order=[('bf16',0),('qsa',0),('qsa',1),('bf16',1),('fp32',1),('fp32',2),('bf16',2),('qsa',2)]
    record=dict(schema='strata-vllm-clean-cohort-v1',passed=False,complete=False,
        boot_id=h.boot(),GPU_executed_by_this_wrapper=False,active_controller=None,
        controller_sha256=sha(__file__),child_controller_sha256=recipe_sha,first_gate=gate,
        continuation_order=order,completed=[],
        supervision='Main serial wrapper only. Each child owns the exclusive measurement lock, all native PID/start/deadlines/RSS/log caps, normal reap and fresh fault checks. No parallel test/build/cleanup; no retry or failed-gate substitution.',
        adoption=False,quality_accepted=False,physical_262144_qualified=False)
    def save():result.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for arm,repetition in order:
            h.require(sha(recipe)==recipe_sha and sha(__file__)==record['controller_sha256'] and h.boot()==record['boot_id'],'cohort source/boot drift')
            h.receipt_gate(o,prior,prior_sha,record['boot_id'],qualification=True)
            target=A/f'own-vllm-clean-{arm}-r{repetition}-v1.json'
            h.require(not target.exists(),'fresh independent child only')
            argv=['/usr/bin/python3','-B',str(recipe),'clean',arm,str(repetition),'--prior',str(prior),'--prior-sha256',prior_sha]
            print(json.dumps(dict(start=arm,repetition=repetition,prior_sha256=prior_sha)),flush=True)
            child=subprocess.Popen(argv,cwd=P)
            record['active_controller']=dict(argv=argv,identity=o.identity(child.pid))
            save()
            # The pinned child performs the finite hardware supervision. Keep
            # this coordinator alive until that controller is actually reaped.
            code=child.wait()
            record['active_controller']=None
            row=dict(arm=arm,repetition=repetition,controller_exit_code=code,path=str(target))
            record['completed'].append(row);save()
            h.require(code==0 and target.is_file(),'child failed; cohort stops without retry')
            next_sha=sha(target)
            value,proof=h.receipt_gate(o,target,next_sha,record['boot_id'],qualification=True)
            h.require(value['schema']=='strata-vllm-long-v1' and value['phase']=='clean' and
                value['arm']==arm and value['repetition']==repetition and
                value['binary_sha256']==first['binary_sha256'],'matching independent native job')
            row.update(sha256=next_sha,gate=proof,
                prompt_tps=value['benchmark_result']['results']['prefill_only']['prompt_tps'])
            prior,prior_sha=target,next_sha;save()
        record.update(passed=True,complete=True)
    except BaseException as exc:
        record['error']=repr(exc)
    finally:
        record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps(dict(passed=record['passed'],complete=record['complete'],receipt=str(result),sha256=sha(result))),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
