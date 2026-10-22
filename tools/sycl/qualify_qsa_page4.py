#!/usr/bin/env python3
"""Qualify frozen compiled artifacts separately from the failed broad IR gate."""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
import build_qsa_page4 as build

KEY='compiled_artifact_qualification'


def divisions(path):
    text=Path(path).read_text()
    rows=[]
    for line in text.splitlines():
        match=re.match(r'\s*%\S+ = (sdiv|udiv) i64 ([^,]+), ([^,\s]+)',line)
        if match:
            op,lhs,rhs=match.groups()
            rows.append(dict(operation=op,dividend=lhs,divisor=rhs,variable=rhs.startswith('%')))
    return dict(kernels=sum(line.startswith('define ') and 'spir_kernel' in line for line in text.splitlines()),
        rows=rows,variable_divisors=sum(row['variable'] for row in rows),
        constants=[(row['operation'],row['divisor']) for row in rows if not row['variable']])


def proof(record):
    build.admission(record,True)
    build.guards.require(record['stage']=='failed' and record['passed'] is False and record['complete'] is False
        and record['ownership_closed'] and not record['active'] and not record['GPU_executed']
        and record['error']=="RuntimeError('planned constant-divisor frontend transformation not established')",
        'preserve exact original pre-GPU failed gate')
    original=json.loads((build.ROOT/'build-record.json').read_text())
    for key in ('source_commit','input_sha256','commands','build_environment','generated','compiler','compiler_inputs','binaries','device_IR'):
        build.guards.require(original[key]==record[key],'original compiled record changed: '+key)
    commands=original['owned_commands']
    build.guards.require(len(commands)==len(record['commands'])==13 and original['ownership_closed'] and not original['active']
        and all(build.supervision.closed(command) and command['exit_code']==0 for command in commands),
        'all original compiler/link/IR commands must have normally succeeded')
    result={}
    for name,item in record['device_IR'].items():
        build.guards.require(build.digest(item['path'])==item['sha256'],'frozen device IR changed')
        result[name]=divisions(item['path'])
    build.guards.require(result['control']['kernels']==result['page4']['kernels']==2
        and result['control']['variable_divisors']==3 and result['page4']['variable_divisors']==0
        and result['control']['constants']==result['page4']['constants']==[('sdiv','2')],
        'exact page-related variable-divisor removal and unchanged constant division required')
    return result


def admission(record):
    item=record[KEY]
    raw=build.gen.git('show',item['source_commit']+':tools/sycl/qualify_qsa_page4.py')
    build.guards.require(raw==Path(__file__).read_bytes() and build.hashlib.sha256(raw).hexdigest()==item['controller_sha256'],
        'closed-data qualifier source changed')
    build.guards.require(item['passed'] and item['complete'] and item['original_failure_preserved']
        and proof(record)==item['IR_analysis'],'frozen compiled-artifact qualification changed')


def main():
    record=build.gen.committed(build.RESULT)
    build.guards.require(KEY not in record,'one separately scoped frozen-data reanalysis;no blind retry')
    revision=build.gen.git('rev-parse','HEAD').decode().strip()
    raw=build.gen.git('show',revision+':tools/sycl/qualify_qsa_page4.py')
    build.guards.require(raw==Path(__file__).read_bytes(),'commit exact qualifier before reanalysis')
    analyzed=proof(record)
    record[KEY]=dict(source_commit=revision,controller_sha256=build.hashlib.sha256(raw).hexdigest(),
        passed=True,complete=True,original_failure_preserved=True,IR_analysis=analyzed,
        scope='Frozen source/recipe/dependencies/finalbinary identities and all13normallyclosed compiler commands;three variable page-size i64divisors removed. Original whole-modulezero-divisor gate remainsfailed. Constant workgroupID/2 remainsinbotharms. No native rerun,recompilation,GPU validation,finalISA orspeed/model qualification.')
    admission(record);build.write(build.RESULT,record)
    print(json.dumps(record[KEY]),flush=True)


if __name__=='__main__':
    with (build.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([]);build.guards.driver_guard();main()
