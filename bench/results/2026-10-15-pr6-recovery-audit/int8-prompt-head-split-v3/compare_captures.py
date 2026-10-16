#!/usr/bin/env python3
"""Offline all-output comparator proposal. No SYCL runtime or GPU invocation."""
import sys
sys.dont_write_bytecode = True
import argparse
import math
import os
from pathlib import Path
import struct
from capture_common import FULL_CASES, MAXCOUNT_CASES, HEX, open_dir, read_bytes, read_json, require, sha, write_json

TOL = 1e-5
IDENTITY = ('source_sha256', 'binary_sha256', 'fixture_sha256', 'device', 'driver')
INPUT_KEYS = {'q','table','ids','steps','k16','v16','k8','v8','ks','vs','k4','v4'}

class Capture:
    def __init__(self, path, case_set, expected_route=None):
        self.path = Path(path)
        self.fd = open_dir(self.path, private=True)
        parent = open_dir(self.path.parent, private=True)
        try:
            job = read_json(parent, 'process.json')
            before = read_json(parent, 'fault-before.json')
            after = read_json(parent, 'fault-after.json')
        finally:
            os.close(parent)
        require(job.get('schema') == 'head-split-process-v2', 'missing process provenance')
        require(job.get('status') == 'CLOSED_EXIT0' and job.get('returncode') == 0 and
                job.get('owned_pid_retired') is True and job.get('process_group_absent') is True,
                'abnormal/unretired/incomplete process cannot be compared')
        require(job.get('capture_dir') == str(self.path), 'process/capture path mismatch')
        require(before.get('schema')=='head-split-fault-before-v2' and
                after.get('schema')=='head-split-fault-after-v2' and before.get('boot_id')==job.get('boot_id') and
                after.get('boot_id')==job.get('boot_id') and after.get('before_cursor')==before.get('cursor') and
                bool(before.get('cursor')) and bool(after.get('after_cursor')) and bool(after.get('filter_id')) and
                after.get('status')=='PASS' and after.get('new_faults')==0 and
                after.get('error_scan_success') is True, 'fresh Main fault gate is missing/failed')
        self.job = job
        self.start = read_json(self.fd, 'start.json')
        self.done = read_json(self.fd, 'complete.json')
        require(self.start.get('schema') == 'head-split-start-v2' and
                self.done.get('schema') == 'head-split-complete-v2', 'capture schema mismatch')
        require(self.start.get('case_set') == case_set, 'case-set mismatch')
        require(type(self.start.get('format')) is int and self.start['format'] in (0,1,2,3) and
                (case_set not in ('full','maxcount') or self.start['format']==1), 'format outside fixed case plan')
        require(self.done.get('status') == 'CLOSED_PASS' and self.done.get('all_admitted') is True and
                self.done.get('fallback_calls') == 0, 'incomplete/declined capture')
        require(self.start.get('source_sha256') == job['source_sha256'] and
                self.start.get('binary_sha256') == job['binary_sha256'] and
                self.start.get('fixture_sha256') == job['fixture_sha256'], 'source/binary receipt mismatch')
        for key in ('source_sha256', 'binary_sha256', 'fixture_sha256'):
            require(HEX.fullmatch(self.start.get(key,'')), 'invalid SHA256 provenance')
        require(self.start.get('environment') == job.get('route_environment'), 'exec environment receipt mismatch')
        require(self.start.get('route') == job.get('route'), 'route receipt mismatch')
        if expected_route is not None:
            require(self.start.get('route') == expected_route, 'wrong route intent')
        self.cases = self.done.get('cases')
        expected = FULL_CASES if case_set == 'full' else MAXCOUNT_CASES if case_set=='maxcount' else ['isolation_format' + str(self.start['format'])]
        require(isinstance(self.cases,list) and self.cases == expected, 'missing/reordered/duplicate cases')
        env = self.start['environment']
        require(env.get('STRATA_SYCL_PROMPT_NATIVE_KV_MIN') in ('@unset','0'), 'Native KV option not fixed off')
        if case_set in ('full','maxcount'):
            require(env.get('STRATA_SYCL_PROMPT_DIRECT') == '1' and
                    env.get('STRATA_SYCL_PROMPT_HEAD_SPLIT') == ('@unset' if expected_route == 'legacy' else expected_route),
                    'full route environment mismatch')

    def close(self):
        os.close(self.fd)

    def case(self, name):
        m = read_json(self.fd, name + '.json')
        require(m.get('schema') == 'head-split-case-v2' and m.get('case_id') == name, 'case identity mismatch')
        require(m.get('status') == 'PASS' and m.get('admitted') is True and m.get('fallback_calls') == 0 and
                m.get('input_immutable') is True and m.get('output_guards') is True and
                m.get('oracle_pass') is True and m.get('oracle_abs_rel') == TOL, 'failed/declined case')
        require(type(m.get('nq')) is int and m['nq'] in (1,3), 'invalid query count')
        require(m.get('format') == self.start['format'], 'format/capture mismatch')
        if self.start['case_set'] == 'isolation':
            spec = {'nq':1,'cap':65,'width':65,'step_width0':65,'physical_pages':5}
        elif name in MAXCOUNT_CASES:
            width=int(name.split('_')[1])
            spec={'nq':1,'cap':width,'width':width,'step_width0':width,'physical_pages':5}
        elif '_tail_' in name:
            width = int(name.rsplit('_',1)[1]); dense = name.startswith('dense_')
            spec = {'nq':1 if dense else 3,'cap':max(1,width),'width':width,'step_width0':width,
                    'physical_pages':((max(1,width)+3)//4+1) if dense else 5}
        else:
            sparse = name in ('all_missing','cap262144_logical262143','negative_width','width_gt_cap')
            spec = {'nq':1 if name=='all_missing' else 3,
                    'cap':262144 if name=='cap262144_logical262143' else 65,
                    'width':65,'step_width0':-1 if name=='negative_width' else 66 if name=='width_gt_cap' else 65,
                    'physical_pages':5 if sparse else 18}
        require(all(m.get(k)==v for k,v in spec.items()), 'case shape differs from fixed v3 plan')
        n = m['nq'] * 24 * 256
        require(m.get('reference_count') == n and m.get('output_count') == n + 32 and
                m.get('guard_count') == 16, 'all-output count/guard mismatch')
        require(m.get('output_file') == name + '.out.f32' and m.get('reference_file') == name + '.ref.f64', 'case filename mismatch')
        require(set(m.get('inputs',{})) == INPUT_KEYS, 'input fingerprint set incomplete')
        for v in m['inputs'].values():
            require(type(v.get('count')) is int and v['count'] >= 0 and HEX.fullmatch(v.get('sha256','')),
                    'invalid input fingerprint')
        rows=m['physical_pages']*2*4; fmt=m['format']
        counts={'q':('f32',n),'table':('i32',65536),'ids':('i32',m['nq']*m['cap']),
                'steps':('i32',m['nq']*4),'k16':('u16',rows*256 if fmt==0 else 0),
                'v16':('u16',rows*256 if fmt==0 else 0),'k8':('i8',rows*256 if fmt in (1,3) else 0),
                'v8':('i8',rows*256 if fmt==1 else 0),'ks':('u16',rows*4 if fmt in (1,3) else 0),
                'vs':('u16',rows*4 if fmt==1 else 0),'k4':('u8',rows*144 if fmt==2 else 0),
                'v4':('u8',rows*144 if fmt in (2,3) else 0)}
        for k,(dtype,count) in counts.items():
            require(m['inputs'][k]['dtype']==dtype and m['inputs'][k]['count']==count, f'input extent/type mismatch: {k}')
        out = read_bytes(self.fd, m['output_file'], (n + 32)*4)
        ref = read_bytes(self.fd, m['reference_file'], n*8)
        require(len(out) == (n+32)*4 and len(ref) == n*8, 'truncated output/reference')
        require(sha(out) == m['output_sha256'] and sha(ref) == m['reference_sha256'], 'capture SHA256 mismatch')
        guard = struct.pack('<f',91.) * 16
        require(out[:64] == guard and out[-64:] == guard, 'output guards differ')
        values = struct.unpack('<' + str(n) + 'f', out[64:-64])
        reference = struct.unpack('<' + str(n) + 'd', ref)
        maximum = 0.
        for i,(x,r) in enumerate(zip(values,reference)):
            require(math.isfinite(x) and math.isfinite(r), f'nonfinite output/reference at {name}:{i}')
            error = abs(x-r)
            maximum = max(maximum,error)
            require(error <= TOL*(1.+abs(r)), f'FP64 oracle failure at {name}:{i}')
        return m,out,ref,values,maximum

def compare(left, right, bitwise, report):
    for k in IDENTITY:
        require(left.start[k] == right.start[k], f'provenance/device mismatch: {k}')
    require(left.cases == right.cases, 'different case lists')
    for key in ('STRATA_SYCL_PROMPT_DIRECT','STRATA_SYCL_PROMPT_NATIVE_KV_MIN','STRATA_PROMPT_ATTN_Q4'):
        require(left.start['environment'][key] == right.start['environment'][key], 'independent exec environment differs')
    for name in left.cases:
        lm,lo,lr,lv,le = left.case(name)
        rm,ro,rr,rv,re = right.case(name)
        for k in ('format','nq','cap','width','step_width0','physical_pages','inputs'):
            require(lm[k] == rm[k], f'input/shape mismatch {name}:{k}')
        require(lr == rr, f'FP64 references differ for identical input {name}')
        same = lo == ro  # Includes prefix/suffix guards and every payload bit.
        maximum = 0.; mismatch = 0; first = []
        for i,(a,b) in enumerate(zip(lv,rv)):
            delta = abs(a-b); maximum = max(maximum,delta)
            if delta > TOL*(1.+abs(a)):
                mismatch += 1
                if len(first) < 8:
                    first.append(i)
        passed = same if bitwise else mismatch == 0
        report['cases'].append({'case_id':name,'left_route':left.start['route'],'right_route':right.start['route'],
                                'output_count':len(lv),'bitwise':same,'bitwise_required':bitwise,
                                'max_abs':maximum,'tolerance_failures':mismatch,'first_failure_indices':first,
                                'left_oracle_max_abs':le,'right_oracle_max_abs':re,'pass':passed})
        require(passed, f'pair gate failed: {name}; bitwise_required={bitwise}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('full','ch32','maxcount','isolation'),required=True)
    p.add_argument('--legacy'); p.add_argument('--ch32'); p.add_argument('--ch64')
    p.add_argument('--left'); p.add_argument('--right'); p.add_argument('--output',required=True)
    a=p.parse_args()
    report={'schema':'head-split-comparison-v2','status':'FAIL','cases':[],
            'fault_gate':'EXTERNAL_MAIN_REQUIRED','route_observation':'EXTERNAL_MAIN_REQUIRED',
            'full_physical_256k_or_restore_proved':False}
    opened=[]
    try:
        if a.mode in ('full','ch32','maxcount'):
            require(a.legacy and a.ch32 and not a.left and not a.right and
                    (bool(a.ch64) if a.mode in ('full','maxcount') else not a.ch64), 'full/ch32 argument mismatch')
            case_set='maxcount' if a.mode=='maxcount' else 'full'
            base=Capture(a.legacy,case_set,'legacy'); opened.append(base)
            c32=Capture(a.ch32,case_set,'32'); opened.append(c32)
            compare(base,c32,True,report)
            if a.mode in ('full','maxcount'):
                c64=Capture(a.ch64,case_set,'64'); opened.append(c64)
                compare(base,c64,False,report)
        else:
            require(a.left and a.right and not a.legacy and not a.ch32 and not a.ch64,'isolation requires exactly left/right')
            left=Capture(a.left,'isolation','legacy'); opened.append(left)
            right=Capture(a.right,'isolation','legacy'); opened.append(right)
            require(left.start['environment']['STRATA_SYCL_PROMPT_DIRECT'] ==
                    right.start['environment']['STRATA_SYCL_PROMPT_DIRECT'],'isolation Direct setting differs')
            require(left.start['environment']['STRATA_SYCL_PROMPT_HEAD_SPLIT'] == '@unset','isolation baseline must have split unset')
            for key in ('STRATA_SYCL_PROMPT_NATIVE_KV_MIN','STRATA_PROMPT_ATTN_Q4'):
                require(left.start['environment'][key] == right.start['environment'][key], 'independent option differs')
            compare(left,right,True,report)
        report['status']='OUTPUT_COMPARE_PASS'
        report['fault_gate']='PASS_MAIN_HELPER'
        report['capture_dirs']=[str(c.path) for c in opened]
        report['source_sha256']=opened[0].start['source_sha256']
        report['binary_sha256']=opened[0].start['binary_sha256']
    except Exception as e:
        report['error']=str(e)
    finally:
        for c in opened:
            c.close()
    write_json(a.output,report)  # Exclusive new file; never overwrite a receipt.
    print(report['status'])
    return 0 if report['status']=='OUTPUT_COMPARE_PASS' else 1

if __name__=='__main__':
    sys.exit(main())
