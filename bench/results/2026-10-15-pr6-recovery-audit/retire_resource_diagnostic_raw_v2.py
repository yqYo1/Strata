#!/usr/bin/env python3
"""Retire closed duplicate diagnostics after exact structured evidence checks."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOTS={'own-metric-pool-diagnostic-v2.json':'main-metric-pool-diagnostic-v2',
       'own-metric-pool-diagnostic-trace-v3.json':'main-metric-pool-diagnostic-trace-v3',
       'own-host-transfer-link-recheck-v2.json':'main-host-transfer-link-recheck-v2',
       'own-host-transfer-link-control-v3.json':'main-host-transfer-link-control-v3'}
FAULT=r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))'


def pin(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    manifest=A/'resource-diagnostic-raw-retirement-v2.json'
    assert not manifest.exists()
    owner_source=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_source)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('resource_retirement_owner',owner_source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    records=[];pending=[]
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (A/'research-scope-review-round912-v1.json').exists()
        for name,rootname in ROOTS.items():
            p=A/name;r=json.loads(p.read_text())
            assert r['complete'] and r['ownership_closed'] and not r['active'] and not r['fresh_faults'] and not r['driver_dump_entries']
            for c in r['commands']:
                assert m.closed(c)
                for ident in c['owners']+[c['direct_identity']]:
                    now=m.identity(ident['pid']);assert now is None or now['start_ticks']!=ident['start_ticks']
            root=B/rootname
            summary=dict(receipt=pin(p),original_status=r.get('status'),original_passed=r['passed'],
                original_complete=r['complete'],kernel_journal=[],retired=[])
            if name.startswith('own-metric'):
                output=root/'owned/native-pool-probe.stdout';jrows=[];library=[]
                for line in output.read_text().splitlines():
                    try:v=json.loads(line)
                    except json.JSONDecodeError:library.append(line);continue
                    if isinstance(v,dict) and 'kind' in v:jrows.append(v)
                    else:library.append(line)
                assert jrows==r['native_rows'] and [v.strip() for v in library]==r['library_stdout']
                # The native controller's Interactive.line() strips whitespace.
                # Preserve the seven original lines, including their trailing spaces.
                summary['raw_library_stdout_lines']=library
                summary['structured_library_normalization']='Interactive.line.strip; exact original lines preserved here'
                assert (root/'owned/native-pool-probe.stderr').read_text()==r['library_stderr']
                retained=[pin(A/n) for n in ['metric-pool-kernel-denial-v1.stdout','metric-pool-kernel-denial-v1.strace','metric-pool-kernel-denial-v1.json']]
            else:
                for case in r['cases']:
                    assert (root/('owned/'+case['label']+'.stdout')).read_text()==case['stdout']
                    assert pin(root/('owned/'+case['label']+'.stderr'))['sha256']==case['stderr_sha256']
                retained=[pin(p)]
            for c in r['commands']:
                for suffix in ['stdout','stderr']:
                    f=root/'owned'/(c['label']+'.'+suffix);v=pin(f);expected=c['logs'][f.name]
                    assert v['bytes']==expected['bytes'] and v['sha256']==expected['sha256']
                    if c['label']=='journal-after' and suffix=='stdout':
                        entries=[json.loads(line) for line in f.read_text().splitlines() if line.strip()]
                        pattern=re.compile(r.get('fault_filter',FAULT))
                        assert [e for e in entries if pattern.search(e.get('MESSAGE',''))]==r['fresh_faults']
                        summary['kernel_journal'].append(dict(entries=len(entries),
                            all_sameboot=all(e.get('_BOOT_ID')==r['boot_id'].replace('-','') for e in entries),
                            fault_entries=r['fresh_faults'],filter=pattern.pattern))
                    if suffix=='stderr' and f.stat().st_size and not name.startswith('own-metric'):
                        data=f.read_text()
                        proof=dict(label=c['label'],bytes=len(f.read_bytes()),UR_success_mentions=data.count('UR_RESULT_SUCCESS'),
                            matched_errors=re.findall(r'(?:UR|ZE)_RESULT_ERROR_[A-Z_]+',data))
                        assert not proof['matched_errors']
                        summary.setdefault('successful_trace_diagnostics',[]).append(proof)
                    item=dict(**v,retained_evidence=retained,
                        reason='Original status/API rows or per-case samples/checks preserved in immutable receipt; representative failure window retained; repeated raw API/transfer/journal histories are redundant.')
                    summary['retired'].append(item);pending.append((f,v))
            if 'syscall_trace' in r:
                f=Path(r['syscall_trace']['path']);v=pin(f)
                assert v['sha256']==r['syscall_trace']['sha256']
                item=dict(**v,retained_evidence=retained,
                    reason='Native reviewer912 closed. Denial/missing-ID windows retain useful evidence; unrelated loader/stat syscalls have no remaining consumer. Actual OA suboperation is not encoded in the full raw trace.')
                summary['retired'].append(item);pending.append((f,v))
            records.append(summary)
        out=dict(schema='strata-resource-diagnostic-raw-retirement-v2',owner='Main',serialized=True,
            completed=False,records=records,logical_bytes=sum(v['bytes'] for _,v in pending),
            file_count=len(pending),original_statuses_unchanged=True,physical_space_reclaimed_not_claimed=True,
            previous_preflight=dict(deleted_files=0,reason='Strict library line equality caught controller stripping trailing ASCII spaces. New manifest preserves exact raw seven lines as well as validating normalized structured values. No measurement result changed.'))
        for summary in records:
            assert pin(Path(summary['receipt']['path']))==summary['receipt']
            for item in summary['retired']:
                for v in item['retained_evidence']:assert pin(Path(v['path']))==v
        for f,v in pending:assert pin(f)==v
        manifest.write_text(json.dumps(out,indent=2)+'\n')
        for f,v in pending:f.unlink()
        out['completed']=True;manifest.write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(manifest=str(manifest),files_retired=len(pending),logical_bytes=out['logical_bytes'],original_statuses_unchanged=True)))


if __name__=='__main__':main()
