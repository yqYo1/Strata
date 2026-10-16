#!/usr/bin/env python3
"""Compact closed first-check UR evidence after the component matrix passed."""
import collections
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        refs={n:A/n for n in ('qsa-device-batch-small-v1.json','qsa-device-batch-matrix-v1.json','qsa-device-batch-dispatch-v1.json')}
        records={n:json.loads(p.read_text()) for n,p in refs.items()}
        owner=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for r in records.values():
            require(r['passed'] and r['complete'] and not r['active'],'actual closed qualification')
            for c in r['commands']:
                require(owner.closed(c),'closed commands')
                for old in c['owners']:
                    now=owner.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'retired owners')
        r=records['qsa-device-batch-small-v1.json']
        path=B/'main-qsa-device-batch-small-root-v1/owned/QSA-batch-small.stderr'
        require(sha(path)==r['native_stderr_sha256'],'actual first-check trace')
        lines=path.read_text().splitlines()
        returns=[m.groups() for line in lines if (m:=re.search(r'<--- (ur\w+)\(.*\) -> (UR_RESULT_\w+);$',line))]
        entries=re.findall(r'^\s*---> (ur\w+)\s*$','\n'.join(lines),flags=re.M)
        counts=collections.Counter(fn for fn,result in returns)
        require(counts==collections.Counter(entries) and all(result=='UR_RESULT_SUCCESS' for _,result in returns),'complete successful API entry/results')
        kernel_lines=[line for line in lines if '<--- urKernelCreate(' in line]
        joined='\n'.join(kernel_lines)
        require('append_completed_blocks' in joined and 'finalize_device_batch' in joined,'actual candidate kernel creations')
        out=A/'qsa-device-batch-trace-compact-v1.json';target=A/'qsa-device-batch-trace-retention-v1.json'
        require(not out.exists() and not target.exists(),'fresh retention')
        result=dict(passed=True,complete=True,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
            references={n:sha(p) for n,p in refs.items()},component=r['component'],
            original_trace=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)),
            API_counts=dict(sorted(counts.items())),API_returns_total=len(returns),all_API_returns_success=True,
            retained_kernel_creation_lines=kernel_lines,
            fresh_faults=r['fresh_faults'],boot_unchanged=r['boot_unchanged'],driver_dump_exists=r['driver_dump_exists'])
        with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        require(json.loads(out.read_text())==result,'compact roundtrip')
        manifest=dict(passed=False,complete=False,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
            replacement=out.name,replacement_sha256=sha(out),
            deleted=[dict(result['original_trace'],reason='All API results, actual candidate kernel names and individual component/fault outcomes verified; broad matrix/strict production dispatch closed PASS, no consumer requires successful full API history.')])
        with target.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
        require(sha(path)==result['original_trace']['sha256'],'identity before unlink');path.unlink()
        manifest.update(passed=True,complete=True,removed_bytes=result['original_trace']['bytes'],finished_utc=owner.utc())
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(passed=True,removed_bytes=manifest['removed_bytes'],API_returns_total=len(returns))))


if __name__=='__main__':main()
