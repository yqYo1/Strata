#!/usr/bin/env python3
"""Retire only closed own diagnostic roots/mirrors after exact comparison."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        names=['canonical-bank-profile-model-v1.json','canonical-bank-profile-model-comparison-v1.json',
            'canonical-bank-profile-r40960-model-v1.json','canonical-bank-profile-r40960-model-comparison-v1.json',
            'qsa-subprofile-build-host-v2.json','qsa-subprofile-model-v2.json','qsa-subprofile-model-comparison-v2.json','qsa-subprofile-analysis-v2.json']
        refs={n:json.loads((A/n).read_text()) for n in names};pins={n:sha(A/n) for n in names}
        for r in refs.values():
            require(r['passed'] and r['complete'] and not r['active'],'original successful closure')
            if 'boot_id' in r:require(r['boot_unchanged'] and not r['fresh_faults'] and not r['driver_dump_exists'],'healthy model')
            for c in r.get('commands',[]):
                require(module.closed(c),'command closed')
                for old in c.get('owners',[]):
                    now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'prior owned identity absent')
        current=refs['qsa-subprofile-model-v2.json'];keep=Path(current['boundary']['raw_path'])
        prior=refs['canonical-bank-profile-model-v1.json'];old_raw=Path(prior['boundary']['raw_path'])
        require(sha(keep)==current['boundary']['raw_sha256']==prior['boundary']['raw_sha256']==sha(old_raw),'identical current replacement')
        require(all(sha(P/n)==h for n,h in current['source_pins'].items()),'actual current source stable')
        roots=[B/n for n in ('main-canonical-bank-profile-model-root-v1','main-canonical-bank-profile-r40960-model-root-v1','main-qsa-subprofile-model-root-v2','main-qsa-subprofile-build-root-v2')]
        compact={};candidates=[]
        for root in roots:
            native=root/'owned/engine.stderr'
            if native.exists():
                text=native.read_text();compact[root.name]=dict(native_sha256=sha(native),native_bytes=native.stat().st_size,
                    startup=text.splitlines()[:16],lifecycle=[l for l in text.splitlines() if l.startswith('strata prefill dense:')],
                    profile_headers=[l for l in text.splitlines() if re.match(r'strata (?:lm(?: qsa)?|native dense host) profile: v=',l)],
                    final=text.splitlines()[-8:])
            for f in sorted(root.rglob('*')):
                if f.is_file() and f!=keep:
                    require(not f.is_symlink(),'no symlink retirement')
                    candidates.append((f,'Closed successful trace/metadata/host binary or superseded1694 executable; actual source, all individual receipts and compact evidence retained. Current f01e build and latest selected raw excluded.'))
        host=roots[-1]/'owned/host-fixture.stderr';text=host.read_text()
        headers=[l for l in text.splitlines() if l.startswith('strata lm qsa profile: v=1 ')]
        aggregate=[l for l in text.splitlines() if l.startswith('strata lm qsa profile: layer=-1 ')]
        require(len(headers)==len(aggregate)==23,'all23 host outcomes preserved')
        compact['host_fixture']=dict(sha256=sha(host),bytes=host.stat().st_size,headers=headers,aggregate=aggregate,
            case_order=['geometry10','geometry32','missing_geometry','lifetime_normal','lifetime_exception','create0','create7','create13','duplicate_begin','record_begin','record_end','query_failure','elapsed_failure','elapsed_nan','elapsed_negative','pending_finish','chunk_mismatch','query_mismatch','staged_mismatch','GDN_cap_mismatch','cross_layer_cap_mismatch','dispatch_cap_mismatch','destroy_failure'])
        build=roots[-1]/'owned/incremental-sycl-build.stdout';stderr=roots[-1]/'owned/incremental-sycl-build.stderr'
        compact['build']=dict(stdout_sha256=sha(build),first=build.read_text().splitlines()[:6],last=build.read_text().splitlines()[-6:],stderr=stderr.read_text())
        for version in ('qsa-subprofile-v1','qsa-subprofile-v2'):
            proposal=A/version;manifest=json.loads((proposal/'manifest.json').read_text())
            for dirname in ('base','proposed'):
                for f in sorted((proposal/dirname).rglob('*')):
                    if f.is_file():
                        entry=manifest['files'][str(f.relative_to(proposal))]
                        require(sha(f)==entry['sha256'] and f.stat().st_size==entry['bytes'],'exact closed proposal mirror')
                        candidates.append((f,'Completed source review/integration; exact base and actual source inGit, compact source.diff/manifest/fixtures retained. No live consumer of full mirror.'))
        out=A/'closed-qsa-diagnostics-compact-v1.json';target=A/'closed-qsa-diagnostics-retention-v1.json'
        require(not out.exists() and not target.exists(),'fresh compact/retention')
        with out.open('x') as f:json.dump(dict(passed=True,complete=True,active=False,original_receipts=pins,evidence=compact,recipe_sha256=sha(Path(__file__))),f,indent=2);f.write('\n')
        manifest=dict(passed=True,complete=False,active=False,original_receipts=pins,compact=dict(path=out.name,sha256=sha(out)),
            files=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f),reason=reason) for f,reason in candidates],
            kept=dict(path=str(keep),bytes=keep.stat().st_size,sha256=sha(keep),consumer='Main own-prompt-k-cellmajor-v1 same78453 selected-row comparator',budget_bytes=32<<20,next_review='After exact candidate model comparison or superseding this candidate'),
            source_result_status_unchanged=True,finished_utc=module.utc())
        with target.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
        require(all(sha(A/n)==h for n,h in pins.items()),'original receipts unchanged before deletion')
        for entry in manifest['files']:
            f=Path(entry['path']);require(f.stat().st_size==entry['bytes'] and sha(f)==entry['sha256'],'exact retirement identity');f.unlink()
        require(all(sha(A/n)==h for n,h in pins.items()) and sha(keep)==current['boundary']['raw_sha256'],'required evidence unchanged')
        manifest.update(complete=True,files_retired=len(candidates),logical_bytes_retired=sum(x['bytes'] for x in manifest['files']))
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('files_retired','logical_bytes_retired','kept')}),flush=True)


if __name__=='__main__':main()
