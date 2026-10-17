#!/usr/bin/env python3
"""Retire closed-run duplicates after committed compact evidence is verified."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    if not __debug__:raise RuntimeError('Run without Python -O')
    dst=A/'closed-placement-diagnostics-retention-v1.json'
    extract=A/'closed-placement-diagnostics-excerpts-v1.json'
    assert not dst.exists() and not extract.exists()
    hp=A/'direct_owner_transient_artifacts_v1.py'
    assert sha(hp)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec=importlib.util.spec_from_file_location('retention_owner',hp);owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=P).decode().split('\0'))
    mapping=[]
    for phase in ('diagnostic','clean'):
        for arm in ('gpu32768','gpuall'):
            for r in range(1 if phase=='diagnostic' else 3):
                stem=f'residual64k-placement-{phase}-{arm}-r{r}-v1';mapping.append((A/f'{stem}.json',B/f'main-{stem}'))
    for stem in ('command-service-model-diagnostic-off-r0-v1','command-service-model-diagnostic-on-r0-v1','command-service-model-diagnostic-on-r0-v2','event-service-stress-v1','event-service-clock-age-baseline-v1','event-service-clock-age-aged-v2'):
        mapping.append((A/f'{stem}.json',B/f'main-{stem}'))
    for mode in ('off','on','error'):
        mapping.append((A/f'command-profile-fixture-{mode}-v2.json',B/f'main-command-profile-fixture-{mode}-v2'))
    mapping.append((A/'moe-accum-plan-host-v1.json',B/'main-moe-accum-plan-host-v1'))
    keep={B/'main-event-service-clock-age-aged-v2/owned/timestamp-stress.stdout',B/'main-event-service-clock-age-aged-v2/owned/timestamp-stress.stderr'}
    candidates=[];excerpts=[]
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=shutil.disk_usage(B).free
        for receipt,root in mapping:
            r=json.loads(receipt.read_text());assert r['ownership_closed'] and not r['active']
            rel=str(receipt.relative_to(P));blob=subprocess.check_output(['git','cat-file','blob','HEAD:'+rel],cwd=P);assert hashlib.sha256(blob).hexdigest()==sha(receipt)
            for c in r['commands']:
                assert owned.closed(c)
                for old in c['owners']+[c['direct_identity']]:
                    now=owned.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
            pins={name:value['sha256'] for c in r['commands'] for name,value in c['logs'].items()}
            for f in sorted((root/'owned').glob('*')):
                if not f.is_file() or f.suffix not in ('.stdout','.stderr'):continue
                assert not f.is_symlink() and f.resolve().parent==root.joinpath('owned').resolve()
                h=sha(f)
                if f.name in pins:assert h==pins[f.name],str(f)
                if f.name=='engine.stderr' and 'native_log_sha256' in r:assert h==r['native_log_sha256']
                text=f.read_text(errors='strict')
                selected=[line for line in text.splitlines() if line.startswith(('strata prefill:','strata MoE window:','strata command profile:','strata command invalid:','strata mtp: prompt expert phase=','strata serve: prompt ','strata generate: GPU ','fixture fallback drain','case ','fixture_status=')) or re.search(r'(?i)(injected-before-drain|missing_checked_retirement|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup.*fail)',line)]
                if selected:excerpts.append(dict(path=str(f),sha256=h,lines=selected))
                if f not in keep:candidates.append((f,receipt,'Closed full text represented by committed receipt and verified compact excerpt; no current tensor/session consumer.'))
            if 'benchmark_result' in r:
                for result in (root/'stock-results').glob('*/results.json'):
                    assert json.loads(result.read_text())==r['benchmark_result']
                    candidates.append((result,receipt,'Exact duplicate of the official benchmark result embedded in the committed receipt.'))
                    md=result.with_name('results.md')
                    if md.is_file():candidates.append((md,receipt,'Generated rendering of the same benchmark result; individual samples retained in Git.'))
        assert sum(f.stat().st_size for f in keep)<=4<<20
        extract.write_text(json.dumps(dict(schema='strata-closed-placement-diagnostic-excerpts-v1',scope='Relevant native progress/configuration, placement, counters, outcomes and timestamp errors; no replacement qualification. Original receipt status is unchanged.',excerpts=excerpts),indent=2)+'\n')
        assert extract.stat().st_size<256<<10
        # These closed worker launch/observation duplicates have compact, reviewed
        # source/model/task/dispatch/completion records in the committed summaries.
        for n in (998,999,1000,1002,1003):
            summary=A/('orca-research-wave998-1000-summary-v1.json' if n<=1000 else 'orca-research-wave1002-1003-summary-v1.json')
            value=json.loads(summary.read_text());assert value['all_workers_released']
            for f in A.glob(f'orca-research-round{n}-*-v1.json'):
                if str(f.relative_to(P)) not in tracked:
                    candidates.append((f,summary,'Superseded closed worker startup/lifecycle observation; reviewed task/model/report/settlement evidence is committed in the compact wave summary.'))
        deleted=[];seen=set()
        for f,replacement,reason in candidates:
            if str(f) in seen:continue
            seen.add(str(f));assert f.is_file() and not f.is_symlink()
            st=f.stat();h=sha(f);assert st.st_nlink==1
            item=dict(path=str(f),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=h,replacement=str(replacement),replacement_sha256=sha(replacement),excerpt=str(extract),excerpt_sha256=sha(extract),reason=reason)
            assert sha(f)==h;f.unlink();deleted.append(item)
        kept=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(keep)]
        manifest=dict(schema='strata-closed-placement-diagnostics-retention-v1',source_sha256=sha(__file__),committed_evidence_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),runs_closed=len(mapping),original_receipts_unchanged=True,deleted=deleted,logical_bytes_removed=sum(x['bytes'] for x in deleted),allocated_bytes_removed=sum(x['allocated_bytes'] for x in deleted),free_bytes_before=before,free_bytes_after=shutil.disk_usage(B).free,free_space_scope='Observed filesystem free bytes; snapshots/other activity mean this is not a physical-reclamation guarantee.',kept_full_trace=dict(files=kept,owner='Main clock-conversion diagnosis',consumer='Exact UR event creation/retention/query sequence for the reproducible aged start-before-submit mechanism, to compare with prospective native raw-tick diagnostic.',budget_bytes=4<<20,next_review='After raw-tick comparison or installed-source mapping resolves/supersedes the conversion/reuse question, retire this full trace; all141rawnumeric values are already in Git.'),excerpt=dict(path=str(extract),sha256=sha(extract)),scope='Only closed measurement text/duplicate benchmark renderings and current-wave untracked lifecycle observations. No models, source, binaries, last fixture inputs, services or GPU state altered.')
        dst.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ['runs_closed','logical_bytes_removed','allocated_bytes_removed']}))
        print(json.dumps({'deleted_files':len(deleted),'kept_representative_trace_bytes':sum(x['bytes'] for x in kept)}))
if __name__=='__main__':main()
