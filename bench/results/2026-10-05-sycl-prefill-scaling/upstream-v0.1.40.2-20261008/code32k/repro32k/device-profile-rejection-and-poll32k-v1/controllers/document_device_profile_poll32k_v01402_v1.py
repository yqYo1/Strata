"""Append current measured evidence and refresh nested public archive manifests."""
from pathlib import Path
import datetime,hashlib,json,shutil
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling'
upstream=parent/'upstream-v0.1.40.2-20261008'
code=upstream/'code32k';repro=code/'repro32k'
out=repro/'device-profile-rejection-and-poll32k-v1'
summary=json.loads((out/'summary.json').read_text())
assert summary['completed'] and summary['poll_four_fresh32768_reads_passed'] and not summary['device_timestamp_configuration_accepted']
target=out/'controllers'/Path(__file__).name;assert not target.exists();shutil.copy2(Path(__file__),target)
updates=[
    (repro/'README.md','device-profile-rejection-and-poll32k-v1/README.md'),
    (code/'README.md','repro32k/device-profile-rejection-and-poll32k-v1/README.md'),
    (upstream/'README.md','code32k/repro32k/device-profile-rejection-and-poll32k-v1/README.md'),
    (parent/'README.md','upstream-v0.1.40.2-20261008/code32k/repro32k/device-profile-rejection-and-poll32k-v1/README.md'),
    (repro/'full256k-watchdog-abort-and-uniform32k-v1/README.md','../device-profile-rejection-and-poll32k-v1/README.md'),
    (repro/'profile-definition-audit-and-poll-rebase-v1/README.md','../device-profile-rejection-and-poll32k-v1/README.md')]
for path,link in updates:
    text=path.read_text();assert 'device-profile-rejection-and-poll32k-v1/README.md' not in text
    text+='\nThe [subsequent GPU timestamp rejection and polling gate]('+link+') retain the actual earlier VTune startup failure on this Ryzen processor and a failed logged 32768-token unitrace GPU timestamp run. The application watchdog stopped at layer 26 of chunk 16384; no new kernel fault was recorded and post-exit GPU/runtime health passed. Its empty/incomplete trace and durations are excluded from performance conclusions. The DD5-based polling candidate subsequently passed four fresh 32768-token reads, all head/used-state/output/logprob/MTP comparisons, actual disk continuation and normal exit. It is still private: quiet matched inputs of at least 32768 tokens and the complete physical 256K gate remain required. First and repeated full reads must be reported separately; the underlying pending-counter cause is unresolved.\n'
    path.write_text(text)
reason='Retain rejected logged32K GPU timestamp configuration, its raw watchdog evidence and post-exit health; correct the incomplete VTune assessment. Archive DD5-based polling candidate source/link checks and completed four-fresh32K/disk-continuation/normal-exit gate. Quiet>=32K and physical256K/adoption remain pending.'
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for directory in [out,repro/'full256k-watchdog-abort-and-uniform32k-v1',repro/'profile-definition-audit-and-poll-rebase-v1',repro,code,upstream,parent]:
    manifest=directory/'manifest.json'
    d=json.loads(manifest.read_text()) if manifest.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':summary['source_commit']}
    d['revised_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['revision_reason']=reason
    d['files']={str(path.relative_to(directory)):{'bytes':path.stat().st_size,'sha256':digest(path)} for path in sorted(directory.rglob('*')) if path.is_file() and path!=manifest}
    manifest.write_text(json.dumps(d,indent=2)+'\n')
    print(str(manifest.relative_to(root)),len(d['files']))
