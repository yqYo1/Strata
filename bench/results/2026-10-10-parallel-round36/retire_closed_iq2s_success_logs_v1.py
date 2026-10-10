"""Retire only the closed parent numerical/trace duplicates, after committed review."""
from pathlib import Path
import datetime, fcntl, hashlib, json, subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-parallel-round36'
Q=P/'bench/results/2026-10-10-iq2s-actual-cohort-qualification'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for cwd,commit in ((P,'906cac75a6f3f48a85e059c64c45393b96a2edc4'),(W,'3ff794d1aa9b0756b436e2fb22eaccf6dea3353b')):
        assert subprocess.check_output(['git','log','-1','--format=%H'],cwd=cwd,text=True).strip()==commit
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=cwd,text=True).strip()
    full=Q/'iq2s-actual-cohort-correctness-v1/complete-numerical-records.csv'
    assert sha(full)=='59c9013d1b6392279f9aad3a4b07ab3e625fa9a6bad2d07879d50d8a0dca2f17'
    tasks=[
      ('iq2s-actual-cohort-correctness-v1','correctness-observed/stdout.csv',full,'Duplicate of complete committed numerical records'),
      ('iq2s-actual-cohort-correctness-v1','correctness-observed/stderr.txt',Q/'iq2s-actual-cohort-correctness-v1/correctness-observed-file-identities.json','Successful loader diagnostics; scoped runtime checks and hashes retained'),
      ('iq2s-actual-cohort-correctness-v1','correctness-syscalls.txt',Q/'iq2s-actual-cohort-correctness-v1/trace-identities.json','Successful scoped pre-exec-to-exit audit; controller, outcome and trace identity retained'),
      ('iq2s-actual-cohort-asan-correctness-v1','asan-untraced-correctness/stdout.csv',full,'Byte-identical sanitizer duplicate; complete numerical rows and original sanitizer receipt retained')]
    records={}
    entries=[]
    for name,rel,replacement,reason in tasks:
        receipt=B/name/'record.json';r=json.loads(receipt.read_text())
        assert r['passed']and r['complete']and not r['active']and not r['cleanup']and not r['survivors']
        records[str(receipt)]=sha(receipt)
        p=B/name/rel;assert p.is_file()and not p.is_symlink()
        entry=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason=reason,replacement=str(replacement),replacement_sha256=sha(replacement),replacement_commit='906cac75a6f3f48a85e059c64c45393b96a2edc4',status='planned')
        if rel.endswith('stdout.csv'):assert entry['sha256']==sha(full)
        else:
            ids=json.loads(replacement.read_text());assert any(x['sha256']==entry['sha256']and x['bytes']==entry['bytes']for x in ids)
        subprocess.run(['git','ls-files','--error-unmatch',str(replacement.relative_to(P))],cwd=P,check=True,stdout=subprocess.DEVNULL,timeout=10)
        entries.append(entry)
    m=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Closed parent successful numerical/loader/trace duplicates only',consumer_closure='R119 actual-weight runtime admission completed and fully read. Current R123 uses immutable source and committed timing/linked archive; R124 uses unrelated immutable GDN source, neither requires these parent raw logs.',no_active_runs=True,failure_evidence_unchanged=True,original_receipts=records,files=entries)
    target=A/'parent-success-log-retirement.json';assert not target.exists()
    def save():target.write_text(json.dumps(m,indent=2)+'\n')
    save()
    for e in entries:
        p=Path(e['path']);assert sha(p)==e['sha256']and p.stat().st_size==e['bytes'];p.unlink();e['status']='deleted';save()
    assert all(sha(p)==v for p,v in records.items())
    m.update(complete=True,total_deleted_bytes=sum(e['bytes']for e in entries),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
    dest=A/Path(__file__).name;dest.write_bytes(Path(__file__).read_bytes())
    identity=A/'archive-file-identities.json';old=json.loads(identity.read_text())
    for p in (target,dest):old[str(p.relative_to(A))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for rel,v in old.items():assert sha(A/rel)==v['sha256']and (A/rel).stat().st_size==v['bytes']
    identity.write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps(dict(files=len(entries),deleted_bytes=m['total_deleted_bytes'],manifest=str(target))))
