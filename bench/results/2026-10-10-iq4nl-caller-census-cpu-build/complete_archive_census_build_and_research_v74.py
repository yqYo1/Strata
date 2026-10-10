"""Finish archive after generated configure log whitespace failed git diff check."""
from pathlib import Path
import fcntl, hashlib, json, subprocess, textwrap
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
parent=B/'archive_census_build_and_research_v74.py'
source=parent.read_text()
namespace={'__file__':str(parent)}
exec(source.split("with (B/'owned-v0141-measurement.lock')",1)[0],namespace)
C,W,CP,A,R=[namespace[k] for k in ['C','W','CP','A','R']]
ident,write,commit,git=[namespace[k] for k in ['ident','write','commit','git']]
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert git(C,'rev-parse','HEAD')=='484f38051c5cdc46313f232b9f746a931ab9e57b'
    assert git(W,'rev-parse','HEAD')=='313d3a9cc43728401f392db5b10bf4084d6fb35f'
    assert not A.exists() and not (R/'report-registry-v74.json').exists()
    run=B/'iq4nl-caller-census-cpu-build-v1'
    d=json.loads((run/'record.json').read_text())
    assert d['passed'] and d['complete'] and not d['active']
    original=CP/'configure.stdout'
    assert ident(original)==ident(run/'configure.stdout')
    lines=original.read_text().splitlines()
    (CP/'configure-compact.txt').write_text('\n'.join(x.rstrip() for x in lines)+'\n')
    write(CP/'configure-compaction.json',dict(original_path=str(run/'configure.stdout'),original_identity=ident(original),derived_path='configure-compact.txt',derived_identity=ident(CP/'configure-compact.txt'),transformation='Only trailing whitespace stripped from generated successful configure text; original log and original build receipt unchanged. Archive whitespace check failed before commit, not build failure.'))
    subprocess.run(['git','rm','--cached','--',str(original.relative_to(C))],cwd=C,check=True)
    original.unlink()
    namespace['copy'](Path(__file__),CP/Path(__file__).name)
    write(CP/'archive-file-identities.json',{str(p.relative_to(CP)):ident(p) for p in sorted(CP.rglob('*')) if p.is_file() and p.name!='archive-file-identities.json'})
    ccommit=commit(C,str(CP.relative_to(C)),'docs(sycl): preserve closed caller-census engine build')
    namespace.update(d=d,run=run,ccommit=ccommit)
    marker="    prev=R/'report-registry-v73.json'"
    tail=marker+source.split(marker,1)[1]
    exec(textwrap.dedent(tail),namespace)
