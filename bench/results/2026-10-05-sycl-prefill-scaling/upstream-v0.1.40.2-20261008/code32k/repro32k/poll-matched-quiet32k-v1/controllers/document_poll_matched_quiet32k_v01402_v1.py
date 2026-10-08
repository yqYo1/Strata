"""Link terminal quiet32K evidence and validate all affected archive manifests."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root/'bench/results/2026-10-05-sycl-prefill-scaling'
upstream = parent/'upstream-v0.1.40.2-20261008'
code = upstream/'code32k'
repro = code/'repro32k'
out = repro/'poll-matched-quiet32k-v1'
s = json.loads((out/'summary.json').read_text())
assert s['completed'] and s['passed'] and s['fresh_full_reads'] == 16 and not s['adopted']
assert abs(s['effects']['subsequent_full_reads']['prefill_tok_s_change_percent']) < 0.1
assert s['minimum_performance_input_tokens'] == 32768 and not s['physical256k_sequence_passed']
target = out/'controllers'/Path(__file__).name
assert not target.exists()
shutil.copy2(Path(__file__), target)
readme = out/'README.md'
text = readme.read_text()
for a, b in [('Matched32K', 'Matched 32K'), ('chunk8192', 'chunk 8192'), ('cache128', 'cache 128'),
             ('MTP4', 'MTP 4'), ('after32', 'after 32'), ('a10 us', 'a 10 us'), ('input32768', 'input 32768'),
             ('output64', 'output 64'), ('RESUME0', 'RESUME 0'), ('REUSED0', 'REUSED 0'),
             ('physical256K', 'physical 256K'), ('all262144', 'all 262144'), ('fresh32K', 'fresh 32K')]:
    text = text.replace(a, b)
readme.write_text(text)
updates = [(repro/'README.md', 'poll-matched-quiet32k-v1/README.md'),
           (code/'README.md', 'repro32k/poll-matched-quiet32k-v1/README.md'),
           (upstream/'README.md', 'code32k/repro32k/poll-matched-quiet32k-v1/README.md'),
           (parent/'README.md', 'upstream-v0.1.40.2-20261008/code32k/repro32k/poll-matched-quiet32k-v1/README.md'),
           (repro/'device-profile-rejection-and-poll32k-v1/README.md', '../poll-matched-quiet32k-v1/README.md'),
           (repro/'full256k-watchdog-abort-and-uniform32k-v1/README.md', '../poll-matched-quiet32k-v1/README.md'),
           (repro/'profile-definition-audit-and-poll-rebase-v1/README.md', '../poll-matched-quiet32k-v1/README.md')]
for path, link in updates:
    text = path.read_text()
    assert 'poll-matched-quiet32k-v1/README.md' not in text
    text += '\nThe [subsequent quiet polling comparison]('+link+') completed sixteen fresh 32768-token reads in control/candidate/candidate/control order on the same B570. First process reads are separate from later full reads. Subsequent prefill was 427.510 tok/s for DD5 and 427.225 tok/s for polling backoff (-0.067%); decode was 16.486 and 16.373 tok/s (-0.686%). All output IDs, logprobs and visible MTP counts matched, with RESUME 0 and REUSED 0, normal QUIT and no new kernel fault. There is no useful prefill gain, and this small sample does not establish a decode benefit or regression. Polling backoff is not adopted. The complete physical 256K gate and pending native counter diagnosis remain unresolved; diagnostic/profiled durations are excluded.\n'
    path.write_text(text)

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

directories = [out, repro/'device-profile-rejection-and-poll32k-v1',
               repro/'full256k-watchdog-abort-and-uniform32k-v1',
               repro/'profile-definition-audit-and-poll-rebase-v1', repro, code, upstream, parent]
manifests = []
for directory in directories:
    manifest = directory/'manifest.json'
    d = json.loads(manifest.read_text()) if manifest.exists() else {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_commit': s['source_commit']}
    d.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             revision_reason='Archive sixteen matched fresh32768 quiet reads with first/subsequent and per-input rates separated. Output/logprob/MTP/freshness/normal-exit gates pass; no useful polling prefill speed gain, no adoption or physical256K/stall-fix claim.')
    d['files'] = {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                  for p in sorted(directory.rglob('*')) if p.is_file() and p != manifest}
    manifest.write_text(json.dumps(d, indent=2)+'\n')
    files = {str(p.relative_to(directory)): p for p in directory.rglob('*') if p.is_file() and p != manifest}
    assert files.keys() == d['files'].keys()
    for name, path in files.items():
        assert path.stat().st_size == d['files'][name]['bytes'] and digest(path) == d['files'][name]['sha256']
    manifests.append({'path': str(manifest), 'files': len(files), 'sha256': digest(manifest)})
    print(str(manifest.relative_to(root)), len(files), flush=True)
for path in (out/'controllers').glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
receipt = {'active': False, 'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'controller_sha256': digest(Path(__file__)), 'archive_summary_sha256': digest(out/'summary.json'),
           'exact_file_sets_sizes_hashes_validated': True, 'all_archived_controllers_ast_passed': True,
           'minimum_input_tokens': 32768, 'fresh_reads': 16, 'adopted': False, 'manifests': manifests}
(base/'poll-matched-quiet32k-v01402-archive-validation-v1.json').write_text(json.dumps(receipt, indent=2)+'\n')
