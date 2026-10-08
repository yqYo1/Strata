"""Read and pin public reference source without executing its code."""
from pathlib import Path
import datetime, hashlib, json, urllib.request

base = Path(__file__).parent
out = base/'b70-attention-hc-reference-v01402-source-v1'
out.mkdir(mode=0o700)
repository = '0xSero/qwen38-flash-next-b70-offload'
api = 'https://api.github.com/repos/'+repository
def fetch(url):
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'Strata-source-review'})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(4*1024**2+1)
    assert len(data) <= 4*1024**2
    return data
commit_bytes = fetch(api+'/commits/main')
commit = json.loads(commit_bytes)
commit_sha = commit['sha']
tree_bytes = fetch(api+'/git/trees/'+commit['commit']['tree']['sha']+'?recursive=1')
tree = json.loads(tree_bytes)
assert not tree['truncated']
(out/'commit.json').write_bytes(commit_bytes)
(out/'tree.json').write_bytes(tree_bytes)
paths = ['README.md', 'LICENSE', 'kernels/n107-qsa/README.md', 'kernels/n107-qsa/csrc/qsa_select_sycl.sycl',
         'kernels/n107-qsa/csrc/qsa_row_sycl.sycl', 'kernels/n107-hc/README.md', 'kernels/n107-hc/hc_xpu.py']
inventory = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
files = {}
for name in paths:
    metadata = inventory[name]
    assert metadata['size'] <= 128*1024
    url = 'https://raw.githubusercontent.com/'+repository+'/'+commit_sha+'/'+name
    data = fetch(url)
    assert len(data) == metadata['size']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == metadata['sha']
    target = out/'source'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    files[name] = {'url': url, 'github_url': 'https://github.com/'+repository+'/blob/'+commit_sha+'/'+name,
                   'bytes': len(data), 'git_blob_sha1': metadata['sha'], 'sha256': hashlib.sha256(data).hexdigest()}
record = {'active': False, 'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'repository': repository, 'commit': commit_sha, 'tree': tree['sha'],
          'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Primary source capture and identities only. Reference code was not built or executed. Different B70 hardware, PyTorch baseline and masked expert misses are not an equivalent B570 performance or mathematical gate.',
          'files': files, 'gpu_used': False, 'reference_code_executed': False}
(out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'passed': True, 'commit': commit_sha, 'files': {k:v['bytes'] for k,v in files.items()}}, indent=2))
