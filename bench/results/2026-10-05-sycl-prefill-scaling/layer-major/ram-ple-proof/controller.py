import array, hashlib, json, math, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-layer-major-aot')
out = Path('/tmp/strata-upstream-arc-layer-major-ram-ple')
out.mkdir(exist_ok=True)
proof_dir = Path('/tmp/strata-upstream-arc-layer-major-aot-initial')
proof = json.loads((proof_dir/'summary.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert len(proof['runs']) == 6 and all(r.get('all_bytes_identical_to_jit') for r in proof['runs'])
assert sha(exe) == proof['binary_sha256']
reference = next(r for r in proof['runs'] if r['name'] == 'gpu-mixed-8087-c4096')
env = dict(os.environ, **reference['env'])
for key in list(env):
    if key in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING',
               'STRATA_PREFILL_PRELOAD_PLE', 'STRATA_DUMP_FIRST_LOGITS',
               'STRATA_PREFILL_DUMP_STATE', 'STRATA_PREFILL_DUMP_R', 'STRATA_PREFILL_DUMP_R_ALL'):
        env.pop(key)
env['STRATA_PREFILL_RING'] = '8'
source = out/'source'
source.mkdir(exist_ok=True)
for name in ('sycl/src/prefill/prefill.cpp', 'sycl/src/program/generate.cpp',
             'include/strata/prefill/prefill.hpp', 'sycl/tools/prefill_profile.py'):
    dest = source/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root/name, dest)
report = dict(binary_sha256=sha(exe), source_sha256={str(p.relative_to(source)):sha(p)
              for p in source.rglob('*') if p.is_file()}, runs=[],
              measurement='Production --ple-io ram, without diagnostic PLE preload. Single observations, not repeated medians. Full-table startup loading is outside prefill time but inside process time; actual locking outcome and memory are recorded.')
def save(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
save()
def memory(pid):
    proc = Path('/proc')/str(pid)
    result = {}
    status = (proc/'status').read_text()
    for field in ('VmRSS', 'VmHWM', 'VmSize', 'VmSwap'):
        m = re.search(r'^'+field+r':\s*(\d+) kB$', status, re.M)
        if m: result[field+'_bytes'] = int(m[1])*1024
    seen = set()
    for fd in (proc/'fd').iterdir():
        try:
            if '/dev/dri/' not in os.readlink(fd): continue
            info = (proc/'fdinfo'/fd.name).read_text()
            identity = re.search(r'^drm-client-id:\s*(\d+)$', info, re.M)
            if not identity or identity[1] in seen: continue
            seen.add(identity[1])
            for key,num,unit in re.findall(r'^(drm-(?:total|resident|shared|active|purgeable)-[\w]+):\s*(\d+)(?:\s*(KiB|MiB|GiB))?$', info, re.M):
                result[key+'_bytes'] = result.get(key+'_bytes',0)+int(num)*{'':1, 'KiB':1024, 'MiB':1024**2, 'GiB':1024**3}[unit]
        except OSError: pass
    stat = (proc/'stat').read_text().split(') ')[-1].split()
    result['minor_faults'] = int(stat[7])
    result['major_faults'] = int(stat[9])
    return result
assert memory(os.getpid())['VmRSS_bytes'] > 0
def run_observed(name, args, run_env):
    dest = out/name
    dest.mkdir(exist_ok=True)
    started = time.monotonic()
    record = dict(name=name, args=args, env={k:v for k,v in run_env.items()
                  if k.startswith(('STRATA_', 'SYCL_', 'ONEAPI_', 'NEO_')) or k=='LD_LIBRARY_PATH'},
                  samples=0, sampled_peak_bytes={})
    print(name+' start', flush=True)
    with (dest/'controller.log').open('w') as log,(dest/'memory-samples.jsonl').open('w') as samples:
        child = subprocess.Popen(args, cwd=root, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()
        errors = []
        def observe():
            while not stop.is_set():
                try:
                    for proc in Path('/proc').iterdir():
                        if not proc.name.isdigit(): continue
                        try:
                            cmd = (proc/'cmdline').read_bytes().split(b'\0')
                            if not cmd or cmd[0] != os.fsencode(exe): continue
                            data = memory(int(proc.name))
                            samples.write(json.dumps(dict(time_ns=time.time_ns(), pid=int(proc.name), bytes=data))+'\n')
                            samples.flush()
                            record['samples'] += 1
                            for k,v in data.items(): record['sampled_peak_bytes'][k] = max(record['sampled_peak_bytes'].get(k,0),v)
                        except (OSError, ValueError): pass
                except Exception as e: errors.append(repr(e))
                stop.wait(1)
        observer = threading.Thread(target=observe)
        observer.start()
        # Allow the finite run to finish even if an observer has an error.
        try: rc = child.wait()
        finally: stop.set(); observer.join()
    record.update(exit_code=rc, process_wall_seconds=time.monotonic()-started, observer_errors=errors)
    report['runs'].append(record); save()
    assert rc == 0 and not errors and record['samples'] > 0, record
    return dest, record
def same(a,b):
    assert a.stat().st_size == b.stat().st_size, (a,b,'size')
    with a.open('rb') as f,b.open('rb') as g:
        while True:
            x,y = f.read(1024*1024),g.read(1024*1024)
            assert x == y, (a,b,'bytes')
            if not x: return

# Verify the actual RAM table path against the existing full original proof,
# including odd final chunk, host/GPU residuals and all persistent state.
name = 'ram-gpu-mixed-8087-c4096-proof'
dest = out/name
dest.mkdir(exist_ok=True)
args = list(reference['args'])
args[0] = str(exe)
args[args.index('--ple-io')+1] = 'ram'
run_env = dict(env, STRATA_PREFILL_RING='16', STRATA_DUMP_FIRST_LOGITS=str(dest/'head.bin'),
               STRATA_PREFILL_DUMP_STATE=str(dest/'state.bin'), STRATA_PREFILL_DUMP_R=str(dest/'residual.bin'),
               STRATA_PREFILL_DUMP_R_ALL='1')
dest, record = run_observed(name, args, run_env)
text = (dest/'controller.log').read_text()
table = re.search(r'PLE table (locked in RAM|loaded \(not locked\)) \(--ple-io ram\) in ([\d.]+) s', text)
assert table
record.update(ple_table_locked=table[1]=='locked in RAM', ple_table_startup_seconds=float(table[2]))
for key in ('head.bin', 'state.bin', 'residual.bin'):
    same(dest/key, proof_dir/'gpu-mixed-8087-c4096'/key)
    record[key+'_sha256'] = sha(dest/key)
record['all_head_residual_state_bytes_identical'] = True
values = array.array('f'); values.frombytes((dest/'head.bin').read_bytes())
assert len(values) == 248320 and all(map(math.isfinite, values))
save()
print(name+' PASS all head/residual/state bytes', flush=True)
if '--proof-only' in sys.argv:
    print('PASS RAM PLE complete model proof; long timing remains pending', flush=True)
    sys.exit(0)

a = reference['args']
common = [sys.executable, str(root/'sycl/tools/prefill_profile.py'), '--exe', str(exe),
          '--pack', a[a.index('--pack')+1], '--native', a[a.index('--native')+1],
          '--expert-profile', a[a.index('--expert-profile')+1], '--expert-cache', '128', '--cwd', str(root),
          '--repeats', '1', '--chunk', '8192', '--max-context', '65538', '--kv', 'int8',
          '--multiple-chunks', '--tokens-file', '/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt',
          '--lengths', '32768', '--ple-io', 'ram', '--timeout', '1800']
long_reference = json.loads(Path('/tmp/strata-upstream-arc-layer-major-c8-confirm/baseline-k8v8-32768-c8192/run.json').read_text())
original = long_reference['runs'][0]
# Use the same automatic native-model PLE shard as the original long run.
# The registered-tests proof above explicitly uses its existing Q2_0 shard.
env.pop('STRATA_PLE_GGUF', None)
env.update(long_reference['env'])
for key in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_PREFILL_PRELOAD_PLE'):
    env.pop(key, None)
expected = (original['logits_sha256'], original['logits_count'], original['output_ids'])
for name, compact, major, mode in [('ram-layer-major-transfer',2,1,'transfer'),
                                 ('ram-baseline-wall',0,0,'wall'), ('ram-layer-major-wall',2,1,'wall')]:
    args = common+['--mode',mode,'--label',name,'--output',str(out/name)]
    dest, record = run_observed(name, args, dict(env, STRATA_PREFILL_COMPACT=str(compact),
                    STRATA_PREFILL_LAYER_MAJOR=str(major), STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768'))
    measured = json.loads((dest/'run.json').read_text())['runs'][0]
    assert measured['logits_finite'] and measured['chunks']==4
    assert (measured['logits_sha256'],measured['logits_count'],measured['output_ids']) == expected
    assert 'preload' not in measured and 'phases' not in measured
    assert 'ple_table_locked' in measured and 'ple_table_startup_seconds' in measured
    record.update(prefill_wall_ms=measured['wall_ms'],tokens_per_second=measured['tokens_per_second'],
                  ple_table_locked=measured['ple_table_locked'],ple_table_startup_seconds=measured['ple_table_startup_seconds'],
                  full_head_and_ids_identical_to_original=True)
    if mode=='transfer': record['transfer'] = measured['transfer']
    else: assert 'transfer' not in measured
    save()
    print(name+' PASS full head; '+str(record['tokens_per_second'])+' tok/s', flush=True)
print('PASS RAM PLE study', flush=True)
