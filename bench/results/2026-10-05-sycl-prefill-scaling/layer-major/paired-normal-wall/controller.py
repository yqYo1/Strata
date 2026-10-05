import ast, hashlib, json, os, re, statistics, subprocess, sys, threading, time
from pathlib import Path

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe = Path('/tmp/strata-upstream-arc-layer-major-gpu-jit')
out = Path('/tmp/strata-upstream-arc-layer-major-paired-wall')
out.mkdir(exist_ok=True)
ref = json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text())
a = ref['runs'][0]['args']
env = dict(os.environ, **ref['env'])
for k in list(env):
    if k in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_PREFILL_PRELOAD_PLE',
             'STRATA_SERVE_NO_MTP', 'STRATA_DUMP_FIRST_LOGITS', 'STRATA_PREFILL_DUMP_STATE',
             'STRATA_PREFILL_DUMP_R', 'STRATA_PREFILL_DUMP_R_ALL'):
        env.pop(k)
env.update(STRATA_PREFILL_ATTN_BATCH='128', STRATA_PREFILL_ATTN_LAYOUT='4',
           STRATA_PREFILL_TOPK_TUNED='1', STRATA_GDN_KEYHEAD='0', STRATA_GDN_KEYHEAD_TUNED='1')
c8 = Path('/tmp/strata-upstream-arc-layer-major-c8-confirm')
control = json.loads((c8/'baseline-k8v8-32768-c8192/run.json').read_text())['runs'][0]
candidate = json.loads((c8/'gpu32-k8v8-32768-c8192/run.json').read_text())['runs'][0]
expected = (control['logits_sha256'], control['logits_count'], control['output_ids'])
assert (candidate['logits_sha256'], candidate['logits_count'], candidate['output_ids']) == expected
assert candidate['transfer']['residual_gpu_tokens'] == 32768
proof = json.loads(Path('/tmp/strata-upstream-arc-layer-major-gpu-initial/summary.json').read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest() == proof['binary_sha256']
tree = ast.parse(Path('/tmp/strata-upstream-arc-layer-major-ple-validate-jit.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'memory')
exec(compile(ast.Module(body=[node], type_ignores=[]), '<memory sampler>', 'exec'))
assert memory(os.getpid())['VmRSS_bytes'] > 0
stop = threading.Event()
peaks = {}
def observe():
    with (out/'memory-samples.jsonl').open('w') as handle:
        while not stop.is_set():
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit(): continue
                try:
                    cmd = (proc/'cmdline').read_bytes().split(b'\0')
                    if not cmd or cmd[0] != os.fsencode(exe) or b'--tokens-file' not in cmd: continue
                    name = Path(os.fsdecode(cmd[cmd.index(b'--tokens-file')+1])).stem
                    data = memory(int(proc.name))
                    handle.write(json.dumps(dict(time_ns=time.time_ns(), pid=int(proc.name), run=name, bytes=data))+'\n')
                    handle.flush()
                    record = peaks.setdefault(name, dict(samples=0, sampled_peak_bytes={}))
                    record['samples'] += 1
                    for k,v in data.items(): record['sampled_peak_bytes'][k] = max(record['sampled_peak_bytes'].get(k,0), v)
                except (OSError, ValueError): pass
            (out/'memory-summary.json').write_text(json.dumps(dict(interval_seconds=1,
                note='Monitoring began before all six CLI runs. Process /proc status and DRM fdinfo deduplicated by client ID; sampled peaks can miss transients.',
                runs=peaks), indent=2)+'\n')
            stop.wait(1)
observer = threading.Thread(target=observe)
common = [sys.executable, str(root/'sycl/tools/prefill_profile.py'), '--exe', str(exe),
          '--pack', a[a.index('--pack')+1], '--native', a[a.index('--native')+1],
          '--expert-profile', a[a.index('--expert-profile')+1], '--expert-cache', '128', '--cwd', str(root),
          '--repeats', '1', '--chunk', '8192', '--max-context', '65538', '--kv', 'int8',
          '--multiple-chunks', '--tokens-file', '/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt',
          '--lengths', '32768', '--mode', 'wall', '--timeout', '1200']
records = []
observer.start()
try:
    for pair in range(1,4):
        configurations = [('baseline',0,0,0), ('layer-major',2,1,32768)]
        if pair % 2 == 0: configurations.reverse()
        for label,compact,major,gpu in configurations:
            name = f'{label}-p{pair}'
            dest = out/name
            print(name+' normal wall start', flush=True)
            subprocess.run(common+['--label',name,'--output',str(dest)], cwd=root,
                env=dict(env, STRATA_PREFILL_COMPACT=str(compact), STRATA_PREFILL_LAYER_MAJOR=str(major),
                         STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu)), check=True, timeout=1500)
            run = json.loads((dest/'run.json').read_text())['runs'][0]
            assert observer.is_alive(), 'Memory observer failed; measurement incomplete'
            assert run['logits_finite'] and run['chunks'] == 4
            assert (run['logits_sha256'],run['logits_count'],run['output_ids']) == expected
            assert not any(k in run for k in ('transfer','phases','preload'))
            records.append(dict(pair=pair,label=label,wall_ms=run['wall_ms'],
                                tokens_per_second=32768/(run['wall_ms']/1000),report=str(dest/'run.json')))
            (out/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
            print(name+' complete head and IDs bit-identical to SAME C8192 original',flush=True)
finally:
    stop.set(); observer.join()
assert len(peaks) == 6 and all(r['samples'] > 0 for r in peaks.values()), 'Missing process memory observations'
medians = {label:statistics.median(r['wall_ms'] for r in records if r['label']==label)
           for label in ('baseline','layer-major')}
summary = dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               tokens=32768,chunk=8192,max_context=65538,kv='int8',pairs=records,median_wall_ms=medians,
               median_tokens_per_second={k:32768/(v/1000) for k,v in medians.items()},
               elapsed_reduction_percent=100*(1-medians['layer-major']/medians['baseline']),
               measurement='Three pairs, reversed order in pair two; no phase markers, transfer timers or PLE preload; fixed 64K context and complete first-head/ID equality.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
