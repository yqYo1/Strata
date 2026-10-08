"""Document the scoped terminal qualification and refresh affected manifests."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil, subprocess

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root / 'bench/results/2026-10-05-sycl-prefill-scaling'
upstream = parent / 'upstream-v0.1.40.2-20261008'
code = upstream / 'code32k'
repro = code / 'repro32k'
out = repro / 'uniform-full256k-and-qsa-source-v1'
s = json.loads((out / 'summary.json').read_text())
assert s['passed'] and s['fresh_full_reads'] == 2 and s['fresh32k_reads'] == 4
assert s['physical_cells_per_layer'] == 262144 and s['last_physical_cell'] == 262143
assert s['ignored_saved_kv_tensor_bytes'] == 0 and not s['qsa_engine_compiled'] and not s['qsa_gpu_tested']
assert s['minimum_performance_input_tokens'] == 32768 and s['diagnostic_durations_excluded']
assert not (out / 'README.md').exists()
(out / 'README.md').write_text('''# Complete DD5 physical 256K qualification and next QSA source candidate

The DD5 binary (`dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34`) completed the full capacity and state sequence on the Intel Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB system RAM. The runtime was NEO 26.31.39395.14, IGC 2.41.5 and oneAPI 2026.1.1. Context was 262144, chunk 8192, int8 KV with 32768 resident cells, requested expert cache 128, five CPU workers and MTP 4. Prompt caching used one checkpoint at 262139, root 0 and turn token -1. The actual argv, environment, boot ID and owned PID/start ticks are in the [terminal receipt](terminal-full256k-r7/record.json).

Two fresh 262140-token inputs, each with four requested outputs, reached physical cell 262143. A different fresh 32768-token input separated the two full reads; both full reads reported RESUME 0. Head, used state, output IDs, logprobs and visible MTP counts matched. All 13 saved KV layers contain 262144 physical cells. Every saved state and KV tensor byte matched between the two full saves and again after an actual disk RESTORE/SAVE round trip; no saved KV bytes were ignored.

The restored 262142-token input with two requested outputs resumed at 262139 and clipped the last verification window to two tokens, with the same output and head as before restoration. An actual 32K disk restoration also reproduced the live continuation. Inputs of 262144 plus one output and 262142 plus three outputs were rejected before GPU verification or token output. A later fresh 32768-token input matched the baseline. All twelve request gates and six session operations passed; QUIT returned zero, no forced cleanup or owned process remained, and the kernel comparison found no new GPU fault. No reset, rebind, reboot, service, package or global setting changed.

This was a diagnostic run with flushed UR, ZE, ZEL and Strata logs. Its times are excluded from performance comparisons. The raw stderr is 67452840326 bytes; its complete SHA-256 is in [summary.json](summary.json), computed by the terminal controller after engine exit. Large head, state, session and stderr files remain in the private run directory, with hashes and tensor comparisons in the receipt. This completed sequence qualifies this DD5 binary for the tested lifecycle; it does not establish the cause of the earlier pending native-event failures. The earlier [second-full watchdog failure](../full256k-watchdog-abort-and-uniform32k-v1/README.md) and [profiler rejection](../device-profile-rejection-and-poll32k-v1/README.md) remain preserved. The [quiet comparison](../poll-matched-quiet32k-v1/README.md) uses only inputs of at least 32768 tokens, with first process reads separate from later full reads; polling backoff gave no useful prefill gain and remains unadopted.

The separate next source candidate ports the existing upstream twelve-head reduction only to the SG32/transposed QSA arm. It uses the actual subgroup lane and group identifiers for cell/head routing. Dot expressions, scales, 64-cell splitting, softmax, PV, merge, launch attributes, barriers, queues, events and host lifetimes are preserved. Source counting reduces exchanges per lane from 60 to 16; this is not a measured speed gain. The exact helper body passed an independent CPU FP32 comparison for 82176 outputs in 2568 groups with ASan/UBSan, including signed zero, subnormal, cancellation, basis and random inputs with finite intermediate sums. The same helper body was retained through the later source revisions, so that proof was reused by its exact hash, not rerun as a GPU test. A separate CPU semantic enumeration checked 1024 abstract subgroup arrangements and masks. The [source review](source-review/qsa-reduce12-sg32-v01402-source-v3/record.json) records the limits; other existing variants retain their earlier workgroup-derived mapping.

At this archive boundary the QSA engine had not been compiled or GPU-tested, and nothing was adopted or installed in production. Its prepared builder requires terminal full DD5 qualification and absence of the owned model/debugger before compiling one object. Its prepared first GPU gate uses four fresh 32768-token inputs alternating two fixtures, both qualified baselines, head/used-state/output/logprob/MTP checks and actual disk continuation, with diagnostic logs and owned stopped native-counter capture on failure. It still needs compilation, that numerical gate, a clean 32K-or-longer speed comparison with first/later reads separate, and its own complete physical 256K lifecycle sequence. The DD5 capacity result cannot be inherited by a different binary.

The three owned CPU debugger fixtures for the counter-field reader passed; actual NEO field compatibility was not exercised because this run did not stop in a native-event failure. The pinned B70 reference and MIT license are retained under `source-review`, with an applicability review under `checks`; its reported hardware results are not B570 measurements. The failed first CPU arithmetic fixture compilation, corrected `<exception>` fixture and passing second run are retained separately.
''')
shutil.copy2(Path(__file__), out / 'controllers' / Path(__file__).name)
updates = [(repro/'README.md','uniform-full256k-and-qsa-source-v1/README.md'),
           (code/'README.md','repro32k/uniform-full256k-and-qsa-source-v1/README.md'),
           (upstream/'README.md','code32k/repro32k/uniform-full256k-and-qsa-source-v1/README.md'),
           (parent/'README.md','upstream-v0.1.40.2-20261008/code32k/repro32k/uniform-full256k-and-qsa-source-v1/README.md'),
           (repro/'device-profile-rejection-and-poll32k-v1/README.md','../uniform-full256k-and-qsa-source-v1/README.md'),
           (repro/'full256k-watchdog-abort-and-uniform32k-v1/README.md','../uniform-full256k-and-qsa-source-v1/README.md'),
           (repro/'profile-definition-audit-and-poll-rebase-v1/README.md','../uniform-full256k-and-qsa-source-v1/README.md'),
           (repro/'poll-matched-quiet32k-v1/README.md','../uniform-full256k-and-qsa-source-v1/README.md')]
for path, link in updates:
    text = path.read_text()
    assert 'uniform-full256k-and-qsa-source-v1/README.md' not in text
    text = text.replace('The complete physical 256K gate and pending native counter diagnosis remain unresolved;',
                        'At that quiet 32K boundary, the complete physical 256K gate and pending native counter diagnosis were unresolved;')
    text += '\nThe [later complete DD5 physical 256K sequence]('+link+') passed two fresh full reads through cell 262143, all 13 saved KV layers with 262144 cells, every saved state/KV tensor byte on repetition and actual disk restoration, a clipped two-token tail, both capacity refusals and a later fresh 32768-token input. Head/used-state/output/logprob/MTP gates and normal owned exit passed without a new GPU fault or reset. Diagnostic durations are excluded; this does not resolve the earlier pending-event cause. A separate twelve-head QSA reduction has CPU arithmetic/routing checks and source-only preparation, with no engine compile, GPU numerical result, speed gain or inherited capacity proof at this archive boundary. Every future performance comparison uses at least 32768 input tokens and separates first/later full reads.\n'
    path.write_text(text)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
directories = [out, repro/'device-profile-rejection-and-poll32k-v1',
               repro/'full256k-watchdog-abort-and-uniform32k-v1',
               repro/'profile-definition-audit-and-poll-rebase-v1',
               repro/'poll-matched-quiet32k-v1', repro, code, upstream, parent]
manifests = []
for directory in directories:
    manifest = directory/'manifest.json'
    d = json.loads(manifest.read_text()) if manifest.exists() else {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_commit':source_commit}
    d.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             revision_reason='Complete DD5 physical256K/repeated full/disk restoration/clipped tail/refusal/later32K qualification; separately archive CPU-tested source-only QSA candidate. No new speed gain or stall-cause claim.')
    files = {str(p.relative_to(directory)):p for p in sorted(directory.rglob('*')) if p.is_file() and p != manifest}
    d['files'] = {name:{'bytes':path.stat().st_size,'sha256':digest(path)} for name,path in files.items()}
    manifest.write_text(json.dumps(d,indent=2)+'\n')
    assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != manifest} == set(d['files'])
    for name,path in files.items():
        assert path.stat().st_size == d['files'][name]['bytes'] and digest(path) == d['files'][name]['sha256']
    manifests.append({'path':str(manifest),'files':len(files),'sha256':digest(manifest)})
    print(str(manifest.relative_to(root)),len(files),flush=True)
for path in (out/'controllers').glob('*.py'):
    ast.parse(path.read_text(),filename=str(path))
validation = {'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'controller_sha256':digest(Path(__file__)),'summary_sha256':digest(out/'summary.json'),
              'exact_file_sets_sizes_hashes_validated':True,'all_archived_controllers_ast_passed':True,
              'minimum_performance_input_tokens':32768,'DD5_full256k_passed':True,
              'qsa_gpu_tested':False,'adopted':False,'manifests':manifests}
p = base/'uniform-full256k-and-qsa-source-v01402-archive-validation-v1.json'
assert not p.exists()
p.write_text(json.dumps(validation,indent=2)+'\n')
