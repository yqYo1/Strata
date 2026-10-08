"""Freeze the failed eager-full comparison, private cache build and reference audit."""
from pathlib import Path
import array
import datetime
import hashlib
import json
import math
import shutil
import sys

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis'
# Reuse the already reviewed bounded-log packaging functions, not its main.
helpers = (base/'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads(', 1)[0]
exec(compile(helpers, str(base/'archive_lazy_full_and_dequant.py'), 'exec'))

full = json.loads((base/'full-context-eager-verifier-serve/record.json').read_text())
health = json.loads((base/'post-full-eager-verifier-health/record.json').read_text())
lazy = json.loads((base/'full-context-verifier-lazy-restore-serve/record.json').read_text())
build = json.loads((base/'expanded-fp16-cache-build/record.json').read_text())
assert not full['active'] and not full['completed'] and not full['healthy']
assert full['error'] == 'AssertionError()' and not full['new_fault_messages']
assert not full['cleanup']['inferior_survived'] and not full['cleanup']['gdb_survived']
assert len(full['requests']) == 1 and health['healthy'] and health['boot_id'] == full['boot_id']
assert health['started_utc'] > full['finished_utc']
for key in ('inferior', 'debugger'):
    old = full[key]; now = process_identity(old['pid'])
    assert not now or now['start_ticks'] != old['start_ticks']
assert build['passed'] and build['production_inputs_unchanged']
payload = (base/'full-context-eager-verifier-serve/first-head.bin').read_bytes()
reference = (base/'full-context-verifier-lazy-restore-serve/fills-context.head.bin').read_bytes()
x = array.array('f'); x.frombytes(payload)
y = array.array('f'); y.frombytes(reference)
assert len(x) == len(y) == 248320 and all(map(math.isfinite, x)) and all(map(math.isfinite, y))
first = full['requests'][0]
assert first['ids'] == lazy['requests'][0]['ids'] == [40, 3172, 1151, 539]
assert first['verify_windows'] == [[262139, 1], [262140, 4], [262141, 3]]
assert first['draft_releases'] == first['draft_restores'] == 1
assert digest(base/'full-context-eager-verifier-serve/first-head.bin') == first['head']['sha256']

eager = parent/'verifier-eager-full-20261007'
cache = parent/'expanded-fp16-cache-prepared-20261007'
study = parent/'b70-offload-reference-20261007'
for directory in (eager, cache, study):
    assert not directory.exists()
    directory.mkdir(mode=0o755)

for name in ('run_full_eager_verifier_serve.py', 'probe_after_full_eager_verifier.py',
             'archive_eager_cache_reference.py'):
    copy(base/name, eager, Path('sources-used')/name)
save_folder(base/'full-context-eager-verifier-serve', eager, 'full-run')
save_folder(base/'post-full-eager-verifier-health', eager, 'health')
checkpoints = []
with (base/'full-context-eager-verifier-serve/debugger/inferior.stderr').open('rb') as stream:
    for line in stream:
        if line.startswith(b'strata prefill memory:'):
            checkpoints.append(line.decode().strip())
(eager/'memory-checkpoints.txt').write_text('\n'.join(checkpoints)+'\n')
assessment = dict(scope='Original eager-window execution fills262144cells; strict LP/full-head comparison fails before repeated-capacity cases. No GPU stall observed, no complete suite or clean speed proof.',
                  first_request=first, whole_head_changed_floats=sum(a != b for a,b in zip(x,y)),
                  whole_head_max_absolute_difference=max(abs(a-b) for a,b in zip(x,y)),
                  reference_head_sha256=hashlib.sha256(reference).hexdigest(),
                  lazy_vs_eager_cause='Unresolved; same first output IDs do not establish arithmetic/state parity.',
                  full_capacity_suite_passed=False, adopted=False, cleanup=full['cleanup'],
                  new_fault_messages=full['new_fault_messages'], post_cleanup_health_passed=True,
                  next='Separate capacity observation from parity gate; reproduce graph/eager difference with bounded smaller contexts, preserve original phase waits.')
(eager/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
(eager/'README.md').write_text('''# Eager verifier full-context comparison on 2026-10-07

On the same B570 10 GiB / Ryzen 5600X / 128 GiB host and boot, the unchanged
3f3e executable with the existing eager-window option completes a normal-MTP
request with 262,140 input and four output tokens. Windows [262139,1],
[262140,4], [262141,3] execute through the last KV cell 262143, and one
verified MTP release/restore pair completes. All printed logprobs and all
248,320 head floats are finite.

The four IDs [40,3172,1151,539] match the earlier lazy candidate, but all
248,320 first-head floats differ, with maximum absolute difference
0.412168025970459. The eager head hash is 26eb4436…992b720a, versus
ce388272…86f7516 in the lazy run. All four logprob lines differ. Their
same-setting 2K comparison had matched, which did not establish full-context
parity. The source of the full-context difference remains unresolved.

The strict comparison stops the controller before the second request. It
removes both owned processes; no new xe fault is observed. The following
logged H2D/kernel/D2H probe passes all three 16,384-word rounds without a
reset. This termination is a comparison failure, not an observed GPU hang.
The repeated-prefill, clipped-tail, overflow-refusal and later-valid gates
remain incomplete. The original eager path is not adopted as a full-capacity
fix. Diagnostic durations are not clean throughput.

[Assessment](assessment.json), [terminal record](full-run/record.json),
[memory checkpoints](memory-checkpoints.txt), health receipt, source/controller
hashes and bounded log tails preserve this result. Full heads and large logs
remain private with recorded SHA-256 hashes.
''')

for name in ('prepare_expanded_fp16_cache.py', 'build_expanded_fp16_cache.py',
             'expanded_fp16_layer_cache.hpp', 'archive_eager_cache_reference.py'):
    copy(base/name, cache, Path('sources-used')/name)
save_folder(base/'expanded-fp16-cache-prepared', cache, 'prepared')
save_folder(base/'expanded-fp16-cache-build', cache, 'build')
(cache/'README.md').write_text('''# Optional expanded-FP16 layer cache: CPU preparation only

The 2K unitrace run recorded 50,030 GU and 50,030 down dequant launches,
4.62848 seconds total, about 60% of its kernel duration. This private
prototype reuses unchanged FP16 expert outputs across chunks within one
layer-major request. It does not change quantization formulas, wrappers,
GEMM dimensions, activation operands, phase waits or decode callbacks.

STRATA_PREFILL_FP16_CACHE_EXPERTS defaults to zero. Each admitted expert
needs 9,830,400 bytes. The candidate limits admission by free VRAM minus
256 MiB and the device maximum allocation size. No available budget, a
single chunk, unsupported dimensions or failed allocation uses the original
scratch path. At the observed full-context free budget of about 51 MiB,
the arithmetic budget would select zero slots; GPU fallback is untested.

The allocation, dequant producers and GEMM consumers use the same in-order
queue. An expert is marked ready only after both producers are enqueued;
layer changes invalidate all tags. The owner waits before freeing USM with
the allocation queue/context, and frees it before main/MTP restoration.
On early return, the existing Restore guard waits and restores the old
cache pointer before the new owner is destroyed. If a wait fails, its
destructor retains memory instead of freeing potentially in-flight USM.
These are source-review invariants, not execution proofs. See the
[SYCL queue reference](https://github.khronos.org/SYCL_Reference/iface/queue.html)
and [USM lifetime reference](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html).

The [offline build](build/record.json) passed in 28.279 seconds. Exactly
one prefill archive member was replaced; other archive members and all
production link inputs remained byte-identical. The private binary SHA-256
is 9d9085d4…d658110. Production source and executable were not changed.
No GPU model run, output parity, clean speed measurement, repeated restore
or 256K fallback gate has run. The first GPU check must capture Level Zero/
UR logs and compare cache-zero and positive-capacity multi-chunk outputs.
''')

repo = base/'qwen38-b70-reference'
import subprocess
commit = subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
assert commit == '557f80fe0f4235a76d4cf835f89e108d599b5e94'
paths = ['README.md','plugin/exl3xpu/nvtier.py','research/18-b70-ingest-ceiling.md',
         'research/19-strata-lessons.md','experimental/n111-victim-ring/STATUS_N111.md',
         'kernels/n107-hc/hc_xpu.py','kernels/n107-hc/README.md','LICENSE']
receipt = {'repository':'https://github.com/0xSero/qwen38-flash-next-b70-offload/',
           'commit':commit, 'read_only':True, 'foreign_code_executed':False,
           'reviewed_files':[{'path':name,'bytes':(repo/name).stat().st_size,'sha256':digest(repo/name)} for name in paths]}
(study/'reference.json').write_text(json.dumps(receipt,indent=2)+'\n')
(study/'README.md').write_text('''# B70 offload reference inspected on 2026-10-07

User-supplied [qwen38-flash-next-b70-offload](https://github.com/0xSero/qwen38-flash-next-b70-offload/)
was read at commit 557f80fe0f4235a76d4cf835f89e108d599b5e94. [File hashes](reference.json)
pin the source examined; none of its code or scripts was executed.

Its platform is B70 32 GB, PCIe Gen4 x16, 32 GB host RAM and an NVMe RAID.
It uses EXL3 weights and SGLang, whereas the current B570 run uses IQ3_S,
10 GiB VRAM and measured explicit H2D about 6.16 GB/s. Its speeds and
kernel choices cannot serve as measurements for this host.

The most relevant implementation is plugin/exl3xpu/nvtier.py:529-623.
It allocates two full-layer VRAM buffers and a configurable ring of RAM
buffers, reads experts ahead with a worker pool, coalesces consecutive
missing expert IDs into one DMA call, and orders reuse with H2D and compute
events. A RAM buffer waits for its previous copy before being rewritten;
a VRAM buffer waits for its previous compute before receiving new data;
compute waits for the new copy. This event ordering is useful for a bounded
pipeline on B570. Our current layer-major loop waits for compute, loads all
512 compressed experts and waits for copy before running the layer.

Full-layer double buffering requires another approximately 1.36 GB on our
native pack, which the observed full-256K free budget cannot provide.
Keep the single-buffer fallback, or examine RAM-only read-ahead and partial
GPU staging. Record peak VRAM and actual transfer/compute overlap rather
than inferring overlap from the presence of two queues. The reference's
research/18-b70-ingest-ceiling.md reports serialization within one process
and overlap across processes; that report is a hypothesis to test on this
runtime, not proof of a driver fault on this host.

Two other useful directions are post-GEMM hyper-connection fusion, with
explicit rounding boundaries, and event-ordered decode bookkeeping instead
of a device-wide wait per step. The reference already adapts Strata attention
and GDN code, so compare the actual source and existing local profiles to
avoid porting an equivalent implementation back as an assumed improvement.

Do not treat its no-victim-write-back figures as exact inference: README
documents masked expert picks. The experimental victim ring has standalone
checks but no server matrix; when full it can drop victims and increase
masking. The reported direct-write-back race (a pending host page punch
versus GPU residency publication) reinforces the invariant that residency
must be published only after data lands and storage is no longer being
reclaimed. Our optimization must retain every selected expert and have a
lossless fallback under pressure.

Next measurements: unchanged-result dequant reuse, bounded RAM read-ahead/
coalesced copies, and an isolated logged copy-compute overlap probe. Keep
original phase waits until the timing-dependent parity issue is resolved;
follow short full-head comparisons with repeated 262144-cell capacity gates.
''')
reports = [manifest(directory) for directory in (eager, cache, study)]
(base/'eager-cache-reference-archive.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
