#!/usr/bin/env python3
"""CPU-only same-slot32-ID reuse census; independently authored, repo LICENSE."""
from array import array
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_owner as owner_module
import measurement_guards as guards
import qsa_selection_capture as capture

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B/'qsa-aligned-tiles-v1'
RESULT = P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'qsa_aligned_tiles'
EXPECTED = capture.headers(73)+capture.headers(6491, 32768)
SPANS = (4, 16, 64)
require = guards.require


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value): path.write_text(json.dumps(value, indent=2)+'\n')
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def git(*args): return subprocess.check_output(['git', *args], cwd=P, timeout=60)


def pins(revision):
    values = {}
    for name in ('tools/sycl/qsa_tile_reuse.py', 'tools/sycl/qsa_selection_capture.py',
                 'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py'):
        raw = git('show', revision+':'+name)
        require(raw == (P/name).read_bytes(), 'commit source first: '+name)
        values[name] = hashlib.sha256(raw).hexdigest()
    return values


def save(row):
    d = json.loads(RESULT.read_text()); d[KEY] = row; write(RESULT, d); write(ROOT/'record.json', row)


def admission(row):
    require(pins(row['source_commit']) == row['inputs'], 'host source changed')
    base = json.loads(RESULT.read_text())['qsa_exact_lists']
    require(canonical(base) == row['capture_result_sha256'], 'capture qualification changed')
    require(digest(row['capture']['path']) == row['capture']['sha256'] and
        capture.identity(Path(row['capture']['path']).stat()) == tuple(row['file_identity']),
        'closed capture changed')
    require(digest(sys.executable) == row['python']['sha256'], 'host Python changed')


def prepare():
    raw = RESULT.read_bytes(); d = json.loads(raw)
    require(raw == git('show', 'HEAD:'+str(RESULT.relative_to(P))) and KEY not in d and not ROOT.exists(),
        'fresh committed host census only')
    base = d['qsa_exact_lists']
    require(base['complete'] and base['passed'] and base['ownership_closed'] and
        base['census']['same_list_pairs'] == 0 and not base['model_adopted'], 'closed source-bound full-list census required')
    source = git('rev-parse', 'HEAD').decode().strip()
    source_pins = pins(source)
    host_controls()
    ROOT.mkdir(mode=0o700); (ROOT/'tmp').mkdir()
    row = dict(stage='prepared', source_commit=source, inputs=source_pins,
        capture_result_sha256=canonical(base), capture=base['capture'],
        file_identity=list(capture.identity(Path(base['capture']['path']).stat())),
        python=dict(path=sys.executable, sha256=digest(sys.executable)),
        command=[sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), 'worker'],
        environment=dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
            PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT/'tmp')),
        question='Within each captured layer and consecutive4/16/64query windows, count exact initialized32-cell sequences at the same online-attention tile index. Equality includes partial-tail length. No query/head/layer or iteration reordering is performed.',
        acceptance='Validate all24source/workload headers and causal ordered initialized IDs again,exact raw hash/identity before/after,source/Python pins and owned normalexit0. Count each query/tile once per span;conserve logical selected cells. Byte keys check exact equality despite dictionary hash collisions.',
        decision_rule='If all layers/spans have<=1% ideal selected-ID load elision,drop this exact-aligned32tile-sharing direction for this sample. 1..10% defers it;>=10% only nominates further source/scheduling/resource comparison,not implementation eligibility. Shared instances alone do not count saved reads.',
        limits='CPU-only logical topology and an ideal read-once-per-class bound. It is not physical VRAM traffic,cache hits,saturation,critical-path or speed evidence. Current32-cell online order and existing KV-head/layer isolation remain. No inference/GPU/math or model qualification.',
        budgets=dict(wall=180, cpu=120, rss_cap=1<<30, text_cap=1<<20, file_cap=4<<20, total_cap=8<<20),
        retention='Input646213056B capture held only through this immediate host census. Retain compact per-layer samples,logical bound,source/recipe/hash/decision;retire input capture and generated output/one-use tools after closure. No new native engine generation.',
        host_controls=dict(cases=3, passed=True),
        passed=False, complete=False, ownership_closed=True, GPU_executed=False, model_adopted=False)
    admission(row); save(row); print('Prepared finite CPU-only aligned-tile census;commit plan before run.', flush=True)


def tiles(ids, rows, cap, span):
    classes = {}; totals = dict(windows=0, query_tiles=0, selected_ID_reads=0,
        distinct_tile_classes=0, repeated_tile_instances=0, duplicate_tile_instances=0,
        ideal_elided_ID_reads=0, maximum_class=0)
    for row in range(rows):
        for first in range(0, cap, 32):
            value = ids[row*cap+first:row*cap+min(first+32, cap)].tobytes()
            key = first, value
            classes[key] = classes.get(key, 0)+1
        if (row+1) % span == 0 or row+1 == rows:
            totals['windows'] += 1
            for (_, value), count in classes.items():
                width = len(value)//4
                totals['query_tiles'] += count
                totals['selected_ID_reads'] += count*width
                totals['distinct_tile_classes'] += 1
                totals['repeated_tile_instances'] += count if count > 1 else 0
                totals['duplicate_tile_instances'] += count-1
                totals['ideal_elided_ID_reads'] += (count-1)*width
                totals['maximum_class'] = max(totals['maximum_class'], count)
            classes.clear()
    require(totals['query_tiles'] == rows*((cap+31)//32) and
        totals['selected_ID_reads'] == rows*cap, 'tile/cell count does not conserve source work')
    totals['ideal_ID_read_elision_fraction'] = totals['ideal_elided_ID_reads']/totals['selected_ID_reads']
    return totals


def host_controls():
    same = array('i', list(range(33))*2)
    a = tiles(same, 2, 33, 16)
    require(a['query_tiles'] == 4 and a['distinct_tile_classes'] == 2 and
        a['ideal_elided_ID_reads'] == 33 and a['selected_ID_reads'] == 66,
        'partial-tile equality control failed')
    shifted = array('i', list(range(64))+list(range(32, 96)))
    require(tiles(shifted, 2, 64, 16)['duplicate_tile_instances'] == 0,
        'different online tile slots were incorrectly merged')
    tail = tiles(array('i', list(range(33))*3), 3, 33, 2)
    require(tail['windows'] == 2 and tail['query_tiles'] == 6 and
        tail['ideal_elided_ID_reads'] == 33 and tail['selected_ID_reads'] == 99,
        'partial query window or per-window isolation failed')


def worker():
    row = json.loads((ROOT/'record.json').read_text()); admission(row)
    parsed = capture.inspect(row['capture']['path'], EXPECTED)
    require(parsed['sha256'] == row['capture']['sha256'], 'validation hash differs')
    samples = []
    with Path(row['capture']['path']).open('rb') as stream:
        for record in parsed['records'][12:]:
            layer, pos, rows, cap = record['header']; require(pos == 32768 and rows == 6491 and
                record['unused_cap_entries'] == 0, 'only initialized sparse tail records admitted')
            stream.seek(record['offset']+16); raw = stream.read(rows*cap*4)
            require(hashlib.sha256(capture.HEADER.pack(*record['header'])+raw).hexdigest() == record['sha256'],
                'record changed during analysis')
            ids = array('i'); ids.frombytes(raw); del raw
            samples.append(dict(layer=layer, query_rows=rows, spans={str(n):tiles(ids, rows, cap, n) for n in SPANS}))
    admission(row); write(ROOT/'analysis.json', dict(samples=samples, raw_sha256=parsed['sha256']))
    print('All12layer tails analyzed;same-slot32cell classes conserve logical reads.', flush=True)


def run():
    raw = RESULT.read_bytes(); row = json.loads(raw)[KEY]
    require(raw == git('show', 'HEAD:'+str(RESULT.relative_to(P))) and row['stage'] == 'prepared' and
        row == json.loads((ROOT/'record.json').read_text()) and not (ROOT/'execution.json').exists(), 'fresh committed plan only')
    admission(row); owner = None
    def persist(): write(ROOT/'execution.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    owner = owner_module.Owner(ROOT/'owned', persist); row['stage'] = 'running'
    try:
        entry, _, _ = owner.run('tile-census', row['command'], row['environment'], P, **row['budgets'])
        require(owner_module.closed(entry) and entry['exit_code'] == 0, 'host census failed')
        analysis = json.loads((ROOT/'analysis.json').read_text()); admission(row)
        row.update(stage='closed', passed=True, complete=True, analysis=analysis)
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row['ownership_closed'] = owner.active is None and all(owner_module.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed']); persist(); save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    if sys.argv[1:] == ['worker']: worker(); raise SystemExit(0)
    require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run|worker')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: raise SystemExit(run())
