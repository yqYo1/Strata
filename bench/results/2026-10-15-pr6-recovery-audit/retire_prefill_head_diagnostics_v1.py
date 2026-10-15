#!/usr/bin/env python3
"""Close successful head-diagnostic trace consumers under the measurement lock."""
import collections
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    target = A/'prefill-head-diagnostics-retention-v1.json'
    compact = A/'prefill-head-diagnostics-compact-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not target.exists() and not compact.exists(), 'fresh manifest')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        jobs = [
            ('prefill-head-component-v1.json', 'main-prefill-head-component-root-v1', 'actual-head-component.stderr'),
            ('prefill-head-lifecycle-c4096-v1.json', 'main-prefill-head-lifecycle-c4096-root-v1', 'engine.stderr'),
            ('prefill-head-lifecycle-c8192-r0-v1.json', 'main-prefill-head-lifecycle-c8192-r0-root-v1', 'engine.stderr')]
        evidence = {}; delete = []; retain = []
        for name, rootname, logname in jobs:
            receipt = A/name; r = json.loads(receipt.read_text()); root = B/rootname
            require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'] and
                    r['boot_unchanged'] and not r['driver_dump_exists'], 'closed successful diagnostic')
            for c in r['commands']:
                require(module.closed(c), 'owned command not closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owner still present')
            path = root/'owned'/logname
            text = path.read_text()
            errors = re.findall(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live', text)
            require(not errors, 'successful trace contains an error')
            item = dict(receipt_sha256=sha(receipt), original_log_sha256=sha(path),
                        original_log_bytes=path.stat().st_size, native_error_matches=errors)
            if logname == 'engine.stderr':
                require(sha(path) == r['native_log_sha256'], 'exact native log')
                require(all(line in text.splitlines() for line in r['head_lifecycle']['diagnostic_lines']),
                        'closed head evidence missing')
                item['selected_lines'] = [line for line in text.splitlines() if line.startswith(
                    ('strata generate:', 'strata prefill:', 'strata prefill heads:', 'strata mtp:',
                     'strata verify:', 'strata serve:', 'strata trace: read ', 'strata trace: prompt done '))
                    or ('free after ' in line or 'free with everything loaded' in line)]
                load = re.findall(r'strata trace: layer-major layer (\d+) of 48: loading (\d+) experts', text)
                done = re.findall(r'strata trace: layer-major layer (\d+) of 48: completed (\d+) chunks', text)
                require(len(load) == len(done) == 96, 'two complete 48-layer scopes')
                require([int(x[0]) for x in load] == list(range(48))*2 and
                        [int(x[0]) for x in done] == list(range(48))*2 and
                        all(x[1] == '512' for x in load), 'layer order/coverage')
                item['layer_order'] = [int(x[0]) for x in done]
                item['chunks_completed_per_layer_pass'] = [int(x[1]) for x in done]
                for j in r['request_journals']:
                    jp = root/j['path']; require(sha(jp) == j['sha256'], 'request evidence identity')
                    retain.append(dict(path=str(jp), bytes=jp.stat().st_size, sha256=sha(jp),
                        byte_budget=64 << 10, consumer='Actual officialbench request-journal parser fixture and cohort/body-hash audit',
                        next_review='When the benchmark evidence client/controller or current reference cohort is replaced.'))
            else:
                result_rows = re.findall(r'<--- (ur\w+)\([^\n]*\) -> (UR_RESULT_\w+);', text)
                require(result_rows and all(v == 'UR_RESULT_SUCCESS' for _, v in result_rows), 'complete success result classifications')
                item['UR_result_counts'] = {api:dict(collections.Counter(v for a, v in result_rows if a == api))
                    for api in sorted({a for a, _ in result_rows})}
                item['allocation_transfer_retirement_results'] = [line for line in text.splitlines() if
                    re.search(r'<--- ur(?:USMDeviceAlloc|USMHostAlloc|USMFree|EnqueueUSMMemcpy|QueueFinish)\(', line)]
                item['review'] = 'Warm full-engine C4096/C8192 admission now measured directly; no static addition of isolated free-counter deltas. The64MiB instantaneous counter difference remains unattributed, not a proven application leak or permanent capacity gain.'
            evidence[name] = item
            st = path.stat()
            delete.append(dict(path=str(path), bytes=st.st_size, allocated_bytes=st.st_blocks*512,
                sha256=sha(path), inode=st.st_ino, device=st.st_dev, deleted=False,
                reason='Successful API/polling/chunk trace consumer closed; compact exact outcomes, useful memory/API results and complete layer order replace verbose history.'))
            for c in r['commands']:
                if c['label'] == 'journal-final':
                    jp = root/'owned'/'journal-final.stdout'; h = c['logs']['journal-final.stdout']
                    require(sha(jp) == h['sha256'], 'journal identity')
                    st = jp.stat()
                    delete.append(dict(path=str(jp), bytes=st.st_size, allocated_bytes=st.st_blocks*512,
                        sha256=sha(jp), inode=st.st_ino, device=st.st_dev, deleted=False,
                        reason='Closed receipt retains exact cursor command/boundaries and zero complete fault-filter matches; full unrelated journal window is redundant.'))
        compact.write_text(json.dumps(dict(passed=True, GPU_executed=False, diagnostics=evidence), indent=2)+'\n')
        r = dict(passed=False, complete=False, active=True, GPU_executed=False,
                 recipe_sha256=sha(Path(__file__)), replacement=str(compact), replacement_sha256=sha(compact),
                 deleted=delete, retained=retain, original_status_and_receipt_bytes_unchanged=True)
        def persist():
            target.write_text(json.dumps(r, indent=2)+'\n')
        persist()
        try:
            for item in delete:
                path = Path(item['path']); st = path.stat()
                require((st.st_size, st.st_dev, st.st_ino) == (item['bytes'], item['device'], item['inode']) and
                        sha(path) == item['sha256'], 'log identity changed')
                path.unlink(); item['deleted'] = True; persist()
            r.update(passed=True, complete=True, logical_bytes_retired=sum(x['bytes'] for x in delete),
                     allocated_bytes_retired=sum(x['allocated_bytes'] for x in delete))
        finally:
            r['active'] = False; persist()
        print(json.dumps(dict(passed=r['passed'], logical_bytes_retired=r.get('logical_bytes_retired'))))


if __name__ == '__main__':
    main()
