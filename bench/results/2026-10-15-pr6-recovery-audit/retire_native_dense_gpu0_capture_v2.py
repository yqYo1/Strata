#!/usr/bin/env python3
"""Retire the completed GPU-prefix0 comparison capture; retain32768 for Direct1."""
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
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pair_path = A/'native-dense-bank-gpu-rows-comparison-v1.json'
        pair = json.loads(pair_path.read_text())
        require(pair['passed'] and pair['complete'] and pair['sampled_FP32_values_equal']
            and pair['sampled_FP32_values_finite'], 'completed selected-row comparison')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for name,h in pair['references'].items():
            path = A/name
            require(sha(path) == h, 'original comparison input')
            r = json.loads(path.read_text())
            require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults']
                and r['boot_unchanged'] and not r['driver_dump_exists'], 'healthy closed model')
            for c in r['commands']:
                require(module.closed(c), 'owned command closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned identity retired')
        source = A/'native-dense-bank-model-1-v1.json'
        r = json.loads(source.read_text())
        root = B/'main-native-dense-bank-model-1-root-v1'
        native = Path(r['config']['log'])
        text = native.read_text()
        completed = [list(map(int,m)) for m in re.findall(r'layer-major layer (\d+) of 48: completed (\d+) chunks', text)]
        require(completed == [[n,10] for n in range(48)], 'all original layer/chunk completions')
        compact = A/'native-dense-gpu0-model-compact-v2.json'
        target = A/'native-dense-gpu0-model-retention-v2.json'
        require(not compact.exists() and not target.exists(), 'fresh retention')
        evidence = dict(passed=True, complete=True, GPU_executed=False, recipe_sha256=sha(Path(__file__)),
            references={source.name:sha(source),pair_path.name:sha(pair_path)}, completed_layer_chunks=completed,
            retained_native_lines=[line for line in text.splitlines() if line.startswith(('strata prefill dense:',
                'strata mtp: prompt expert phase=', 'strata prefill: layer-major candidate,','strata serve: prompt '))],
            boundary=r['boundary'], dense_lifecycle=r['dense_lifecycle'], fresh_faults=r['fresh_faults'])
        with compact.open('x') as output:
            json.dump(evidence, output, indent=2); output.write('\n')
        require(json.loads(compact.read_text()) == evidence, 'verified compact evidence')
        files = [(Path(r['boundary']['raw_path']),r['boundary']['raw_sha256']),
            (Path(r['boundary']['text_path']),r['boundary']['text_sha256']), (native,r['native_log_sha256'])]
        for c in r['commands']:
            files.extend((root/'owned'/name,v['sha256']) for name,v in c['logs'].items())
        files.extend((root/v['path'],v['sha256']) for v in r['request_journals'])
        duplicate = root/'record.json'
        require(json.loads(duplicate.read_text()) == r, 'duplicate closed record')
        files.append((duplicate,sha(duplicate)))
        deleted = []
        for path,h in files:
            require(path.is_file() and not path.is_symlink() and sha(path) == h, 'closed file identity')
            deleted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=h,
                reason='GPU0-vs32768 selected768 rows exactly equal/finite and same-run752 copies checked; no remaining consumer for the GPU0 raw or complete diagnostic/protocol history. Individual original receipt/status and compact evidence retained.'))
        manifest = dict(passed=False, complete=False, GPU_executed=False, recipe_sha256=sha(Path(__file__)),
            replacement=compact.name, replacement_sha256=sha(compact), deleted=deleted,
            retained_32768_capture_consumer='run_native_dense_bank_direct1_v1.py: same-native FP32GPU32768 Direct0-vs1 selected-row comparison; paired64MiB raw cap.')
        with target.open('x') as output:
            json.dump(manifest, output, indent=2); output.write('\n')
        for item in deleted:
            path = Path(item['path'])
            require(sha(path) == item['sha256'], 'identity before unlink')
            path.unlink()
        manifest.update(passed=True, complete=True, removed_bytes=sum(x['bytes'] for x in deleted), finished_utc=module.utc())
        target.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(passed=True, removed_bytes=manifest['removed_bytes'], retired_files=len(deleted))), flush=True)


if __name__ == '__main__':
    main()
