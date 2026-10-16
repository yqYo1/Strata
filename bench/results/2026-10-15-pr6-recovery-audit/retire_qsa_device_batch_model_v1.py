#!/usr/bin/env python3
"""Retire completed QSA batch captures after exact full-byte selected comparison."""
import collections
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pair_path=A/'qsa-device-batch-model-comparison-v1.json';pair=json.loads(pair_path.read_text())
        require(pair['passed'] and pair['complete'] and pair['sampled_layer_residuals_bitwise_equal'],'actual exact comparison')
        output=A/'qsa-device-batch-model-compact-v1.json'
        target=A/'qsa-device-batch-model-retention-v1.json'
        require(not output.exists() and not target.exists(),'fresh retention')
        owner=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        items=[];evidence=[]
        for name,identity in pair['references'].items():
            receipt=A/name;r=json.loads(receipt.read_text())
            require(sha(receipt)==identity and r['passed'] and r['complete'] and not r['active'],'immutable closed receipt')
            for c in r['commands']:
                require(owner.closed(c),'closed command')
                for old in c['owners']:
                    now=owner.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'retired owners')
            boundary=r['boundary'];raw=Path(boundary['raw_path']);text=Path(boundary['text_path'])
            require(sha(raw)==boundary['raw_sha256'] and raw.stat().st_size==boundary['raw_bytes'],'raw identity')
            require(sha(text)==boundary['text_sha256'],'boundary log identity')
            lines=text.read_text().splitlines();counts=collections.Counter()
            for line in lines:
                m=re.match('phase=(\\w+) ',line)
                if m:counts[m[1]]+=1
            require(counts['output_before_store']==768 and counts['stored']==counts['next_input']==752 and
                lines[-1]=='status=complete boundary_memcmp=equal','exact phase coverage')
            native=raw.parent/'owned/engine.stderr'
            require(sha(native)==r['native_log_sha256'],'native log identity')
            trace=native.read_text().splitlines()
            layer_done=[list(map(int,m.groups())) for line in trace if
                (m:=re.search(r'layer-major layer (\d+) of 48: completed (\d+) chunks',line))]
            require(layer_done==[[l,10] for l in range(48)],'all layer/chunk completions')
            keep=[line for line in trace if line.startswith(('strata prefill heads:',
                'strata mtp: prompt expert phase=','strata prefill: layer-major candidate,',
                'strata prefill: FP16 MoE compact','strata serve: prompt ')) or 'recapture' in line]
            evidence.append(dict(receipt=name,receipt_sha256=identity,qsa_batch_arm=r['qsa_batch_arm'],
                boundary_shape=lines[0],boundary_completion=lines[-1],phase_counts=dict(counts),
                layer_completed_chunks=layer_done,raw_sha256=sha(raw),raw_bytes=raw.stat().st_size,
                native_selected_lines=keep,native_sha256=sha(native),boundary_text_sha256=sha(text)))
            for path in (raw,text,native):
                items.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),receipt=name,
                    reason='Complete cross-run exact selected comparison and same-run boundary checks are recorded; no remaining consumer needs successful full raw/trace.'))
        compact=dict(passed=True,complete=True,GPU_executed=False,comparison_sha256=sha(pair_path),
            recipe_sha256=sha(Path(__file__)),runs=evidence)
        with output.open('x') as out:json.dump(compact,out,indent=2);out.write('\n')
        require(json.loads(output.read_text())==compact,'compact roundtrip')
        manifest=dict(passed=False,complete=False,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
            comparison_sha256=sha(pair_path),replacement=output.name,replacement_sha256=sha(output),deleted=[])
        with target.open('x') as out:json.dump(manifest,out,indent=2);out.write('\n')
        for item in items:
            path=Path(item['path']);require(sha(path)==item['sha256'],'identity before unlink')
            path.unlink();manifest['deleted'].append(item)
            target.write_text(json.dumps(manifest,indent=2)+'\n')
        manifest.update(passed=True,complete=True,removed_bytes=sum(i['bytes'] for i in items),finished_utc=owner.utc())
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(passed=True,removed_bytes=manifest['removed_bytes'])))


if __name__=='__main__':main()
