"""Root-only closed-run retirement. No GPU/service operations or test changes."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
Q = B/'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r6'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W/'bench/results/2026-10-10-parallel-round35'
RECORD_SHA = 'f606f5a1f2d00dba240213a7feeb3cc84c9da4f1487075d1e97c336a558e4c89'
SUBSET_SHA = '95ae2a0a7e61f2240f6138f21fdcb35c1a70ec54300b22a7b99fd6b92873b4a7'
RAW_SHA = '3309c390f8e5f047321ca84bcdc6d182e9c13ca1b8db178c79a88b108e13ab3f'
FAULT = re.compile(r'page fault|fault response|timestamp stuck|wedg|engine reset|gt reset|guc.*(?:timeout|timed out|failed)|memory.*cat|device.*lost', re.I)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_nlink == 1
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


def verified_state(path, capture):
    before = identity(path)
    assert before[2] == capture['bytes'] and len(capture['parts']) == 66
    whole = hashlib.sha256()
    with path.open('rb') as stream:
        for index, part in enumerate(capture['parts']):
            assert part['index'] == index and stream.tell() == part['offset']-8
            header = stream.read(8)
            assert len(header) == 8 and struct.unpack('<Q', header)[0] == part['bytes']
            whole.update(header)
            section = hashlib.sha256()
            left = part['bytes']
            while left:
                data = stream.read(min(left, 8*1024**2))
                assert data
                left -= len(data)
                section.update(data)
                whole.update(data)
            assert section.hexdigest() == part['sha256'], 'saved raw section changed'
        assert stream.tell() == before[2] and not stream.read(1)
    assert identity(path) == before
    return whole.hexdigest()


def main():
    assert __debug__
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert digest(Q/'record.json') == RECORD_SHA
        d = json.loads((Q/'record.json').read_text())
        for field in ('completed','healthy','math_gate_passed','pair_gate_passed','pair_shape_gate_passed','histogram_gate_passed','protocol_boundary_gate_passed','text_budget_gate_passed'):
            assert d[field] is True
        assert d['active'] is False and d['exit_code'] == 0 and d['exit_signal'] is None
        assert d['cleanup'] == dict(forced=False, inferior_survived=False, gdb_survived=False)
        assert not d['new_fault_messages'] and not d['preflight_fault_messages']
        assert d['adopted'] is False and d['performance_eligible'] is False and d['full_lifecycle_passed'] is False
        assert d['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        owners=[]
        for role in ('inferior','debugger'):
            owner=d[role];p=Path(f"/proc/{owner['pid']}/stat")
            actual=None if not p.exists() else int(p.read_text().rsplit(')',1)[1].split()[19])
            assert actual != owner['start_ticks'], 'owned run process still exists'
            owners.append(dict(role=role, recorded=owner, actual_start_ticks=actual, owned_absent=True))
        assert digest(A/Q.name/'record.json') == RECORD_SHA
        git_path=str((A/Q.name/'record.json').relative_to(W))
        committed=subprocess.check_output(['git','show','0098e2c3:'+git_path],cwd=W)
        assert hashlib.sha256(committed).hexdigest() == RECORD_SHA
        subset_path=B/'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json'
        assert digest(subset_path) == SUBSET_SHA
        refs=json.loads(subset_path.read_text())['requests']
        assert len(refs) == len(d['requests']) == 4
        assert digest(Q/'debugger/inferior.stderr') == RAW_SHA
        selections=[]
        for request, reference in zip(d['requests'], refs):
            assert request['math_gate_passed'] is True
            assert all(request['head_and_live_state_comparison'].values())
            assert not request['live_prefill_comparison']['different_live_parts']
            assert len(request['live_prefill_comparison']['part_evidence']) == 66
            for key in ('first_head','prefill_state'):
                capture=request[key];path=Path(capture['file']);replacement=Path(reference[key]['file'])
                assert path.parent == Q and identity(replacement)[2] == (reference[key].get('bytes',reference[key].get('floats',0)*4))
                before=identity(path)
                sha=verified_state(path,capture) if key=='prefill_state' else digest(path)
                if key=='first_head':assert sha == capture['sha256'] == reference[key]['sha256']
                selections.append(dict(path=str(path),identity=list(before),bytes=before[2],sha256=sha,
                                       reported_file_allocation_bytes=path.stat().st_blocks*512,
                                       retained_replacement=str(replacement),replacement_receipt=str(subset_path),
                                       reason='Successful duplicate candidate capture; exact head/all66 live comparisons are committed; canonical comparison/RESTORE fixtures retained; no current consumer needs candidate bytes',removed=False))
        kept={Q/'record.json',Q/'debugger/inferior.stderr',Q/'probes/health.stdout',Q/'probes/health-environment.json'}
        selected={Path(x['path']) for x in selections}
        kernel_checks=[]
        for name in ('kernel-gap','kernel-after'):
            path=Q/'probes'/f'{name}.stdout';rows=[json.loads(line) for line in path.read_text().splitlines() if line]
            faults=[]
            for row in rows:
                message=row.get('MESSAGE','')
                device=('0000:05:00.0' in message or re.search(r'\bxe\b',message))
                if (device and FAULT.search(message)) or ('strata' in message and 'segfault' in message):
                    faults.append(message)
            assert not faults
            kernel_checks.append(dict(path=str(path),rows=len(rows),faults=faults,filter=FAULT.pattern,
                                      device_filter='0000:05:00.0 or word xe; additionally Strata segfault',kernel_command_exit=0))
        for path in sorted(Q.rglob('*')):
            if not path.is_file() or path in selected or path in kept:
                continue
            assert path.suffix != '.bin' and path.resolve().is_relative_to(Q.resolve())
            before=identity(path);sha=digest(path);assert identity(path)==before
            archived=A/Q.name/path.relative_to(Q)
            replacement=str(archived) if archived.exists() and digest(archived)==sha else str(A/Q.name/'record.json')
            selections.append(dict(path=str(path),identity=list(before),bytes=before[2],sha256=sha,
                                   reported_file_allocation_bytes=path.stat().st_blocks*512,
                                   retained_replacement=replacement,
                                   reason='Closed successful repeated trace/transport/probe or derived prompt copy; command/environment/individual math/fault/process evidence committed; no replay/RESTORE consumer',removed=False))
        manifest_path=B/'cache-route-r6-artifact-retirement-v1.json'
        assert not manifest_path.exists()
        space=os.statvfs(B)
        manifest=dict(active=True,complete=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      controller_sha256=digest(Path(__file__)),exclusive_lock=str(B/'owned-v0141-measurement.lock'),
                      original_receipt_sha256=RECORD_SHA,committed_evidence='0098e2c3',owners=owners,
                      tensor_consumer_review='No candidate RESTORE/comparison scheduled; canonical four32K inputs and all full256K baseline/unresolved C/D captures untouched',
                      retained_raw_consumer=dict(owner='/root',path=str(Q/'debugger/inferior.stderr'),bytes=1389517,sha256=RAW_SHA,
                                                 consumer='Pinned static-count replay and prefix-quota trace scoring',next_review='Close or supersede static-prefix data selection/holdout question'),
                      kernel_evidence=kernel_checks,available_before_bytes=space.f_bavail*space.f_frsize,
                      entries=selections,logical_bytes=sum(x['bytes'] for x in selections),
                      reported_file_allocation_bytes=sum(x['reported_file_allocation_bytes'] for x in selections),
                      physical_unique_reclaimed_bytes=None,
                      limitations='Per-file allocated-byte sum is not unique physical extents; available-space delta may reflect other activity/snapshots. Original result status stays unchanged.')
        def save():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        save();start=time.monotonic()
        for entry in selections:
            path=Path(entry['path']);assert identity(path)==tuple(entry['identity']);path.unlink();entry['removed']=True;save()
        assert digest(Q/'record.json')==RECORD_SHA and digest(Q/'debugger/inferior.stderr')==RAW_SHA
        for request in refs:
            for key in ('first_head','prefill_state'):assert Path(request[key]['file']).is_file()
        space=os.statvfs(B);manifest.update(active=False,complete=True,elapsed_unlink_seconds=time.monotonic()-start,
                                         available_after_bytes=space.f_bavail*space.f_frsize,
                                         finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        manifest['available_delta_bytes']=manifest['available_after_bytes']-manifest['available_before_bytes'];save()
        print(json.dumps(dict(manifest=str(manifest_path),sha256=digest(manifest_path),removed=len(selections),logical_bytes=manifest['logical_bytes'],reported_file_allocation_bytes=manifest['reported_file_allocation_bytes'],available_delta_bytes=manifest['available_delta_bytes'],original_receipt_unchanged=True)))


if __name__ == '__main__':main()
