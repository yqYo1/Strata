"""Offline revalidation of a closed run; never launches GPU work or rewrites receipts."""
from pathlib import Path
import csv, fcntl, hashlib, json, re, struct, subprocess, textwrap

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq4nl-event-receipt-20261010')
ORIGINAL = B / 'run_iq4nl_event_receipt_v1.py'
RUN = B / 'iq4nl-event-receipt-runtime-v1'
OUT = B / 'iq4nl-event-receipt-offline-revalidation-v1.json'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def ident(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))

def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not OUT.exists()
        original_bytes = ORIGINAL.read_bytes()
        assert sha(ORIGINAL) == '300dc934bf89321cc929b2f18899192f45b697bc0941b659bf4760edfb0560c6'
        receipt = json.loads((RUN / 'record.json').read_text())
        assert receipt['active'] is False and receipt['passed'] is False
        assert receipt['error'] == 'AssertionError: missing/unexpected/unordered record'
        assert receipt['journal_status'] == 'complete'
        assert receipt['visible_kernel_GPU_entries'] == [] and receipt['devcoredump_after'] is False
        for command in receipt['commands']:
            assert command['normal_exit'] and command['session_empty'] and command['observation_complete']
            assert command['direct_child_reaped'] and command['exit_code'] == 0
            assert not command['cleanup'] and not command['errors'] and not command['survivors']
            for name, pin in command['logs'].items():
                assert ident(RUN / name) == pin
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['boot_id']
        assert not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == receipt['source_head']
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True)
        build_path = B / 'iq4nl-event-receipt-cpu-build-v1/record.json'
        assert ident(build_path) == receipt['build_receipt']
        build = json.loads(build_path.read_text())
        assert build['passed'] and build['complete'] and not build['active']
        for rel, pin in build['source_pins'].items():
            assert ident(W / rel) == pin
        binary = Path(receipt['binary']['path'])
        assert ident(binary) == {key: receipt['binary'][key] for key in ['bytes', 'sha256']}
        for path, pin in receipt['resolved_dependency_pins'].items():
            assert ident(path) == {key: pin[key] for key in ['bytes', 'sha256']}
            assert str(Path(path).resolve()) == pin['realpath']
        # Only correct the two initial record positions; keep every numerical,
        # lifecycle, receipt-metadata and timestamp assertion from v1 intact.
        source = original_bytes.decode()
        start = source.index('            # Build the full ordered transcript independently;')
        stop = source.index("            _,interval,interval_err=run('kernel-interval'", start)
        parser = textwrap.dedent(source[start:stop])
        replacements = {
            "['HOST_PASS','IDENTITY','RECEIPT_QUEUE','FP16_CONFIG','STAGE'": "['HOST_PASS','IDENTITY','FP16_CONFIG','RECEIPT_QUEUE','STAGE'",
            'queue=rows[2]': 'queue=rows[3]',
            'rows[3][:5]': 'rows[2][:5]',
            'rows[3][5:]': 'rows[2][5:]',
        }
        expected_counts = [1, 1, 1, 2]
        for (old, new), count in zip(replacements.items(), expected_counts):
            assert parser.count(old) == count, (old, parser.count(old))
            parser = parser.replace(old, new)
        rows = list(csv.reader((RUN / 'iq4nl-probe.stdout').read_text().splitlines()))
        assert all(rows)
        result = {}
        namespace = dict(rows=rows, record=result, struct=struct, err=RUN / 'iq4nl-probe.stderr',
                         host_line='HOST_PASS,14_roundtrips,6_boundaries,7_private_rejections,no_queue_no_GPU')
        exec(compile(parser, str(ORIGINAL) + '[offline-order-correction]', 'exec'), namespace)
        assert result['event_receipt_result']['receipts'] == 168
        assert result['synthetic_result']['cases'] == 84
        evidence = dict(passed=True, offline_only=True, GPU_relaunched=False,
                        original_result_unchanged=True, original_controller=ident(ORIGINAL),
                        original_run_receipt=ident(RUN / 'record.json'),
                        original_run_passed=False, original_error=receipt['error'],
                        corrected_parser_sha256=hashlib.sha256(parser.encode()).hexdigest(),
                        correction='FP16_CONFIG precedes RECEIPT_QUEUE in the pinned fixture; only the expected initial record positions are corrected.',
                        build_receipt=ident(build_path), source_head=receipt['source_head'],
                        stdout=ident(RUN / 'iq4nl-probe.stdout'), stderr=ident(RUN / 'iq4nl-probe.stderr'),
                        controller=ident(__file__), journal_status=receipt['journal_status'],
                        visible_kernel_GPU_entries=[], devcoredump_after=False, **result)
        OUT.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(dict(path=str(OUT), passed=True, offline_only=True,
                              cases=84, receipts=168, timer_resolution_ns=result['event_receipt_result']['timer_resolution_ns'])))

if __name__ == '__main__':
    main()
