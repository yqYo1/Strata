#!/usr/bin/env python3
"""Archive the one closed upload-fix suite; never launch, stop or delete a run.

Run only after runner 69843 exits. The shared exclusive lock refuses concurrent
measurement/build/cleanup. Original receipts and result statuses are immutable.
"""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).resolve().parent
O = B/'main-primary-long-generation-suite-root-v4-upload-fix-v1'
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_HASH = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, raw):
    if path.exists():
        require(path.read_bytes() == raw, f'existing artifact differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(raw)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = read(O/'record.json')
        require(record.get('active') is False, 'run remains active or closure unknown')
        require(sha(OWNER) == OWNER_HASH, 'owner source changed')
        spec = importlib.util.spec_from_file_location('suite_owner', OWNER)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        commands = record['commands']
        require(commands and all(owner.closed(c) for c in commands), 'ownership closure unproved')
        require({c['label'] for c in commands} == {'journal-before', 'extended-suite', 'journal-after'},
                'missing/extra command receipt')
        start, exit_receipt = read(O/'engine-start.json'), read(O/'engine-exit.json')
        same_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
        if same_boot:
            identities = [i for c in commands for i in c.get('owners', [])]
            identities += [start['native'], start['wrapper']]
            for identity in identities:
                current = owner.identity(identity['pid'])
                require(not current or current['start_ticks'] != identity['start_ticks'],
                        f'owned process still present: {identity}')
        # No numeric SID scan after leader reaping: rely on the owner's final,
        # pre-reap session scan plus exact PID/start identities above.
        result = read(O/'result.json')
        suite_paths = list((O/'stock-results').glob('*/results.json'))
        journal_paths = list((O/'stock-results').glob('*/requests.jsonl'))
        require(len(suite_paths) == len(journal_paths) == 1, 'missing/ambiguous suite artifacts')
        suite_path, journal_path = suite_paths[0], journal_paths[0]
        require(suite_path.parent == journal_path.parent, 'suite/journal directory mismatch')
        suite = read(suite_path)
        require(sha(suite_path) == record['benchmark_result_sha256'] and
                suite == record['benchmark_result'], 'suite differs from closure receipt')
        require(result == record['result'], 'server/bench result differs from closure receipt')
        saved_journals = record['request_journals']
        require(len(saved_journals) == 1, 'missing/ambiguous closure journal receipt')
        saved_journal = saved_journals[0]
        require(str(journal_path.relative_to(O)) == saved_journal['path'] and
                sha(journal_path) == saved_journal['sha256'] and
                journal_path.stat().st_size == saved_journal['bytes'] and
                read(journal_path.parent/'journal-state.json') == saved_journal['state'],
                'journal/state differs from closure receipt')
        pins = {**record['source_pins'], **record['benchmark_source'], **record['recipes']}
        pins_unchanged = all(sha(P/name) == pin for name, pin in pins.items())
        binary_unchanged = sha(Path(start['engine'])) == record['binary_sha256']
        require(all(sha(P/n) == h for n, h in record['benchmark_source'].items()),
                'pinned journal verifier source changed; archive before editing the client')
        sys.path.insert(0, str(P/'tools'))
        from strata_bench import verify_request_journal
        journal_check = verify_request_journal(journal_path.parent)
        require(journal_check == saved_journal['verification'], 'journal verification differs from closure receipt')
        rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
        starts = {r['index']: r for r in rows if r['kind'] == 'start'}
        samples, request_identity_valid = [], True
        for row in rows:
            if row['kind'] == 'result':
                initial = starts.get(row['request_index'], {})
                sample = row['sample']
                request_identity_valid &= bool(initial.get('request_sha256')) and (
                    initial.get('request_sha256') == sample.get('request_sha256'))
                samples.append({k: row[k] for k in ('index', 'phase', 'run', 'turn')} |
                               {'thinking': initial.get('thinking'), 'sample': sample})
        selected = lambda phase: [r for r in samples if r['phase'] == phase]
        agent = selected('agent')
        agent_complete = len(agent) == 18 and {
            (r['run'], r['turn']) for r in agent} == {(r, t) for r in range(3) for t in range(6)} and all(
                r['thinking'] is True and r['sample'].get('ok') and
                r['sample'].get('stream_complete') and r['sample'].get('finish_reason') == 'stop' and
                type(r['sample'].get('generated_tokens')) is int and r['sample']['generated_tokens'] > 0 and
                r['sample'].get('output_chars', 0) > 0 for r in agent)
        prompt = selected('prompt/32K')
        prompt_usage_valid = len(prompt) == 3 and {r['run'] for r in prompt} == {0, 1, 2} and all(
            r['thinking'] is True and r['sample'].get('ok') and
            r['sample'].get('stream_complete') and positive(r['sample'].get('prompt_tps')) and
            type(r['sample'].get('prompt_tokens')) is int and r['sample']['prompt_tokens'] > 0 and
            r['sample'].get('prompt_tps_source') == 'engine' for r in prompt)
        nominal_prompt_meets_32768 = prompt_usage_valid and all(r['sample']['prompt_tokens'] >= 32768 for r in prompt)
        long_samples, bands = {}, {}
        for name, strong in [('regular-code', True), ('varied-prose', False)]:
            entries = selected('long-generation/' + name)
            long_samples[name] = entries
            valid = len(entries) == 3 and {r['run'] for r in entries} == {0, 1, 2}
            for row in entries:
                s = row['sample']
                offered, accepted = s.get('drafts_offered'), s.get('drafts_accepted')
                counts = type(offered) is int and type(accepted) is int and offered >= 256 and 0 <= accepted <= offered
                rate = accepted / offered if counts else None
                valid &= bool(row['thinking'] is True and s.get('ok') and s.get('stream_complete') and
                    s.get('finish_reason') in ('stop', 'length') and s.get('eligible') and
                    type(s.get('prompt_tokens')) is int and s['prompt_tokens'] >= 32768 and
                    type(s.get('generated_tokens')) is int and 1024 <= s['generated_tokens'] <= 16384 and
                    s.get('output_chars', 0) >= 512 and s.get('reused_tokens') == 0 and
                    s.get('decode_tps_source') == 'engine' and positive(s.get('decode_tps')) and
                    positive(s.get('decode_ms')) and counts and rate == s.get('acceptance_rate') and
                    (rate >= .60 if strong else rate <= .40))
            bands[name] = valid
        sample_fields = ('request_sha256', 'prompt_tokens', 'generated_tokens', 'decode_ms', 'decode_tps',
                         'output_sha256', 'reasoning_sha256', 'drafts_offered', 'drafts_accepted')
        gen = suite.get('results', {}).get('long_generation', {})
        suite_matches_journal = True
        saved_prompt = suite.get('results', {}).get('prompts', {}).get('32K', {}).get('requests', [])
        by_run = {r.get('run'): r for r in saved_prompt}
        suite_matches_journal &= len(saved_prompt) == len(prompt) == len(by_run) == 3
        for row in prompt:
            saved_sample = by_run.get(row['run'], {})
            suite_matches_journal &= all(saved_sample.get(k) == row['sample'].get(k) for k in
                ('request_sha256', 'prompt_tokens', 'prompt_ms', 'prompt_tps', 'generated_tokens'))
        sessions = suite.get('results', {}).get('agent', {}).get('sessions', [])
        saved_agent = [r for session in sessions for r in session]
        by_turn = {(r.get('run'), r.get('turn')): r for r in saved_agent}
        suite_matches_journal &= len(sessions) == 3 and len(saved_agent) == len(by_turn) == 18
        for row in agent:
            saved_sample = by_turn.get((row['run'], row['turn']), {})
            suite_matches_journal &= all(saved_sample.get(k) == row['sample'].get(k) for k in sample_fields)
        for name, entries in long_samples.items():
            saved = gen.get('workloads', {}).get(name, {}).get('requests', [])
            by_run = {r.get('run'): r for r in saved}
            suite_matches_journal &= len(saved) == len(entries) == 3 and len(by_run) == 3
            for row in entries:
                saved_sample = by_run.get(row['run'], {})
                suite_matches_journal &= all(saved_sample.get(k) == row['sample'].get(k) for k in sample_fields)
        native_identity_matches = all(exit_receipt.get(k) == start.get(k) for k in
            ('engine', 'binary_sha256', 'native', 'wrapper', 'argv', 'started_utc', 'stdin'))
        gates = dict(original_passed=record.get('passed') is True, original_complete=record.get('complete') is True,
            commands_exit_zero=all(c.get('exit_code') == 0 for c in commands),
            runner_passed=result.get('passed') is True,
            native_server_bench_exit_zero=all(v == 0 for v in (exit_receipt.get('exit_code'),
                result.get('server_exit_code'), result.get('bench_exit_code'))),
            engine_exit_receipts_match=result.get('real_engine_exit') == exit_receipt,
            native_identity_matches=native_identity_matches,
            binary_matches=exit_receipt.get('binary_sha256') == record['binary_sha256'] and binary_unchanged,
            source_client_recipe_unchanged=pins_unchanged,
            journal_valid_complete=journal_check['valid'] and journal_check['complete'],
            request_identity_valid=request_identity_valid, suite_matches_journal=suite_matches_journal,
            agent_complete=agent_complete, nominal_prompt_usage_valid=prompt_usage_valid,
            strong_weak_bands_confirmed=all(bands.values()),
            suite_complete=suite.get('complete') is True,
            thinking_and_context=suite.get('params', {}).get('think') is True and
                gen.get('thinking') is True and suite.get('machine', {}).get('context') == 262144,
            original_fault_check_passed=record.get('fresh_faults') == [] and
                record.get('driver_dump_exists') is False and record.get('boot_unchanged') is True,
            boot_still_unchanged=same_boot)
        archive = D/'receipts'/O.name
        archive.mkdir(parents=True, exist_ok=True)
        files = [O/n for n in ('record.json', 'result.json', 'engine-start.json', 'engine-exit.json',
                              'config.json', 'plan.json')]
        files += [suite_path, journal_path, journal_path.parent/'journal-state.json']
        receipts = []
        for path in files:
            raw = path.read_bytes()
            packed = gzip.compress(raw, mtime=0)
            require(gzip.decompress(packed) == raw, 'archive round-trip mismatch')
            target = archive/(path.name + '.gz')
            write_new(target, packed)
            receipts.append(dict(original=str(path), archive=str(target.relative_to(D)), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), archive_bytes=len(packed), archive_sha256=sha(target)))
        index = dict(run_root=str(O), files=receipts, originals_modified=False, cleanup_performed=False)
        write_new(archive/'index.json', (json.dumps(index, indent=2) + '\n').encode())
        summary = dict(run_root=str(O), source_revision=record['source_revision'],
            binary_sha256=record['binary_sha256'], boot_id=record['boot_id'],
            start=start, exit=exit_receipt, gates=gates, journal=journal_check,
            benchmark_protocol_qualified=all(gates.values()), original_passed=record['passed'],
            quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            speed_improvement_qualified=False, adopted=False, samples=samples,
            nominal_prompt_meets_32768=nominal_prompt_meets_32768,
            nominal_32k_scope='Actual input counts retained; capped 24-token thinking is prefill evidence only.',
            long_generation_scope='Engine decode clock includes thinking; answer-body presence is not task-quality proof.',
            receipts_index=str((archive/'index.json').relative_to(D)))
        write_new(D/'suite-v4-upload-fix-final-summary.json', (json.dumps(summary, indent=2) + '\n').encode())
        print(json.dumps({'benchmark_protocol_qualified': summary['benchmark_protocol_qualified'], 'gates': gates}))


if __name__ == '__main__':
    main()
