"""SOURCE ONLY candidate: finite actual coarse-phase N64 sanity thenN32768 fixed8192 diagnostic."""
import hashlib
import fcntl
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import time
import types

from phase_protocol import validate, validate_trace, windows, output_deltas
from phase_observer import validate as validate_phase, require_phase_admission
from runtime import validate_runtime
from transfer_ledger import validate as validate_transfers
from timer_overhead import validate as validate_overhead
from route_census import validate as validate_routes
from batch_census import validate as validate_batch
from admission import (admitted_variant, compact_environment, require_exact_variant_env, require_source_pins,
    require_qualification, require_publication, strong_closed_command, require_normal_model,
    phase_root_path as root_path, phase_predecessor_path as predecessor_path, require_phase_prefix as require_sequence_prefix, safe_record_path, require_additional_mappings, compare_strict_outputs,
    normal_quit, close_after_gate_failure, require_published_state,
    require_host_prerequisite, require_build_prerequisite, require_compiler_prerequisite, require_lint_prerequisite,
    require_chunk_qualification, require_completed_matrix, require_same_recipe, require_original_controller_proofs, matrix_path,
    SHARED, CODE_NAMES, PHASE_SEQUENCE as SEQUENCE, MATRIX, PUBLISHED_REVISION, SOURCE_REVIEW_SHA256,
    require_phase_qualification, require_closed_chunk_diagnostics, phase_environment, require_phase_environment,
    require_prior_phase, admitted_phase_variant, phase_reference_pin, require_phase_proof, IMMUTABLE_PHASE_INPUTS, root_path as chunk_root_path)

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
HERE = Path(__file__).resolve().parent
LLVM = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
OWNER = B / 'xe-llvm-v711-source-v5/direct_owner.py'
HELPER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
FIXTURE = B / 'clean64k-task-fixture-root-r3/coding-review-65536-tokens.txt'
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
MODEL = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
MTP = Path('/home/yayoi/.local/share/strata-sycl/mtp/rt')
PROFILE = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010/data/expert-profile.bin')
BASE = dict(PATH=str(LLVM / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi',
            LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(LLVM / 'lib'),
            XDG_RUNTIME_DIR='/run/user/' + str(os.getuid()))
ERROR = re.compile(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]|GPU completion unconfirmed|terminate called|Assertion .*failed|CAT error|fault response|out of resources', re.I)
with OWNER.open('rb') as owner_stream:
    if hashlib.file_digest(owner_stream, 'sha256').hexdigest() != 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f':
        raise ValueError('qualified owner identity before import')
s = importlib.util.spec_from_file_location('qualified_owner', OWNER)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')


def stat_identity(path):
    v = path.stat()
    return dict(device=v.st_dev, inode=v.st_ino, bytes=v.st_size,
                mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns)


def mapped_identity(entry, poll):
    now = m.identity(entry['pid'])
    m.require(now and now['start_ticks'] == entry['direct_identity']['start_ticks'] and
              now['state'] not in ('Z','X','x') and
              all(now[k] == entry['direct_identity'][k] for k in ('pgid','sid')),
              'mapping snapshot PID/start identity')
    poll()


def mapping_selection(entry, poll):
    mapped_identity(entry, poll)
    with Path('/proc', str(entry['pid']), 'maps').open('rb') as stream:
        data = stream.read((2 << 20) + 1)
    m.require(len(data) <= 2 << 20, 'mapping text byte budget')
    lines = data.decode('utf-8', errors='strict').splitlines()
    m.require(len(lines) <= 8192, 'mapping row budget')
    selected = {}
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        name = Path(fields[5]).name.lower()
        core = any(k in name for k in ('libsycl', 'libur_', 'libze_', 'libumf'))
        extra = any(k in name for k in ('igc', 'fcl', 'opencl-clang', 'iga', 'igdgmm')) or name.startswith(('libc.so', 'ld-linux'))
        if not (core or extra):
            continue
        m.require(not fields[5].endswith(' (deleted)'), 'selected mapped file deleted')
        selected.setdefault(fields[5], []).append(fields)
    m.require(len(selected) <= 64, 'selected mapping file budget')
    return data, lines, selected


def mapping_snapshot(entry, poll, budget, on_partial=None):
    """Read mapped files only; preserve bounded evidence on failed snapshots."""
    result = dict(pid=entry['pid'], start_ticks=entry['direct_identity']['start_ticks'],
                  files={}, complete=False, failed_stage='initial_maps', failed_path=None)
    try:
        data, lines, selected = mapping_selection(entry, poll)
        result.update(map_bytes=len(data), map_rows=len(lines))
        for path, rows in selected.items():
            result.update(failed_stage='mapped_file', failed_path=path)
            mapped_identity(entry, poll)
            canonical = Path(path).resolve(strict=True)
            fd = os.open(canonical, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                m.require(0 < before.st_size <= 512 << 20, 'mapped file size budget')
                for row in rows:
                    major, minor = (int(x, 16) for x in row[3].split(':'))
                    m.require(before.st_ino == int(row[4]) and before.st_dev == os.makedev(major, minor),
                              'mapped path inode/device identity')
                digest = hashlib.sha256()
                remaining = before.st_size
                while remaining:
                    mapped_identity(entry, poll)
                    chunk = os.read(fd, min(1 << 20, remaining))
                    m.require(chunk, 'mapped file truncated')
                    budget['bytes'] += len(chunk)
                    m.require(budget['bytes'] <= 4 << 30, 'aggregate mapped file read byte budget')
                    digest.update(chunk)
                    remaining -= len(chunk)
                def read_at(offset, length):
                    m.require(0 <= offset <= before.st_size and 0 <= length <= 1 << 20 and
                              offset + length <= before.st_size, 'bounded ELF note read')
                    mapped_identity(entry, poll)
                    value = os.pread(fd, length, offset)
                    budget['bytes'] += len(value)
                    m.require(len(value) == length and budget['bytes'] <= 4 << 30, 'bounded complete ELF read')
                    return value
                header = read_at(0, 64)
                m.require(header[:4] == b'\x7fELF' and header[4] == 2 and header[5] in (1, 2), 'ELF64 mapping')
                endian = '<' if header[5] == 1 else '>'
                phoff = struct.unpack_from(endian + 'Q', header, 32)[0]
                phsize, phnum = struct.unpack_from(endian + 'HH', header, 54)
                m.require(phsize == 56 and phnum <= 256, 'ELF program header budget')
                ids = set()
                note_bytes = 0
                for i in range(phnum):
                    ph = read_at(phoff + i * phsize, phsize)
                    if struct.unpack_from(endian + 'I', ph)[0] != 4:
                        continue
                    offset, size = struct.unpack_from(endian + 'Q', ph, 8)[0], struct.unpack_from(endian + 'Q', ph, 32)[0]
                    note_bytes += size
                    m.require(note_bytes <= 1 << 20, 'ELF aggregate note budget')
                    notes = read_at(offset, size)
                    at = 0
                    while at + 12 <= len(notes):
                        namesz, descsz, kind = struct.unpack_from(endian + 'III', notes, at)
                        at += 12
                        desc_at = at + ((namesz + 3) & ~3)
                        end = desc_at + ((descsz + 3) & ~3)
                        m.require(end <= len(notes), 'complete ELF note')
                        if kind == 3 and notes[at:at + namesz].rstrip(b'\0') == b'GNU':
                            m.require(0 < descsz <= 128, 'GNU build-id byte budget')
                            ids.add(notes[desc_at:desc_at + descsz].hex())
                        at = end
                m.require(len(ids) == 1, 'one ELF GNU build-id')
                after = os.fstat(fd)
                def state(v):
                    return dict(device=v.st_dev, inode=v.st_ino, bytes=v.st_size,
                                mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns)
                m.require(state(before) == state(after) == stat_identity(canonical), 'stable mapped file stat')
                result['files'][path] = dict(path=path, canonical_path=str(canonical),
                    sha256=digest.hexdigest(), stat=state(after), elf_build_id=next(iter(ids)),
                    mapping_rows=len(rows), mapped_rows=[' '.join(row) for row in rows], original_runtime=any(k in Path(path).name.lower() for k in ('libsycl', 'libur_', 'libze_', 'libumf')))
            finally:
                os.close(fd)
            if on_partial is not None:
                on_partial(result)
        result.update(failed_stage='final_maps', failed_path=None)
        final_data, final_lines, final_selected = mapping_selection(entry, poll)
        m.require(final_selected == selected, 'selected mapped rows stable throughout snapshot')
        mapped_identity(entry, poll)
        result.update(complete=True, selected_rows_stable=True,
                      final_map_bytes=len(final_data), final_map_rows=len(final_lines),
                      failed_stage=None, failed_path=None)
        return result
    except BaseException as error:
        result['error'] = repr(error)
        if on_partial is not None:
            on_partial(result)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-tokens',type=int,choices=[64,32768],required=True)
    parser.add_argument('--chunk', type=int, choices=[8192], required=True)
    parser.add_argument('--group', type=int, choices=[16], default=16)
    parser.add_argument('--compact', choices=['on'], required=True)
    parser.add_argument('--qualification', type=Path, default=HERE/'qualification.pending.json')
    parser.add_argument('--qualification-sha256', default='PENDING_MAIN_EXACT_SHA256')
    args = parser.parse_args()
    try:
        admitted_phase_variant(args.input_tokens,args.chunk,args.compact,args.group)
    except ValueError as error:
        parser.error(str(error))
    safe_record_path(args.qualification.resolve())
    m.require(re.fullmatch('[0-9a-f]{64}', args.qualification_sha256) and
              m.sha(args.qualification) == args.qualification_sha256, 'actual Main exact qualification manifest hash')
    qualification = json.loads(args.qualification.read_text())
    m.require(qualification['status'] == 'passed' and qualification['main_reviewed'] is True and
              qualification['base_revision'] == '0740b43488d812117f021ccbe4bd0e15d223f955', 'actual Main qualification, no pending launch')
    expected_review = json.loads((HERE/'source-review.expected.json').read_text())
    m.require(m.sha(HERE/'source-review.expected.json') == SOURCE_REVIEW_SHA256, 'immutable42 applied review copy')
    original_path = Path(qualification['original_qualification'])
    safe_record_path(original_path)
    m.require(original_path == B/'xe-compact-moe-main-admission-converter-root-v1/qualification.resolved.json' and
              m.sha(original_path) == qualification['original_qualification_sha256'] == qualification['receipt_sha256'][str(original_path)],
              'exact actual original qualified admission')
    original_qualification = json.loads(original_path.read_text())
    chunk_path = Path(qualification['chunk_qualification'])
    safe_record_path(chunk_path)
    m.require(chunk_path == B/'xe-compact-moe-chunk8192-main-admission-root-v1/qualification.resolved.json' and
              m.sha(chunk_path) == qualification['chunk_qualification_sha256'] == qualification['receipt_sha256'][str(chunk_path)],
              'actual immutable qualified chunk admission')
    chunk_qualification = json.loads(chunk_path.read_text())
    require_phase_qualification(qualification, expected_review['source_pins'], chunk_qualification, original_qualification)
    n,chunk,group,compact = args.input_tokens,args.chunk,args.group,args.compact
    kind = 'coarse-phase-n' + str(n) + '-chunk8192'
    proof_path=Path(qualification['phase_proof_qualification'])
    recipe_path=Path(qualification['resolved_phase_recipe'])
    for path,key in ((proof_path,'phase_proof_qualification_sha256'),(recipe_path,'resolved_phase_recipe_sha256')):
        safe_record_path(path)
        m.require(path.stat().st_size <= 8 << 20 and m.sha(path)==qualification[key], 'actual separate bounded Main phase converter bytes')
    phase_proof=json.loads(proof_path.read_text())
    phase_recipe=json.loads(recipe_path.read_text())
    require_phase_proof(phase_proof,qualification,expected_review['source_pins'])
    require_phase_admission(phase_recipe,phase_proof,qualification['phase_proof_qualification_sha256'])
    expected_mode = {offset: ('full_nonresident_stream' if size >= 2048 else 'routed') for offset, size in windows(n, chunk)}
    reference = None
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace = B/'xe-compact-moe-phase-observer-root-v1'
        allowed_roots = {root_path(B,*pair) for pair in SEQUENCE}
        m.require(not namespace.exists() or set(namespace.iterdir()) <= allowed_roots, 'only exact two finite roots in phase namespace')
        present = {pair for pair in SEQUENCE if root_path(B, *pair).exists()}
        require_sequence_prefix(present,n,chunk)
        campaign_started = time.monotonic()
        if present:
            first_path = root_path(B,*SEQUENCE[0])/'record.json'
            safe_record_path(first_path)
            m.require(first_path.stat().st_size <= 8 << 20, 'bounded first phase receipt')
            first_record = json.loads(first_path.read_text())
            current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            require_prior_phase(first_record,current_boot,64,8192,m.sha(__file__),args.qualification_sha256,
                                phase_reference_pin(qualification,64))
            campaign_started = first_record['campaign_started_monotonic']
        m.require(0 <= time.monotonic()-campaign_started < 1800, 'finite whole campaign admission before any new owned command')
        out = root_path(B,n,chunk)
        m.require(not out.exists(), 'immutable fresh diagnostic output')
        out.mkdir(mode=0o700, parents=True)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        r = dict(active=True, complete=False, passed=False, commands=[],
                 started_utc=m.utc(), started_monotonic=time.monotonic(),
                 campaign_started_monotonic=campaign_started, boot_id=boot, controller_sha256=m.sha(__file__),
                 protocol_sha256=m.sha(HERE / 'protocol.py'), owner_sha256=m.sha(OWNER),
                 hash_helper_sha256=m.sha(HERE / 'hash_inputs.py'), runtime_helper_sha256=m.sha(HERE / 'runtime.py'),
                 mode='free', candidate_revision=qualification['base_revision'], source_uncommitted=False, input_tokens=n, group=group, compact=compact, batch='off',
                 qualification_sha256=args.qualification_sha256, qualification_path=str(args.qualification),
                 mapping_snapshots={}, mapping_read_budget=dict(bytes=0), gen_sent=False,
                 gen_send_attempted=False,
                 adopted=False, upstream_pristine=False, performance_eligible=False, diagnostic_kind=kind,
                 full_model_quality_qualified=False, broad_quality_qualified=False, full262144_qualified=False,
                 restore_qualified=False, requested_chunk=chunk, expected_actual_chunk=chunk,
                 sequence_index=SEQUENCE.index((n,chunk)),sequence_root_budget=2,quiet_arm=False,
                 campaign_max_wall_seconds=1800, campaign_max_artifact_bytes=192 << 20,
                 campaign_max_raw_bytes=128 << 20, campaign_max_receipt_bytes=16 << 20,
                 reset_executed=False, retry_count=0, model_launch_attempted=False,
                 raw_text_log_budget_bytes=64 << 20, total_artifact_budget_bytes=96 << 20,
                 whole_wall_budget_seconds=900, model_wall_budget_seconds=500,
                 model_rss_budget_bytes=88 << 30, observed_libraries={},
                 scope='Finite coarse adjacent compute-marker phase diagnostic N64 sanity thenN32768, fixed8192 compactON G16/context262144. Original bounded trace/count/validation/warning environment plus ONLY timing1. GEN16/LP5, no exclusive kernel/H2D/criticalpath or performance/broadquality/full262144/RESTORE claim; total observer overhead unmeasured.')

        def save():
            payload = json.dumps(r, indent=2) + '\n'
            encoded_bytes = len(payload.encode())
            m.require(encoded_bytes <= 8 << 20, 'individual diagnostic receipt byte budget')
            paths = [p for p in out.rglob('*') if p.is_file() and p != out/'record.json']
            m.require(sum(p.stat().st_size for p in paths) + encoded_bytes <= 96 << 20,
                      'combined model/provenance/receipt artifact byte budget')
            m.require(sum(p.stat().st_size for p in paths if p.suffix in ('.stdout', '.stderr')) <= 64 << 20,
                      'combined model/provenance diagnostic raw log byte budget')
            (out / 'record.json').write_text(payload)

        owner = m.Owner(out / 'commands', save)
        r['commands'] = owner.commands
        # Separate anchored read-only provenance owner: never overwrite the active model Owner's state.
        guard_owner = m.Owner(out / 'guards', save)
        r['guard_commands'] = guard_owner.commands
        save()
        begin = time.monotonic()
        cursor = None
        helper = None

        def run(label, argv, env=BASE, wall=20, interaction=None, rss=2 << 30):
            remaining = min(900 - (time.monotonic() - begin),
                            1800 - (time.monotonic() - r['campaign_started_monotonic']))
            m.require(remaining > 0, 'whole diagnostic deadline')
            e, so, se = owner.run(label, argv, env, W, interaction=interaction,
                                  wall=min(wall, remaining), cpu=int(wall) * 2,
                                  text_cap=64 << 20, total_cap=96 << 20,
                                  rss_cap=rss)
            if label == 'model':
                r['model_normal_close'] = bool(strong_closed_command(e) and e['exit_code'] == 0)
                r.setdefault('model_close', {}).update(exit_code=e.get('exit_code'),
                    owner_error=e.get('error'), owner_errors=e.get('errors'), cleanup=e.get('cleanup'),
                    normal=r['model_normal_close'], forced=bool(e.get('cleanup')))
                r['phase_raw_report_lines']=[line for line in se.read_text().splitlines() if line.startswith(('strata prefill timing:','strata prefill timer overhead:'))]
                save()  # Preserve raw observer reports before runtime/health/parser/output judgments.
                if r.get('primary_model_failure'):
                    raise RuntimeError(r['primary_model_failure'])
            m.require(strong_closed_command(e) and e['exit_code'] == 0, 'normal closed ' + label)
            return so.read_text(), se.read_text()

        def kernel(label, args):
            text, _ = run(label, ['/usr/bin/journalctl', '-k', *args, '--no-pager', '-o', 'json'])
            rows = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
            relevant = [x for x in rows if '0000:05:00.0' in x.get('MESSAGE', '') or re.search(r'\bxe\b', x.get('MESSAGE', ''))]
            r[label + '_relevant_entries'] = relevant
            faults = [x['MESSAGE'] for x in relevant if helper.FAULT.search(x['MESSAGE'])]
            m.require(not faults and not DUMP.exists(), 'fault-free ' + label)
            return faults

        try:
            def receipt(path, same_boot=False, release_cli_rejections=False, controller_cli_rejections=False, prerequisite=None):
                safe_record_path(path)
                m.require(Path(path).stat().st_size <= 8 << 20, 'individual retained receipt byte budget')
                if str(path) in qualification['receipt_sha256']:
                    m.require(m.sha(path) == qualification['receipt_sha256'][str(path)], 'exact Main prerequisite receipt hash')
                else:
                    m.require(str(path).startswith(str(B/'xe-compact-moe-phase-observer-root-v1') + '/'), 'Main must pin every prerequisite receipt')
                    m.require(Path(path).name == 'record.json', 'exact dynamic diagnostic receipt')
                v = json.loads(path.read_text())
                m.require(v['passed'] is True and v['complete'] is True and v['active'] is False and
                          v.get('commands') and all(strong_closed_command(c) for c in v['commands']), 'actual closed PASS receipt ' + str(path))
                if prerequisite is not None:
                    prerequisite(v)
                allowed_rejections = {'invalid-max-new-nan': ['--max-new','nan'],
                    'invalid-max-new-9223372036854775808': ['--max-new','9223372036854775808'],
                    'invalid-max-context-262144x': ['--max-context','262144x']} if release_cli_rejections else {}
                if controller_cli_rejections:
                    m.require(not release_cli_rejections, 'distinct expected rejection recipe')
                    cases = [['--input-tokens','64','--group','16','--batch','on'],
                             ['--input-tokens','64','--group','16','--batch','off','--reference','/tmp/unused-no-open'],
                             ['--input-tokens','65','--group','16','--batch','off'],
                             ['--input-tokens','64','--group','65','--batch','off']]
                    prior_controller = B/'xe-native-dequant-prefill-model-controller-source-v8/run_root.py'
                    allowed_rejections = {'invalid-cli-'+str(i): [str(prior_controller),*args] for i,args in enumerate(cases)}
                for c in v['commands']:
                    if prerequisite is not None:
                        # The exported prerequisite predicate checked every direct/reused command
                        # and its exact narrow nonzero recipe before this immutable receipt is admitted.
                        continue
                    if c['label'] in allowed_rejections:
                        m.require(c['exit_code'] == 2 and c['argv'][1:] == allowed_rejections[c['label']] and not c['errors'], 'exact successful malformed CLI rejection')
                    else:
                        m.require(c['exit_code'] == 0, 'successful prerequisite command ' + c['label'])
                if release_cli_rejections or controller_cli_rejections:
                    m.require({c['label'] for c in v['commands'] if c['exit_code'] == 2} == set(allowed_rejections), 'all exact expected malformed CLI cases rejected')
                if same_boot:
                    m.require(v['boot_id'] == boot, 'same boot ' + str(path))
                r.setdefault('receipt_pins', {})[str(path)] = m.sha(path)
                return v

            review_path = B/'xe-compact-moe-applied-main-review-v2.json'
            m.require(m.sha(review_path) == SOURCE_REVIEW_SHA256 == qualification['source_review_sha256'], 'exact Main applied42 source review')
            review = json.loads(review_path.read_text())
            m.require(review == expected_review and review['main_reviewed'] is True, 'immutable actual source review copy')
            r['source_applied_main_review_sha256'] = m.sha(review_path)
            r['source_pins'] = review['source_pins']
            require_source_pins(r['source_pins'], qualification['production_source_pins'])
            host = receipt(Path(qualification['host_fixture_receipt']))
            require_host_prerequisite(host, r['source_pins'], W, qualification['compiler_receipt_sha256'])
            m.require(all(m.sha(Path(value['path'])) == value['sha256'] and Path(value['path']).stat().st_size == value['bytes']
                          for value in host['binaries'].values()), 'current actual release/sanitizer host fixture binaries')
            proofs = {}
            for mode in ('free', 'contrib-llvm', 'free-asan', 'contrib-llvm-asan'):
                path = Path(qualification['build_receipts'][mode])
                m.require(m.sha(path) == qualification['build_receipt_sha256'][mode], 'exact four-mode qualification hash')
                v = receipt(path, release_cli_rejections=not mode.endswith('-asan'))
                m.require(v['binary'] == qualification['qualified_build_binaries'][mode] and
                          m.sha(Path(v['binary']['path'])) == v['binary']['sha256'] and
                          Path(v['binary']['path']).stat().st_size == v['binary']['bytes'], 'current actual qualified mode binary')
                for value in v['build_recipe'].values():
                    m.require(m.sha(Path(value['path'])) == value['sha256'], 'exact retained build recipe bytes')
                cache = Path(v['build_recipe']['CMakeCache.txt']['path']).read_text()
                require_build_prerequisite(v, mode, r['source_pins'], W, m.sha(Path(qualification['host_fixture_receipt'])),
                    qualification['compiler_receipt_sha256'], qualification['qualified_build_binaries'][mode], cache)
                proofs[mode] = v
            release = proofs['free']
            m.require(release['binary'] == qualification['free_release_binary'], 'exact Main free binary')
            compiler = receipt(Path(qualification['compiler_receipt']))
            m.require(m.sha(Path(qualification['compiler_receipt'])) == qualification['compiler_receipt_sha256'], 'actual compiler proof hash')
            require_compiler_prerequisite(compiler)
            m.require(all(m.sha(Path(value['path'])) == value['sha256'] and Path(value['path']).stat().st_size == value['bytes']
                          for value in compiler['installed_files'].values()), 'exact installed compiler/runtime qualification')
            def lint_check(value):
                original = None
                original_sha = None
                if value.get('reused_commands'):
                    original_path = Path(value['original_failed_comparison']['path'])
                    safe_record_path(original_path)
                    m.require(original_path.stat().st_size <= 8 << 20, 'original retained lint receipt byte budget')
                    original_sha = qualification['receipt_sha256'][str(original_path)]
                    m.require(m.sha(original_path) == original_sha, 'Main pinned original failed lint comparison bytes')
                    original = json.loads(original_path.read_text())
                    r.setdefault('receipt_pins', {})[str(original_path)] = original_sha
                require_lint_prerequisite(value, r['source_pins'], review['changed_source_pins'],
                    qualification['build_receipt_sha256'], original, original_sha)
            lint = receipt(Path(qualification['lint_receipt']), prerequisite=lint_check)
            m.require(m.sha(Path(qualification['lint_receipt'])) == qualification['lint_receipt_sha256'], 'actual Main lint qualification hash')
            bridge_path = Path(qualification['publication_bridge'])
            safe_record_path(bridge_path)
            m.require(m.sha(bridge_path) == qualification['publication_bridge_sha256'], 'actual Main publication bridge hash')
            bridge = json.loads(bridge_path.read_text())
            require_publication(bridge, qualification, r['source_pins'], W)
            m.require(bridge['published_revision'] == PUBLISHED_REVISION, 'exact unchanged committed compact candidate39e')
            r['publication_bridge'] = dict(path=str(bridge_path), sha256=m.sha(bridge_path), published_revision=bridge['published_revision'])
            r['published_revision'] = bridge['published_revision']
            r['working_diff_sha256'] = hashlib.sha256(b'').hexdigest()

            def source_guard(phase):
                m.require(all(m.sha(W/p) == h for p,h in r['source_pins'].items()), 'exact published42 source bytes at ' + phase)
                m.require(m.sha(Path(release['binary']['path'])) == release['binary']['sha256'], 'exact built free binary at ' + phase)
                m.require(all(m.sha(Path(value['path'])) == value['sha256'] and Path(value['path']).stat().st_size == value['bytes']
                              for value in compiler['installed_files'].values()), 'qualified compiler files at ' + phase)
                m.require(all(m.sha(Path(proof['binary']['path'])) == proof['binary']['sha256'] for proof in proofs.values()),
                          'all four actual qualified binaries at ' + phase)
                m.require(all(m.sha(HERE/name) == h for name,h in qualification['controller_source_pins'].items()), 'whole controller helpers at ' + phase)
                m.require(all(m.sha(HERE/name)==digest for name,digest in IMMUTABLE_PHASE_INPUTS.items()), 'byte-exact original phase/window parser and pending recipe at '+phase)
                m.require(all(m.sha(B/'xe-compact-moe-model-controller-source-v2'/name) == h
                              for name,h in original_qualification['controller_source_pins'].items()), 'original actual controller closure at ' + phase)
                m.require(all(m.sha(B/'xe-compact-moe-chunk8192-controller-source-v1'/name) == h
                              for name,h in chunk_qualification['controller_source_pins'].items()), 'actual chunk controller closure at '+phase)
                m.require(m.sha(OWNER) == r['owner_sha256'] and m.sha(HELPER) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'global owner/helper at ' + phase)
                m.require(m.sha(review_path) == SOURCE_REVIEW_SHA256 and m.sha(bridge_path) == r['publication_bridge']['sha256'] and
                          m.sha(args.qualification) == r['qualification_sha256'], 'Main admission manifests stable at ' + phase)
                m.require(m.sha(proof_path)==qualification['phase_proof_qualification_sha256'] and
                          m.sha(recipe_path)==qualification['resolved_phase_recipe_sha256'], 'actual separate phase converter proof/recipe at '+phase)
                m.require(all(m.sha(Path(path)) == digest for path,digest in qualification['receipt_sha256'].items()),
                          'all immutable Main proof/retained reference bytes at ' + phase)
                for proof in proofs.values():
                    m.require(all(m.sha(Path(value['path'])) == value['sha256'] for value in proof['build_recipe'].values()),
                              'actual source/compiler/build recipe bytes at ' + phase)
                if phase != 'admission':
                    m.require(m.sha(controller_review_path) == r['controller_main_review_sha256'], 'new controller Main review at ' + phase)
                def read_git(label, argv):
                    remaining = min(900 - (time.monotonic() - begin),
                            1800 - (time.monotonic() - r['campaign_started_monotonic']))
                    m.require(remaining > 0, 'whole diagnostic provenance deadline')
                    entry, stdout, _ = guard_owner.run(label + '-' + phase, argv, BASE, W,
                        wall=min(20, remaining), cpu=40, text_cap=1 << 20, total_cap=96 << 20, rss_cap=2 << 30)
                    m.require(strong_closed_command(entry) and entry['exit_code'] == 0, 'strong read-only source provenance closure')
                    logs = [p for directory in (out/'commands', out/'guards') for p in directory.iterdir()
                            if p.suffix in ('.stdout', '.stderr')]
                    m.require(sum(p.stat().st_size for p in logs) <= 64 << 20, 'combined diagnostic/provenance raw log cap')
                    return stdout.read_text()
                head = read_git('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
                status = read_git('source-status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=all'])
                patch = read_git('source-diff', ['/usr/bin/git', 'diff', 'HEAD'])
                require_published_state({p:m.sha(W/p) for p in r['source_pins']}, r['source_pins'],
                    head, r['published_revision'], status, patch, m.sha(Path(release['binary']['path'])), release['binary']['sha256'])
                r.setdefault('source_guard_phases', []).append(phase)
            source_guard('admission')
            observer = receipt(B/'xe-native-dequant-batch-census-observer-cpu-root-v2/record.json')
            # Main records original OFF and independent ON validator pins, never generic full-model qualification.
            observer_pins = observer['source_pins']
            qualified_locations = {'transfer_ledger.py':B/'xe-source-llvm-census32k-v3/transfer_ledger.py',
                                   'batch_census.py':B/'xe-native-dequant-batch-census-observer-source-v2/batch_census.py'}
            for name, location in qualified_locations.items():
                m.require(observer_pins[str(location)] == m.sha(HERE/name), 'Main CPU-qualified ON v2 helper ' + name)
            off_observer = receipt(B/'xe-chunk6144-capture-and-observer-cpu-root-v1/record.json')
            old_route = B/'xe-prefill-route-census-observer-source-v1/route_census.py'
            m.require(off_observer['observer_main_review']['artifact_pins'][str(old_route)] == m.sha(HERE/'route_census.py') == m.sha(old_route),
                      'unchanged independently CPU-qualified original OFF route observer')
            prior_cpu = receipt(B/'xe-native-dequant-prefill-model-controller-cpu-root-v6/record.json', controller_cli_rejections=True)
            prior_review_path = B/'xe-native-dequant-prefill-model-controller-main-review-v7.json'
            m.require(m.sha(prior_review_path) == 'b974d05cb360ddd2f70be8af37b9279241b380610577834b4a70468074f00118', 'actual CPUv6 controller Main review')
            prior_review = json.loads(prior_review_path.read_text())
            m.require(prior_review['main_reviewed'] is True and prior_cpu['source_pins'] == prior_review['source_pins'] and
                      all(prior_cpu['source_pins'][name] == digest == m.sha(HERE/name) for name,digest in SHARED.items()), 'byte-exact eight shared helpers and actual CPUv6 provenance')
            cpu = receipt(Path(qualification['controller_cpu_receipt']))
            r['diagnostic_helper_pins'] = cpu['source_pins']
            m.require(r['diagnostic_helper_pins'] == qualification['controller_source_pins'] and
                      set(r['diagnostic_helper_pins']) == CODE_NAMES and
                      all(m.sha(HERE/name) == h for name,h in r['diagnostic_helper_pins'].items()), 'actual Main new controller/exported-fixture/helper qualification')
            controller_review_path = Path(qualification['controller_review'])
            safe_record_path(controller_review_path)
            m.require(m.sha(controller_review_path) == qualification['controller_review_sha256'], 'actual Main new controller review hash')
            controller_review = json.loads(controller_review_path.read_text())
            m.require(controller_review['main_reviewed'] is True and controller_review['source_pins'] == r['diagnostic_helper_pins'], 'Main reviewed new controller exact closure')
            require_original_controller_proofs(cpu,controller_review,qualification)
            r['controller_main_review_sha256'] = m.sha(controller_review_path)
            original_review_path = Path(original_qualification['controller_review'])
            safe_record_path(original_review_path)
            m.require(m.sha(original_review_path) == original_qualification['controller_review_sha256'], 'actual original Main controller review identity')
            original_cpu = receipt(Path(original_qualification['controller_cpu_receipt']))
            original_review = json.loads(original_review_path.read_text())
            require_original_controller_proofs(original_cpu, original_review, original_qualification)
            matrix = {}
            for matrix_n, matrix_variant in MATRIX:
                key = f'n{matrix_n}-{matrix_variant}'
                pin = qualification['six_case_receipts'][key]
                path = matrix_path(B, matrix_n, matrix_variant)
                m.require(pin['path'] == str(path) and pin['sha256'] == qualification['receipt_sha256'][str(path)] == m.sha(path),
                          'actual immutable all-six current6144 prerequisite pins')
                matrix[key] = receipt(path, True)
            r['six_case_admission'] = require_completed_matrix(matrix, qualification['six_case_receipts'],
                original_qualification, qualification['original_qualification_sha256'], boot, B)
            chunk_cpu = receipt(Path(chunk_qualification['controller_cpu_receipt']))
            chunk_review_path = Path(chunk_qualification['controller_review'])
            m.require(m.sha(chunk_review_path) == chunk_qualification['controller_review_sha256'], 'actual chunk controller review hash')
            require_original_controller_proofs(chunk_cpu, json.loads(chunk_review_path.read_text()), chunk_qualification)
            diagnostics = {}
            for diag_n, diag_chunk in ((32768,8192),(65536,6144),(65536,8192)):
                key = f'n{diag_n}-t{diag_chunk}'
                pin = qualification['three_case_receipts'][key]
                path = chunk_root_path(B,diag_n,diag_chunk)/'record.json'
                m.require(pin['path'] == str(path) and pin['sha256'] == qualification['receipt_sha256'][str(path)] == m.sha(path),
                          'Main actual closed three-case pins, never unresolved/live reads')
                diagnostics[key] = receipt(path,True)
            r['three_case_admission'] = require_closed_chunk_diagnostics(diagnostics,qualification['three_case_receipts'],
                chunk_qualification,qualification['chunk_qualification_sha256'],boot,B)
            reference_pin=phase_reference_pin(qualification,n)
            reference=matrix['n64-on'] if n==64 else diagnostics['n32768-t8192']
            anchor=reference
            m.require(str(Path(qualification['first_predecessor'])) in qualification['receipt_sha256'], 'Main exact closed campaign predecessor pin')
            prior_cases={}
            for prior_n,prior_chunk in SEQUENCE[:SEQUENCE.index((n,chunk))]:
                path=root_path(B,prior_n,prior_chunk)/'record.json'
                value=receipt(path,True)
                require_prior_phase(value,boot,prior_n,prior_chunk,r['controller_sha256'],r['qualification_sha256'],phase_reference_pin(qualification,prior_n))
                require_phase_environment(matrix['n64-on']['environment'],value['environment'])
                m.require(value['source_pins']==r['source_pins'] and value['binary']==release['binary'] and
                          value['model_inputs']==anchor['model_inputs'] and value['input']==matrix['n64-on']['input'], 'exact prior sanity source/model/input')
                require_same_recipe(matrix['n64-on']['argv'],value['argv'],6144,8192)
                for phase in ('ready_before_gen','after_gen'):
                    require_additional_mappings(value['mapping_snapshots'][phase],qualification['qualified_additional_mappings'][phase])
                m.require(all(value[k]['all_observed_exactly_qualified'] is True for k in
                    ('ready_runtime_mapping_validation','after_gen_runtime_mapping_validation','runtime_mapping_validation')), 'prior phase runtime guards')
                m.require(value['health-before_protocol'][:6]==value['health-after_protocol'][:6]==anchor['health-before_protocol'][:6], 'prior phase freshhealth')
                prior_cases[(prior_n,prior_chunk)]=value
            if prior_cases:
                r['campaign_started_monotonic']=prior_cases[SEQUENCE[0]]['campaign_started_monotonic']
            m.require(0<=time.monotonic()-r['campaign_started_monotonic']<1800, 'finite1800s sameboot phase campaign')
            predecessor=predecessor_path(B,n,chunk,qualification['first_predecessor'])
            previous=receipt(predecessor,True)
            require_normal_model(previous,boot)
            r['predecessor']=dict(path=str(predecessor),sha256=m.sha(predecessor))
            r['prior_case_pins']={str(root_path(B,x,t)/'record.json'):m.sha(root_path(B,x,t)/'record.json') for x,t in prior_cases}
            r['phase_recipe']=dict(path=str(recipe_path),sha256=m.sha(recipe_path),proof_path=str(proof_path),proof_sha256=m.sha(proof_path))
            # Original six-case records retain their own validator status; no active/retired raw logs are consumers.
            verified = {'health':receipt(B/'xe-source-llvm-small-health-root-v3/record.json', True),
                        'cap':receipt(B/'xe-free-host-capabilities-root-v3/record.json', True)}
            cap = verified['cap']
            m.require(cap['mode'] == 'free' and cap['query_output'].count('HOST_USM 1') == 1 and
                      cap['query_output'].count('ATOMIC_HOST_USM 0') == 1 and cap['query_output'].count('ZE_HOST_CAPS 3') == 1,
                      'actual host-USM without concurrent HOST access capability receipt')
            m.require(all(m.sha(Path(x['path'])) == x['sha256'] for x in cap['loaded_libraries']), 'capability actual libraries stable')
            gpu_header = (W/'include/strata/core/gpu.hpp').read_text()
            prefill_source = (W/'src/prefill/prefill.cpp').read_text()
            m.require('inline constexpr bool host_handoff_concurrent(Stream = nullptr) { return false; }' in gpu_header and
                      'return cpu_share_env() != 0.0 && strata::gpu::host_handoff_concurrent();' in prefill_source,
                      'pinned protocol hard-disables CPU share independently of route/group diagnostics')
            prior_model_path = B/'xe-source-llvm-census32k-root-v3/counts/record.json'
            prior_model = receipt(prior_model_path, True)
            r['cpu_exclusion_proof'] = dict(cpu_experts=0, capability_receipt_path=str(B/'xe-free-host-capabilities-root-v3/record.json'),
                capability_receipt_sha256=m.sha(B/'xe-free-host-capabilities-root-v3/record.json'),
                gpu_header_sha256=r['source_pins']['include/strata/core/gpu.hpp'],
                prefill_sha256=r['source_pins']['src/prefill/prefill.cpp'],
                old_model_capability_field_present='capabilities' in prior_model,
                provenance='actual independent HOST_USM1 ATOMIC_HOST_USM0 ZE_HOST_CAPS3; pinned host_handoff_concurrent false AND cpu_share_on constraint; old model record does not itself expose capabilities')
            m.require(int(prior_model['effective_settings']['vram_free_mib']) >= 512, 'previous actual READY reserve')
            r['memory_plan'] = dict(previous_path=str(prior_model_path), previous_sha256=m.sha(prior_model_path),
                previous_free_mib=int(prior_model['effective_settings']['vram_free_mib']), group=group,
                minimum_ready_free_mib=512, requested_ring=8, owned_ring=16, fullstream_ring=16, routed_ring=8,
                source_requested_owned_bytes=2385490688 if chunk == 6144 else 2999276288,
                source_requested_delta_6144_to8192=613785600, physical_commitment_derived=False,
                preinit_free_bytes=None, peak_phase_free_bytes=None,
                unavailable_telemetry='Pinned source exposes READY post-init free memory; no new allocation/phase hooks introduced.',
                scope='R148 requested owned extents, not fit proof; do not subtract old READY free memory to infer fit. Actual init/READY reserve/firstGEN/fault/health gates independently required;8192OFF excluded.')
            layout_path = B/'xe-prefill-native-transfer-layout-v1.json'
            m.require(m.sha(layout_path) == 'c9b40d122d19f277280560a495ad0a1c6ce98376ef0660c1f75e9b3026bd642b', 'independent model metadata')
            layout = json.loads(layout_path.read_text())
            m.require(m.sha(Path(layout['layout_path'])) == layout['layout_sha256'] and layout['layers'] == 48 and
                      layout['experts_per_layer'] == 512 and [x['layer'] for x in layout['expected_layers']] == list(range(48)), 'full native48layer geometry')
            r['layout_reference_sha256'] = m.sha(layout_path)
            expected_layers = {x['layer']:{k:x[k] for k in ('blob_bytes','gu_type','down_type')} for x in layout['expected_layers']}
            expectations = dict(expected_device=0,expected_stage=[0,48],expected_pos0=0,expected_n=n-1,
                expected_chunk_limit=chunk,expected_mode=expected_mode,expected_issuer='inline',expected_format='native_gguf',expected_layers=expected_layers)
            r['transfer_expectations'] = dict(expectations, expected_mode={str(k):v for k,v in expected_mode.items()}, expected_layers={str(k):v for k,v in expected_layers.items()})
            r['binary'] = release['binary']
            binary = Path(r['binary']['path'])
            m.require(m.sha(binary) == r['binary']['sha256'], 'qualified current free release executable')
            m.require(os.geteuid() != 0 and not DUMP.exists(), 'unprivileged no preexisting GPU dump')
            m.require(m.sha(HELPER) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'diagnostic helper')
            helper = types.ModuleType('read_only_diagnostics')
            exec(compile(HELPER.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(HELPER), 'exec'), helper.__dict__)
            m.require(m.sha(FIXTURE) == '5284d8fa53d1ef28f6ba916a629faf901c890bc0c669bf986ee04ed72aa2c3ff', 'canonical parent fixture')
            ids = [int(x) for x in re.split(r'[\s,]+', FIXTURE.read_text().strip())]
            m.require(len(ids) == 65536 and all(0 <= x < 248320 for x in ids), 'canonical parent count/domain')
            fixture_manifest_path = FIXTURE.with_name('manifest.json')
            m.require(m.sha(fixture_manifest_path) == '0fbf6a16a90b55a3767145d1217b2027e218c0537a487faa428ab6527e6771b1', 'immutable canonical task-tail proof')
            fixture_manifest = json.loads(fixture_manifest_path.read_text())
            m.require(fixture_manifest['passed'] and not fixture_manifest['active'] and
                      fixture_manifest['fixture']['sha256'] == m.sha(FIXTURE) and
                      fixture_manifest['suffix']['tokens'] == 49 and
                      fixture_manifest['suffix']['encode_exact'] and fixture_manifest['suffix']['decode_exact'],
                      'verified canonical 49-token closing task and assistant suffix')
            tokens = ids[:n-49] + ids[-49:]
            m.require(len(tokens) == n and tokens[-49:] == ids[-49:], 'complete task boundary preserved at exact requested N')
            m.require(n != 65536 or tokens == ids, '64K uses the entire canonical fixture without omission')
            r['input_derivation'] = dict(parent_manifest_path=str(fixture_manifest_path),
                parent_manifest_sha256=m.sha(fixture_manifest_path), prefix_tokens=n-49,
                suffix_tokens=49, rule='canonical first(N-49) body tokens plus canonical last49 verified task/chat suffix; middle omitted; no new tokenization',
                limitation='Historical body may end mid-code and this is not independent quality data; no guarantee of16 outputs or draft acceptance.')
            (out / 'tokens.txt').write_text(','.join(map(str, tokens)) + '\n')
            r['input'] = dict(parent_path=str(FIXTURE), parent_sha256=m.sha(FIXTURE),
                              count=n, sha256=m.sha(out / 'tokens.txt'))
            m.require(m.sha(PROFILE) == '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf', 'fixed expert profile')
            files = sorted(p for base in (PACK, MODEL, MTP) for p in base.rglob('*') if p.is_file()) + [PROFILE]
            prior_path = B / 'xe-source-llvm-model8k-root-v1/record.json'
            m.require(m.sha(prior_path) == qualification['receipt_sha256'][str(prior_path)], 'exact retained successful full-hash phase in original failed preflight')
            prior = json.loads(prior_path.read_text())
            m.require(not prior['active'] and not prior['model_launch_attempted'], 'closed CPU-only prior preflight')
            phase = next(c for c in prior['commands'] if c['label'] == 'model-input-hashes')
            m.require(strong_closed_command(phase) and phase['exit_code'] == 0 and
                      prior['hash_helper_sha256'] == m.sha(HERE / 'hash_inputs.py'), 'actual successful full hash phase')
            r['model_inputs'] = prior['model_inputs']
            m.require([str(p.resolve()) for p in files] == [x['path'] for x in r['model_inputs']], 'same complete model file set')
            m.require(all(stat_identity(Path(x['path'])) == x['stat'] for x in r['model_inputs']), 'full-hashed model files unchanged')
            r['model_hash_phase_source'] = dict(path=str(prior_path), sha256=m.sha(prior_path),
                scope='Successful full-file CPU hash phase only; original overall preflight failure remains failed. Exact inode/device/size/mtime/ctime verified for every unchanged file.')
            memory = {x.split(':')[0]: int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:', 'MemAvailable:'))}
            r['initial_memory'] = memory
            m.require(memory['MemAvailable'] >= 80 << 30, 'RAM headroom for known model/native arena')
            cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
            r['cgroup_limits'] = []
            for p in (cgroup, *cgroup.parents):
                if not p.is_relative_to('/sys/fs/cgroup'):
                    continue
                if (p / 'memory.max').exists():
                    maximum, current = (p / 'memory.max').read_text().strip(), int((p / 'memory.current').read_text())
                    r['cgroup_limits'].append(dict(path=str(p), maximum=maximum, current=current))
                    m.require(maximum == 'max' or int(maximum) - current >= 80 << 30, 'cgroup memory headroom')
            state, _ = run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'])
            m.require(state.strip() == 'ActiveState=inactive', 'embedding remains inactive')
            adapter = verified['health']['adapter']
            m.require(m.sha(Path(adapter['path'])) == adapter['sha256'], 'qualified adapter')
            health_env = helper.diagnostic_environment(dict(BASE, ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                STRATA_GPU_PCI='0000:05:00.0', STRATA_EVENT_MARK='barrier',
                SYCL_CACHE_PERSISTENT='0', UR_ADAPTERS_FORCE_LOAD=adapter['path'],
                UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
                EnableDirectSubmission='0'))
            env = dict(BASE, ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                STRATA_GPU_PCI='0000:05:00.0', STRATA_EVENT_MARK='barrier',
                SYCL_CACHE_PERSISTENT='0', UR_ADAPTERS_FORCE_LOAD=adapter['path'],
                UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
                EnableDirectSubmission='0', STRATA_PREFILL_RING='8', STRATA_PREFILL_FIRST='0',
                STRATA_ARENA_HOST_USM='1', STRATA_PREFILL_ISSUER='0', STRATA_MTP_PREFILL_SYNC='1')
            env['STRATA_PREFILL_TRANSFER_COUNTS'] = '1'
            env['STRATA_PRELOAD_KERNELS'] = '0'
            env['STRATA_PREFILL_FUSED'] = '0'
            env['STRATA_PREFILL_NATIVE_DEQUANT_BATCH'] = '0'
            env['STRATA_TRACE'] = '1'
            # The historical detailed N64 attempt exhausted its finite log budget.
            # Keep validation, warning/error logs and progress; health retains detailed APIs.
            env.update(ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOG_CONSOLE='1',
                ZEL_LOADER_LOGGING_LEVEL='warning', ZE_ENABLE_VALIDATION_LAYER='1',
                ZE_ENABLE_PARAMETER_VALIDATION='1',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            m.require('UR_ENABLE_LAYERS' not in env and 'UR_LOG_TRACING' not in env, 'bounded model diagnostic forbids full API trace')
            env=compact_environment(env,compact)
            env=phase_environment(env)
            require_phase_environment(anchor['environment'],env)
            r['diagnostic_environment_policy']='qualified bounded trace1/count1/validation/warning environment plus ONLY STRATA_PREFILL_TIMING1; health unchanged diagnostic'
            r['timing_environment_scope']='source adjacent ordered compute profiling-tag spans and separate host counters; total observer overhead unmeasured; never quiet or exclusive kernel/H2D/criticalpath timing'
            m.require(env.get('STRATA_PREFILL_BF16X2', '0') == '0', 'unchanged explicit/default BF16X2 zero')
            r['environment'] = env
            m.require(env['STRATA_PRELOAD_KERNELS'] == '0', 'diagnostic all-kernel preload sweep disabled')
            argv = [str(binary), '--pack', str(PACK), '--native', str(MODEL / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),
                    '--serve', '--max-context', '262144', '--spec', '4', '--spec-min-p', '0',
                    '--mtp', str(MTP), '--suffix-draft', '0', '--prefill', str(chunk), '--prefill-experts', str(group), '--no-prefill-borrow',
                    '--expert-cache', '128', '--expert-profile', str(PROFILE), '--expert-cache-per-layer',
                    '--pool-workers', '5', '--pcie-frac', '0', '--spec-split', '--adapt-swaps', '0',
                    '--conversation-cache-mib', '0', '--conversation-cache-slots', '4', '--greedy',
                    '--short-read', '0', '--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1',
                    '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
            r['argv'] = argv
            require_same_recipe(anchor['argv'],argv,6144 if n==64 else 8192,8192)
            require_phase_environment(anchor['environment'],env)
            m.require(anchor['model_inputs'] == r['model_inputs'] and
                      anchor['source_pins'] == r['source_pins'] and anchor['binary'] == r['binary'],
                      'same original ON environment/model/source/binary; only fixed chunk argv differs')
            reference_path=Path(reference_pin['path'])
            r['reference']=dict(reference_pin)
            require_phase_environment(reference['environment'],env)
            m.require(reference['input']==r['input'] and reference['model_inputs']==r['model_inputs'] and
                      reference['source_pins']==r['source_pins'] and reference['binary']==r['binary'] and
                      reference['cpu_exclusion_proof']==r['cpu_exclusion_proof'], 'exact same-N untimed diagnostic source/binary/model/input/CPU0 reference')
            r['reference_observer_difference']=dict(environment_delta={'STRATA_PREFILL_TIMING':'1'},
                reference_chunk=6144 if n==64 else 8192,current_chunk=8192,
                note='N64 single63-row sanity compares prior6144ON; N32768 matches same8192ON. Timing is diagnostic instrumentation, not quiet throughput.')

            kernel('kernel-before', ['-b'])
            text, _ = run('kernel-cursor', ['/usr/bin/journalctl', '-k', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'^-- cursor: (.+)$', text, re.M).group(1)
            r['kernel_cursor_before'] = cursor
            probe = verified['health']['binary']
            m.require(m.sha(Path(probe['path'])) == probe['sha256'], 'qualified integer probe')
            r['health_probe'] = probe

            def health(label):
                text, _ = run(label, [probe['path'], '0000:05:00.0'], env=health_env, wall=30, rss=4 << 30)
                expected = ['STATUS 0000:05:00.0 0', 'QUEUE 0000:05:00.0', 'H2D 65536 complete',
                            'KERNEL 16384 complete', 'D2H 65536 complete',
                            'PASS 0000:05:00.0: 1 round, 16384 exact words']
                m.require(text.splitlines()[:6] == expected, 'exact integer ' + label)
                r[label + '_protocol'] = text.splitlines()

            health('health-before')
            kernel('kernel-after-prehealth', ['--after-cursor', cursor])
            admitted_at = time.monotonic()
            stderr = out / 'commands/model.stderr'

            def capture_maps(phase, entry, poll):
                r['mapping_snapshots'][phase] = dict(complete=False, requested=True)
                def partial(value):
                    r['mapping_snapshots'][phase] = value
                    save()
                snapshot = mapping_snapshot(entry, poll, r['mapping_read_budget'], partial)
                r['mapping_snapshots'][phase] = snapshot
                for path, value in snapshot['files'].items():
                    if value['original_runtime']:
                        if path in r['observed_libraries']:
                            keys = ('path', 'canonical_path', 'sha256', 'stat', 'elf_build_id')
                            m.require(all(r['observed_libraries'][path][k] == value[k] for k in keys), 'stable original runtime snapshot')
                        r['observed_libraries'][path] = value
                r['additional_process_mappings'] = {phase: {p:x for p,x in snap.get('files', {}).items() if not x['original_runtime']}
                    for phase,snap in r['mapping_snapshots'].items()}
                save()

            def model_interaction(io, entry):
                original_poll = io.poll

                def poll():
                    original_poll()
                    m.require(not DUMP.exists(), 'new GPU dump: stop owned model')
                io.poll = poll
                m.require(time.monotonic() - admitted_at < 30, 'fresh health admission')
                r['startup'] = []
                while True:
                    line = io.line(120)
                    r['startup'].append(line)
                    m.require(len(r['startup']) < 1000 and not line.startswith('ERR'), 'startup progress')
                    if line.startswith('READY '):
                        r['ready_seen'] = True
                        m.require(line.split()[1] == '262144', 'physical configured context')
                        break
                capture_maps('ready_before_gen', entry, io.poll)
                source_guard('ready')
                r['ready_additional_mapping_validation'] = require_additional_mappings(r['mapping_snapshots']['ready_before_gen'], qualification['qualified_additional_mappings']['ready_before_gen'])
                info = next(x for x in r['startup'] if x.startswith('INFO '))
                settings = dict(x.split('=', 1) for x in info.split()[1:] if '=' in x)
                r['effective_settings'] = settings
                m.require(all(settings.get(k) == v for k, v in dict(context='262144', kv='int8',
                    kv_resident='32768', spec='4', spec_min_p='0.00', expert_slots='144', pool_workers='5', pcie_frac='0.00').items()), 'effective model settings')
                m.require(int(settings['vram_free_mib']) >= 512, 'VRAM startup reserve')
                startup_text = stderr.read_text()
                m.require(startup_text.count('strata serve: the prompt path allocates its own buffers') == 1 and
                          'strata serve: the prompt path borrows' not in startup_text,
                          'actual owned prompt buffers; no borrowing')
                r['memory_admission'] = dict(ready_post_init_free_mib=int(settings['vram_free_mib']),
                    configured_context=262144, requested_chunk=chunk, owned_buffers=True, minimum_ready_free_mib=512,
                    init_completed_before_ready=True, preinit_free_bytes=None, peak_free_bytes=None,
                    limitation='READY admits this bounded diagnostic only; requested extents are not physical peak or allocation-limit evidence')
                exe = Path('/proc', str(entry['pid']), 'exe')
                m.require(exe.resolve() == binary.resolve() and m.sha(exe) == r['binary']['sha256'], 'actual process executable')
                actual = dict(x.decode().split('=', 1) for x in Path('/proc', str(entry['pid']), 'environ').read_bytes().split(b'\0') if b'=' in x)
                m.require(Path('/proc',str(entry['pid']),'cmdline').read_bytes().split(b'\0')[:-1] == [a.encode() for a in argv], 'actual explicit argv including --prefill-experts G')
                m.require(actual == env, 'actual explicit environment')
                r['actual_environment_matches'] = True
                r['startup_runtime_error_lines'] = [line[:2048] for line in stderr.read_text().splitlines() if ERROR.search(line)][:128]
                m.require(not r['startup_runtime_error_lines'], 'no severe startup runtime error')
                m.require(not ERROR.search('\n'.join(r['startup'])), 'no severe startup stdout runtime error')
                r['ready_runtime_mapping_validation'] = validate_runtime(r['observed_libraries'], verified['health']['loaded_libraries'])
                r['request_protocol'] = []
                # Final unchanged strict UR/ZE gate immediately before the only GEN send.
                r['pre_gen_runtime_error_lines'] = [line[:2048] for line in stderr.read_text().splitlines() if ERROR.search(line)][:128]
                m.require(not r['pre_gen_runtime_error_lines'], 'no severe runtime error before GEN')
                r['gen_send_attempted'] = True
                r['mapping_snapshots']['after_gen'] = dict(complete=False, requested=False, availability='pending GEN completion')
                io.send(('GEN 16 ckpt=1 logprobs=5 ' + ','.join(map(str, tokens)) + '\n').encode())
                r['gen_sent'] = True
                while True:
                    line = io.line(120)
                    r['request_protocol'].append(line)
                    m.require(len(r['request_protocol']) < 1000 and not line.startswith('ERR'), 'request progress')
                    if line.startswith('PP '):
                        print(line, flush=True)
                        save()
                    if line.startswith('DONE '):
                        r['request_done_seen'] = True
                        break
                capture_maps('after_gen', entry, io.poll)
                source_guard('done')
                r['done_additional_mapping_validation'] = require_additional_mappings(r['mapping_snapshots']['after_gen'], qualification['qualified_additional_mappings']['after_gen'])
                r['after_gen_runtime_mapping_validation'] = validate_runtime(
                    {p:x for p,x in r['mapping_snapshots']['after_gen']['files'].items() if x['original_runtime']},
                    verified['health']['loaded_libraries'])
                # Finish an already completed request normally before judging its coverage.
                # Output/trace validation below cannot turn a valid early EOS into SIGTERM.
                save()
                normal_quit(io, r)

            def interaction(io, entry):
                owner_poll = io.poll
                try:
                    model_interaction(io, entry)
                except BaseException as failure:
                    r.setdefault('primary_model_failure', repr(failure))
                    r['model_failure_phase'] = 'after_gen' if r['gen_send_attempted'] else ('ready_before_gen' if r.get('ready_seen') else 'startup')
                    if r.get('request_done_seen') and not r['mapping_snapshots'].get('after_gen', {}).get('complete'):
                        try:
                            capture_maps('after_gen', entry, owner_poll)
                        except BaseException as capture_error:
                            r['mapping_snapshots']['after_gen'].update(complete=False, unavailable_error=repr(capture_error))
                    if close_after_gate_failure(io, r, owner_poll, save):
                        return
                    raise

            r['model_launch_attempted'] = True
            save()
            so, se = run('model', argv, env=env, wall=500, interaction=interaction, rss=88 << 30)
            r['model_normal_close'] = True
            r['runtime_error_lines'] = [line for line in (so + se).splitlines() if ERROR.search(line)]
            m.require(not r['runtime_error_lines'], 'no severe model runtime error')
            r['runtime_mapping_validation'] = validate_runtime(r['observed_libraries'], verified['health']['loaded_libraries'])
            m.require(all('/opt/intel/oneapi/' not in p for p in r['observed_libraries']), 'source free runtime only')
            kernel('kernel-after-model', ['--after-cursor', cursor])
            health('health-after')
            kernel('kernel-after-health', ['--after-cursor', cursor])
            r['validation'] = validate(r['request_protocol'], n=n, chunk=chunk)
            r['validation']['diagnostic_only'] = True
            r['validation']['performance_eligible'] = False
            if reference is not None:
                r['reference_comparison_observation'] = output_deltas(reference, r)
                save()  # Preserve actual IDs/LP5/deltas before any strict continuation refusal.
            r['diagnostic_request_timing']=dict(prefill_ms=r['validation']['prompt_ms'],cold_decode16_ms=r['validation']['decode_ms'],
                input_tokens=n,actual_batched_positions=n-1,generated_tokens=r['validation']['generated_tokens'],performance_eligible=False,total_observer_overhead_measured=False)
            fill_matches = re.findall(r'^strata generate: pre-filled (\d+) of (\d+) slots from the profile .*; slot (\d+) verified$', se, re.M)
            resident_matches = re.findall(r'^strata generate: expert residency table: (\d+) resident experts$', se, re.M)
            m.require(len(fill_matches) == 1 and fill_matches[0][:2] == ('144', '144') and resident_matches == ['144'], 'actual full per-layer profile occupancy')
            m.require(r['effective_settings']['expert_slots'] == '144', 'actual cache capacity matches fill and residency')
            r['profile_fill_validation'] = dict(filled=144, capacity=144, residents=144, verified_slot=int(fill_matches[0][2]), per_layer_quota=3, geometry_provenance='Same full48x512 unique ranking/model; per-layer slot ranges and total occupancy imply each of48 layers fills its3 slots. Only one slot readback is the existing byte check, not all144 byte checks.')
            r['trace_validation']=validate_trace(se,n=n,chunk=chunk)
            r['transfer_ledger']=validate_transfers(se,**expectations)
            common=dict(enabled=True,h2d_validate=validate_transfers,h2d_expectations=expectations,
                n_experts=512,route_k=10,moe_layers=set(range(48)),nonresident_population={layer:509 for layer in range(48)})
            m.require('[prefill-native-dequant-group-' not in se, 'nativebatch0 diagnostic absence')
            r['route_census']=validate_routes(se,**common)
            m.require(all(x['mmq_enabled']==0 and x['plan_fallback']==1 for x in r['route_census']['rows']), 'all48layers pure fallback')
            r['phase_observer']=validate_phase(se,batched_positions=n-1)
            r['timer_overhead']=validate_overhead(se,timing_enabled=True)
            m.require(r['validation']['generated_tokens'] == 16, 'strict16 coverage after normal QUIT; early EOS remains normally closed but fails qualification')
            m.require(not re.search(r'prompt chunk (?:auto:|\d+ ->)', se), 'no automatic or reduced chunk recipe')
            r['actual_chunk_validation'] = dict(requested=chunk, actual=chunk, no_borrow=True,
                windows=[dict(offset=o, rows=t, mode=expected_mode[o], effective_ring=16 if t >= 2048 else 8)
                         for o,t in windows(n, chunk)], owned_ring=16,
                provenance='Serve initializes fixed8192 beforeGEN, explicit actual argv/env and pinned no-borrow source plus exact PP/trace/48layer route+transfer full/tail geometry. N64 tail63 routed ring8; N32768 three8192 plus8191 fullstream ring16. Requested extents are not physical peak evidence.')
            if reference is not None:
                m.require({k:v for k,v in reference['effective_settings'].items() if k != 'vram_free_mib'} == {k:v for k,v in r['effective_settings'].items() if k != 'vram_free_mib'}, 'reference/current exact actual model settings; each observed free-memory reserve checked independently')
                m.require(r['reference_comparison_observation']['strict_continuation_passed'],
                          'observed strict16 IDs/LP5 differ; deltas retained; no numerical tolerance or retry authorized')
                r['reference_comparison'] = r['reference_comparison_observation']
                m.require(m.sha(Path(r['reference']['path'])) == r['reference']['sha256'], 'immutable same-N/chunk diagnostic reference remains stable')
            m.require(all(stat_identity(Path(x['path'])) == x['stat'] for x in r['model_inputs']), 'model inputs unchanged')
            m.require(all(m.sha(W / p) == digest for p, digest in r['source_pins'].items()), 'source stable')
            m.require(m.sha(binary) == r['binary']['sha256'], 'binary stable')
            m.require(m.sha(HERE / 'protocol.py') == r['protocol_sha256'] and m.sha(HERE / 'hash_inputs.py') == r['hash_helper_sha256'], 'helpers stable')
            m.require(m.sha(HERE / 'runtime.py') == r['runtime_helper_sha256'], 'runtime helper stable')
            m.require(all(m.sha(HERE/n) == h for n,h in r['diagnostic_helper_pins'].items()), 'diagnostic parsers stable')
            m.require(m.sha(OWNER) == r['owner_sha256'], 'owner stable')
            m.require(m.sha(__file__) == r['controller_sha256'] and
                      m.sha(HELPER) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'controller/diagnostic helper stable')
            m.require(m.sha(args.qualification) == r['qualification_sha256'], 'Main qualification manifest stable')
            m.require(all(m.sha(Path(x['path'])) == x['sha256'] for x in r['observed_libraries'].values()), 'mapped runtime identities stable')
            m.require(all(m.sha(Path(p)) == h for p,h in r['receipt_pins'].items()), 'prerequisite receipts stable')
            m.require(m.sha(review_path) == r['source_applied_main_review_sha256'] and m.sha(controller_review_path) == r['controller_main_review_sha256'], 'reviews stable')
            m.require(r['validation']['mtp_counts'][1] > 0, 'actual speculative draft activity')
            source_guard('final')
            m.require(all(m.sha(Path(pin['path'])) == pin['sha256'] for pin in qualification['six_case_receipts'].values()),
                      'all original six-case prerequisite bytes stable')
            m.require(all(m.sha(Path(pin['path'])) == pin['sha256'] for pin in qualification['three_case_receipts'].values()), 'all three diagnostic prerequisite bytes stable')
            m.require(all(m.sha(Path(path)) == digest for path,digest in r['prior_case_pins'].items()), 'all prior phase sanity receipts stable')
            m.require(m.sha(predecessor) == r['predecessor']['sha256'], 'precedessor strong-close receipt stable')
            r['complete'] = True
        except BaseException as e:
            r['error'] = repr(e)
        finally:
            if cursor and helper and owner.active is None:
                try:
                    r['new_fault_messages'] = kernel('kernel-final', ['--after-cursor', cursor])
                    r['kernel_gate_passed'] = True
                except BaseException as e:
                    r['kernel_gate_error'] = repr(e)
            r['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            r['active'] = owner.active is not None or guard_owner.active is not None
            r['passed'] = bool(r['complete'] and r.get('kernel_gate_passed') and r['boot_unchanged']
                               and not r['active'] and not r.get('error') and not r.get('kernel_gate_error')
                               and all(strong_closed_command(c) and c['exit_code'] == 0 for c in r['commands'] + r['guard_commands']))
            r['performance_eligible'] = False
            if 'validation' in r:
                r['validation']['performance_eligible'] = False
            r['finished_utc'] = m.utc()
            save()
        print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'kernel_gate_error', 'model_launch_attempted')}))
        return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
