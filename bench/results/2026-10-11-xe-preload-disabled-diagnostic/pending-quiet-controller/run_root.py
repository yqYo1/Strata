"""SOURCE ONLY candidate: finite fresh-process quiet paired timing observations."""
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

from protocol import validate, validate_trace
from runtime import validate_runtime
from transfer_ledger import validate as validate_transfers
from timer_overhead import validate as validate_overhead
from route_census import validate as validate_routes
from batch_census import validate as validate_batch

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
HERE = Path(__file__).resolve().parent
LLVM = W / '.tools/intel-llvm-v711/install'
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
    parser.add_argument('--input-tokens', type=int, choices=[32768], required=True)
    parser.add_argument('--group', type=int, choices=[16], required=True)
    parser.add_argument('--sample-index', type=int, choices=range(100), required=True)
    parser.add_argument('--batch', choices=['off','on'], required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--qualification', type=Path, default=HERE/'qualification.pending.json')
    parser.add_argument('--qualification-sha256', default='PENDING_MAIN_EXACT_SHA256')
    args = parser.parse_args()
    if (args.batch == 'on') != (args.reference is not None):
        parser.error('ON requires immutable OFF record.json; OFF forbids --reference')
    m.require(re.fullmatch('[0-9a-f]{64}', args.qualification_sha256) and
              m.sha(args.qualification) == args.qualification_sha256, 'actual Main exact qualification manifest hash')
    qualification = json.loads(args.qualification.read_text())
    m.require(qualification['status'] == 'passed' and qualification['main_reviewed'] is True and
              qualification['base_revision'] == 'fa3e20e842fbed6986a69bade8afe3edabedee07', 'actual Main qualification, no pending launch')
    m.require(re.fullmatch('[0-9a-f]{40}', qualification['published_head']) and
              re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', qualification['campaign']), 'actual Main published HEAD and immutable campaign')
    indices = qualification['sample_indices']
    m.require(isinstance(indices, list) and 3 <= len(indices) <= 100 and
              all(type(i) is int and 0 <= i < 100 for i in indices) and len(set(indices)) == len(indices) and
              args.sample_index in indices, 'Main planned at least three distinct paired samples')
    n, group, batch = args.input_tokens, args.group, args.batch
    kind = 'quiet-native-batch-' + batch
    expected_mode = {offset: ('full_nonresident_stream' if min(6144,n-1-offset) >= 2048 else 'routed') for offset in range(0,n-1,6144)}
    reference = None
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample_root = B / 'xe-native-dequant-prefill-quiet-timing-root-v1' / qualification['campaign'] / f'sample-{args.sample_index:03d}'
        out = sample_root / f'n{n}-g{group}-{batch}'
        m.require(not out.exists(), 'immutable fresh diagnostic output')
        out.mkdir(mode=0o700, parents=True)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        r = dict(active=True, complete=False, passed=False, commands=[],
                 started_utc=m.utc(), boot_id=boot, controller_sha256=m.sha(__file__),
                 protocol_sha256=m.sha(HERE / 'protocol.py'), owner_sha256=m.sha(OWNER),
                 hash_helper_sha256=m.sha(HERE / 'hash_inputs.py'), runtime_helper_sha256=m.sha(HERE / 'runtime.py'),
                 mode='free', candidate_revision=qualification['base_revision'], source_uncommitted=False, input_tokens=n, group=group, batch=batch,
                 historical_build_base=qualification['base_revision'], published_head=qualification['published_head'],
                 campaign=qualification['campaign'], sample_index=args.sample_index, planned_sample_indices=indices,
                 qualification_sha256=args.qualification_sha256, qualification_path=str(args.qualification),
                 mapping_snapshots={}, mapping_read_budget=dict(bytes=0), gen_sent=False,
                 gen_send_attempted=False,
                 adopted=False, upstream_pristine=False, performance_eligible=False, diagnostic_kind=kind,
                 full_model_quality_qualified=False, full262144_qualified=False,
                 reset_executed=False, retry_count=0, model_launch_attempted=False,
                 raw_text_log_budget_bytes=64 << 20, total_artifact_budget_bytes=96 << 20,
                 whole_wall_budget_seconds=900, model_wall_budget_seconds=500,
                 model_rss_budget_bytes=88 << 30, observed_libraries={},
                 scope='One finite fresh-process quiet timing observation in a Main-pinned >=3-pair campaign. Exactly32768 tokens/G16, FUSED0, issuer0, physical262144, chunk6144, ring8 (fullstream16/routed8). Strict16/T/LP5; no full-model, 256K or performance qualification.')

        def save():
            (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

        owner = m.Owner(out / 'commands', save)
        r['commands'] = owner.commands
        save()
        begin = time.monotonic()
        cursor = None
        helper = None

        def run(label, argv, env=BASE, wall=20, interaction=None, rss=2 << 30):
            remaining = 900 - (time.monotonic() - begin)
            m.require(remaining > 0, 'whole diagnostic deadline')
            e, so, se = owner.run(label, argv, env, W, interaction=interaction,
                                  wall=min(wall, remaining), cpu=int(wall) * 2,
                                  text_cap=64 << 20, total_cap=96 << 20,
                                  rss_cap=rss)
            if label == 'model':
                r['model_normal_close'] = bool(m.closed(e) and e['exit_code'] == 0)
                r.setdefault('model_close', {}).update(exit_code=e.get('exit_code'),
                    owner_error=e.get('error'), owner_errors=e.get('errors'), cleanup=e.get('cleanup'),
                    normal=r['model_normal_close'], forced=bool(e.get('cleanup')))
                if r.get('primary_model_failure'):
                    raise RuntimeError(r['primary_model_failure'])
            m.require(m.closed(e) and e['exit_code'] == 0, 'normal closed ' + label)
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
            def receipt(path, same_boot=False, release_cli_rejections=False, controller_cli_rejections=False):
                if str(path) in qualification['receipt_sha256']:
                    m.require(m.sha(path) == qualification['receipt_sha256'][str(path)], 'exact Main prerequisite receipt hash')
                else:
                    m.require(path.resolve() == (sample_root/f'n{n}-g{group}-off'/'record.json').resolve(), 'Main must pin every prerequisite receipt; only current paired OFF may be newly produced')
                v = json.loads(path.read_text())
                m.require(v['passed'] is True and v['complete'] is True and v['active'] is False and
                          v.get('commands') and all(m.closed(c) for c in v['commands']), 'actual closed PASS receipt ' + str(path))
                allowed_rejections = {'invalid-max-new-nan': ['--max-new','nan'],
                    'invalid-max-new-9223372036854775808': ['--max-new','9223372036854775808'],
                    'invalid-max-context-262144x': ['--max-context','262144x']} if release_cli_rejections else {}
                if controller_cli_rejections:
                    m.require(not release_cli_rejections, 'distinct expected rejection recipe')
                    cases = [['--input-tokens','32768','--group','16','--sample-index','0','--batch','on'],
                             ['--input-tokens','32768','--group','16','--sample-index','0','--batch','off','--reference','/tmp/unused-no-open'],
                             ['--input-tokens','65','--group','16','--sample-index','0','--batch','off'],
                             ['--input-tokens','32768','--group','65','--sample-index','0','--batch','off']]
                    allowed_rejections = {'invalid-cli-'+str(i): [str(HERE/'run_root.py'),*args] for i,args in enumerate(cases)}
                for c in v['commands']:
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

            review_path = B / 'xe-optional-preload-control-applied-main-review-v1.json'
            m.require(m.sha(review_path) == qualification['source_review_sha256'], 'Main exact applied source-review hash')
            review = json.loads(review_path.read_text())
            m.require(review['main_reviewed'] is True and review['applied'] is True and
                      review['base_revision'] == r['candidate_revision'], 'Main applied preload-control source review V1')
            r['source_applied_main_review_sha256'] = m.sha(review_path)
            m.require(review['source_pins'] == qualification['source_review_pins'], 'exact Main pending source-review pins resolved')
            proofs = {}
            for mode, prefix in [('free','xe-e32-free'), ('contrib-llvm','xe-e32-contrib-llvm')]:
                for flavor, version in [('release',22),('asan',19)]:
                    name = prefix + ('-full-build-root-v' if flavor == 'release' else '-asan-full-build-root-v') + str(version)
                    v = receipt(B / name / 'record.json', release_cli_rejections=(flavor == 'release'))
                    m.require(v['mode'] == mode and v['source_base'] == r['candidate_revision'] and
                              len(v['source_pins']) == 38 and '100% tests passed, 0 tests failed' in v['cpu_test_output'], 'actual 38-pin both-mode release/hostASAN CPU PASS')
                    m.require(v['binary'] == qualification['qualified_build_binaries'][mode + '-' + flavor] and
                              m.sha(Path(v['binary']['path'])) == v['binary']['sha256'], 'exact current qualified build binary ' + mode + '-' + flavor)
                    proofs[mode + '-' + flavor] = v
            release = proofs['free-release']
            r['source_pins'] = release['source_pins']
            m.require(r['source_pins'] == qualification['production_source_pins'] and
                      release['binary'] == qualification['free_release_binary'], 'Main exact production source and binary qualification')
            m.require(r['source_pins']['src/core/device.cpp'] == '3e7523a2947af7c82d41fc9a7653f4f9aad0fa3cbb062999de19553659e0ec27', 'applied exact preload-control candidate')
            m.require(all(v['source_pins'] == r['source_pins'] for v in proofs.values()), 'same38 sources in both release and ASAN modes')
            m.require(all(m.sha(W/p) == h for p,h in r['source_pins'].items()), 'exact current38 production sources')
            fixed = {'src/prefill/prefill.cpp':'c5cfa727302d334e2f6a8aaad8c1bf59e7d581dd8c2cec2af0d031faf807272d',
                     'src/kernels/xe/iq_kernels.cpp':'75f2ac1dfdb337c424bba19e62c34159a1bde547a776b9b5d6cf7975dfc29d50',
                     'include/strata/kernels/iq_kernels.hpp':'61513ccb71831d1065bd25829541c508adf3ae58e1e3df6d3bc0c9681cb54f6d',
                     'src/core/device.cpp':'3e7523a2947af7c82d41fc9a7653f4f9aad0fa3cbb062999de19553659e0ec27'}
            m.require(review['source_pins'] == fixed and all(r['source_pins'][p] == h for p,h in fixed.items()), 'fixed caller and component API source identity')
            m.require(all(v['source_patch_sha256'] == release['source_patch_sha256'] for v in proofs.values()), 'same reviewed working diff')
            r['working_diff_sha256'] = release['source_patch_sha256']
            m.require(r['working_diff_sha256'] == qualification['source_patch_sha256'], 'Main exact qualified working diff')
            m.require(m.sha(B/'xe-e32-free-full-build-root-v22/source.patch') == r['working_diff_sha256'], 'retained working diff bytes')
            head, _ = run('source-head', ['/usr/bin/git','rev-parse','HEAD'])
            patch, _ = run('source-working-diff', ['/usr/bin/git','diff','HEAD'])
            status, _ = run('source-tracked-status', ['/usr/bin/git','status','--porcelain=v1','--untracked-files=no'])
            m.require(head.strip() == r['published_head'] and patch == '' and status == '', 'actual Main published HEAD and clean tracked working tree')
            tracked, _ = run('source38-tracked', ['/usr/bin/git','ls-files','-z','--',*sorted(r['source_pins'])])
            m.require(set(tracked.split('\0')[:-1]) == set(r['source_pins']) and len(tracked.split('\0')[:-1]) == 38,
                      'all38 qualified content paths tracked by the clean published HEAD')
            r['published_working_diff_sha256'] = m.sha(out/'commands/source-working-diff.stdout')
            publication_path = Path(qualification['publication_review_path'])
            m.require(m.sha(publication_path) == qualification['publication_review_sha256'], 'Main exact publication review hash')
            publication = json.loads(publication_path.read_text())
            m.require(publication['main_reviewed'] is True and publication['published_head'] == r['published_head'] and
                      publication['historical_build_base'] == r['historical_build_base'] and
                      publication['historical_build_patch_sha256'] == r['working_diff_sha256'] and
                      publication['production_source_pins'] == r['source_pins'] and
                      publication['clean_tracked_working_tree'] is True, 'Main published source content equals historical qualified38 content')
            r['publication_review'] = dict(path=str(publication_path), sha256=m.sha(publication_path))
            lint = receipt(B/'xe-e32-lints-baseline-compare-root-v17/record.json')
            m.require(lint['source_pins'] == qualification['lint_source_pins'] and
                      all(r['source_pins'][p] == h for p,h in lint['source_pins'].items()) and
                      lint['source_review_sha256'] == m.sha(review_path), 'actual lintv17 reviewed device.cpp')
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
            cpu = receipt(B/'xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v1/record.json', controller_cli_rejections=True)
            r['diagnostic_helper_pins'] = cpu['source_pins']
            m.require(set(r['diagnostic_helper_pins']) == {'run_root.py','protocol.py','runtime.py','transfer_ledger.py','route_census.py','batch_census.py','timer_overhead.py','hash_inputs.py','fixtures.py'} and
                      all(m.sha(HERE/name) == h for name,h in r['diagnostic_helper_pins'].items()), 'actual Main controller/selfcheck exact helper pins')
            controller_review_path = B/'xe-native-dequant-prefill-quiet-timing-controller-main-review-v1.json'
            m.require(m.sha(controller_review_path) == qualification['controller_review_sha256'], 'Main exact controller-review hash')
            controller_review = json.loads(controller_review_path.read_text())
            m.require(controller_review['main_reviewed'] is True and controller_review['source_pins'] == r['diagnostic_helper_pins'], 'Main reviewed controller exact pins')
            r['controller_main_review_sha256'] = m.sha(controller_review_path)
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
                prefill_sha256=fixed['src/prefill/prefill.cpp'],
                old_model_capability_field_present='capabilities' in prior_model,
                provenance='actual independent HOST_USM1 ATOMIC_HOST_USM0 ZE_HOST_CAPS3; pinned host_handoff_concurrent false AND cpu_share_on constraint; old model record does not itself expose capabilities')
            m.require(int(prior_model['effective_settings']['vram_free_mib']) >= 512, 'previous actual READY reserve')
            r['memory_plan'] = dict(previous_path=str(prior_model_path), previous_sha256=m.sha(prior_model_path),
                previous_free_mib=int(prior_model['effective_settings']['vram_free_mib']), group=group,
                dq_bytes_per_expert=2*(1280*2560+2560*640), previous_group=16,
                group_buffer_delta_bytes=(group-16)*2*(1280*2560+2560*640),
                estimated_ready_free_mib=int(prior_model['effective_settings']['vram_free_mib'])-(group-16)*2*(1280*2560+2560*640)/(1<<20),
                minimum_ready_free_mib=512, requested_ring=8, fullstream_ring=16, routed_ring=8,
                scope='G-dependent allocations judged at actual READY before GEN. G64 may fail preflight; reserve threshold remains512MiB.')
            m.require(r['memory_plan']['estimated_ready_free_mib'] >= 512, 'conservative G buffer reserve preflight; no model launch if below512MiB')
            layout_path = B/'xe-prefill-native-transfer-layout-v1.json'
            m.require(m.sha(layout_path) == 'c9b40d122d19f277280560a495ad0a1c6ce98376ef0660c1f75e9b3026bd642b', 'independent model metadata')
            layout = json.loads(layout_path.read_text())
            m.require(m.sha(Path(layout['layout_path'])) == layout['layout_sha256'] and layout['layers'] == 48 and
                      layout['experts_per_layer'] == 512 and [x['layer'] for x in layout['expected_layers']] == list(range(48)), 'full native48layer geometry')
            r['layout_reference_sha256'] = m.sha(layout_path)
            expected_layers = {x['layer']:{k:x[k] for k in ('blob_bytes','gu_type','down_type')} for x in layout['expected_layers']}
            expectations = dict(expected_device=0,expected_stage=[0,48],expected_pos0=0,expected_n=n-1,
                expected_chunk_limit=6144,expected_mode=expected_mode,expected_issuer='inline',expected_format='native_gguf',expected_layers=expected_layers)
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
            prior = json.loads(prior_path.read_text())
            m.require(not prior['active'] and not prior['model_launch_attempted'], 'closed CPU-only prior preflight')
            phase = next(c for c in prior['commands'] if c['label'] == 'model-input-hashes')
            m.require(m.closed(phase) and phase['exit_code'] == 0 and
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
            env['STRATA_PRELOAD_KERNELS'] = '0'
            env['STRATA_PREFILL_FUSED'] = '0'
            env['STRATA_PREFILL_NATIVE_DEQUANT_BATCH'] = '1' if batch == 'on' else '0'
            # Presence-based switches must be absent; assigning "0" enables them.
            env.update(ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOG_CONSOLE='1',
                ZEL_LOADER_LOGGING_LEVEL='warning',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            quiet_absent = ('STRATA_TRACE', 'STRATA_PREFILL_TRANSFER_COUNTS', 'STRATA_PREFILL_TIMING',
                'STRATA_DRAFT_TIMING', 'UR_ENABLE_LAYERS', 'UR_LOG_TRACING',
                'ZE_ENABLE_VALIDATION_LAYER', 'ZE_ENABLE_PARAMETER_VALIDATION', 'ZE_ENABLE_TRACING_LAYER')
            m.require(all(k not in env for k in quiet_absent), 'quiet timing has trace/validation/census/timer switches absent')
            r['quiet_switches_absent'] = list(quiet_absent)
            r['diagnostic_environment_policy'] = 'quiet model: warning/error logs only; trace/API tracing/validation/transfer census/GPU phase timer absent; integer health remains detailed outside measured request'
            r['timing_environment_scope'] = 'Engine DONE prefill/decode phase times per fresh process. Mapping hashes before GEN/after DONE outside request, with admitted CPU/I/O/cache effects; performance eligibility awaits Main review.'
            r['environment'] = env
            m.require(env['STRATA_PRELOAD_KERNELS'] == '0', 'diagnostic all-kernel preload sweep disabled')
            argv = [str(binary), '--pack', str(PACK), '--native', str(MODEL / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),
                    '--serve', '--max-context', '262144', '--spec', '4', '--spec-min-p', '0',
                    '--mtp', str(MTP), '--suffix-draft', '0', '--prefill', '6144', '--prefill-experts', str(group), '--no-prefill-borrow',
                    '--expert-cache', '128', '--expert-profile', str(PROFILE), '--expert-cache-per-layer',
                    '--pool-workers', '5', '--pcie-frac', '0', '--spec-split', '--adapt-swaps', '0',
                    '--conversation-cache-mib', '0', '--conversation-cache-slots', '4', '--greedy',
                    '--short-read', '0', '--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1',
                    '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
            r['argv'] = argv
            diagnostic_pair = {}
            for arm in ('off', 'on'):
                pin = qualification['diagnostic_pair'][arm]
                diagnostic_path = Path(pin['path']).resolve(strict=True)
                m.require(diagnostic_path == (B/'xe-native-dequant-prefill-model-root-v6'/f'n{n}-g{group}-{arm}'/'record.json').resolve() and
                          m.sha(diagnostic_path) == pin['sha256'], 'exact Main closed32K diagnostic pin ' + arm)
                d = receipt(diagnostic_path, True)
                m.require(d['input_tokens'] == n and d['group'] == group and d['batch'] == arm and
                          d['source_pins'] == r['source_pins'] and d['working_diff_sha256'] == r['working_diff_sha256'] and
                          d['binary'] == r['binary'] and d['input'] == r['input'] and d['model_inputs'] == r['model_inputs'] and
                          d['argv'] == argv and d['actual_environment_matches'] is True and
                          d['model_normal_close'] is True and d['kernel_gate_passed'] is True and d['boot_unchanged'] is True and
                          d['validation']['generated_tokens'] == 16 and d['performance_eligible'] is False and
                          d['full_model_quality_qualified'] is False and d['full262144_qualified'] is False and
                          not d.get('runtime_error_lines') and not d.get('startup_runtime_error_lines') and not d.get('pre_gen_runtime_error_lines'),
                          'actual closed strict32K caller/numerical diagnostic, never historical performance qualification')
                m.require(validate(d['request_protocol'], n=n) == d['validation'] and
                          d['controller_sha256'] == qualification['diagnostic_controller_sha256'] and
                          d['source_applied_main_review_sha256'] == r['source_applied_main_review_sha256'] and
                          all(d['diagnostic_helper_pins'][name] == m.sha(HERE/name) for name in r['diagnostic_helper_pins'] if name != 'run_root.py') and
                          d['cpu_exclusion_proof'] == r['cpu_exclusion_proof'] and
                          d['health-before_protocol'][:6] == d['health-after_protocol'][:6] and
                          d['health_probe'] == verified['health']['binary'], 'independently validated diagnostic protocol/controller/CPU0/health')
                m.require(d['runtime_mapping_validation']['all_observed_exactly_qualified'] is True and
                          validate_runtime(d['observed_libraries'], verified['health']['loaded_libraries']) == d['runtime_mapping_validation'] and
                          all(m.sha(Path(x['path'])) == x['sha256'] for x in d['observed_libraries'].values()), 'stable actual diagnostic runtime mappings')
                rows = d['route_census']['rows']
                m.require(len(rows) == len(expected_mode)*48 and [(x['offset'],x['layer']) for x in rows] ==
                          [(offset,layer) for offset in expected_mode for layer in range(48)] and
                          all(x['mmq_enabled'] == 0 for x in rows) and
                          d['transfer_expectations'] == r['transfer_expectations'] and
                          d['trace_validation']['chunk_start_positions'] == list(expected_mode), 'actual closed32K six-chunk caller/census qualification')
                diagnostic_pair[arm] = d
            diag_env = diagnostic_pair['off']['environment']
            m.require(diag_env == qualification['diagnostic_off_environment'] and
                      diagnostic_pair['on']['environment'] == dict(diag_env, STRATA_PREFILL_NATIVE_DEQUANT_BATCH='1') and
                      diag_env['STRATA_PREFILL_NATIVE_DEQUANT_BATCH'] == '0' and
                      diag_env['STRATA_PREFILL_FUSED'] == '0' and diag_env['STRATA_PRELOAD_KERNELS'] == '0', 'Main exact diagnostic pair environment differs only by batch')
            def output_lines(v):
                return [x for x in v['request_protocol'] if x.startswith(('T ', 'LP '))]
            m.require(output_lines(diagnostic_pair['off']) == output_lines(diagnostic_pair['on']) and
                      diagnostic_pair['on']['reference_comparison']['output_ids_equal'] is True and
                      diagnostic_pair['on']['reference_comparison']['selected_logprob_top5_equal'] is True,
                      'actual closed32K strict16/T/LP5 numerical pair qualification')
            m.require(qualification['diagnostic_controller_sha256'] == 'fe4f7a476583ae5c5645892ba3e1fbd17ded776904ddf93fb782ad91b6d40da6',
                      'immutable Main v8 controller identity for32K diagnostic prerequisites')
            diagnostic_review_path = Path(qualification['diagnostic_pair_review_path'])
            m.require(m.sha(diagnostic_review_path) == qualification['diagnostic_pair_review_sha256'], 'Main exact32K diagnostic pair review hash')
            diagnostic_review = json.loads(diagnostic_review_path.read_text())
            m.require(diagnostic_review['main_reviewed'] is True and diagnostic_review['caller_numerical_passed'] is True and
                      diagnostic_review['diagnostic_pair'] == qualification['diagnostic_pair'] and
                      diagnostic_review['production_source_pins'] == r['source_pins'] and
                      diagnostic_review['performance_qualified'] is False, 'Main actual32K diagnostic caller review; separate from timing qualification')
            r['diagnostic_qualification'] = dict(pair=qualification['diagnostic_pair'], review_path=str(diagnostic_review_path),
                review_sha256=m.sha(diagnostic_review_path), caller_numerical_passed=True, performance_qualified=False)
            if batch == 'on':
                reference_path = args.reference.resolve(strict=True)
                expected_reference = (sample_root/f'n{n}-g{group}-off'/'record.json').resolve()
                m.require(reference_path == expected_reference, 'exact immutable OFF root for same N/G')
                reference = receipt(reference_path, True)
                m.require(reference['batch'] == 'off' and reference['input_tokens'] == n and reference['group'] == group and
                          reference['campaign'] == r['campaign'] and reference['sample_index'] == r['sample_index'] and
                          reference['published_head'] == r['published_head'] and
                          reference['mode'] == 'free' and reference['source_pins'] == r['source_pins'] and
                          reference['working_diff_sha256'] == r['working_diff_sha256'] and reference['binary'] == r['binary'] and
                          reference['input'] == r['input'] and reference['model_inputs'] == r['model_inputs'] and reference['argv'] == argv and
                          reference['model_normal_close'] is True and reference['kernel_gate_passed'] is True and
                          reference['boot_unchanged'] is True and reference['validation']['generated_tokens'] == 16,
                          'qualified fresh OFF exact source/binary/input/settings and normal owner close')
                off_env = dict(env, STRATA_PREFILL_NATIVE_DEQUANT_BATCH='0')
                m.require(reference['environment'] == off_env and off_env['STRATA_PREFILL_FUSED'] == '0' and
                          reference['actual_environment_matches'] is True and reference['cpu_exclusion_proof'] == r['cpu_exclusion_proof'],
                          'independent FUSED0 and CPU0 actual OFF settings')
                m.require(reference['runtime_mapping_validation']['all_observed_exactly_qualified'] is True and
                          all(m.sha(Path(x['path'])) == x['sha256'] for x in reference['observed_libraries'].values()), 'stable actual reference runtimes')
                m.require(validate(reference['request_protocol'], n=n) == reference['validation'], 'independently revalidate preserved OFF protocol')
                m.require(reference['controller_sha256'] == r['controller_sha256'] and
                          reference['diagnostic_helper_pins'] == r['diagnostic_helper_pins'] and
                          reference['source_applied_main_review_sha256'] == r['source_applied_main_review_sha256'] and
                          reference['controller_main_review_sha256'] == r['controller_main_review_sha256'] and
                          reference['transfer_expectations'] == r['transfer_expectations'], 'exact reviewed OFF controller, helpers, and transfer contexts')
                m.require(reference['health-before_protocol'][:6] == reference['health-after_protocol'][:6] and
                          reference['health_probe'] == verified['health']['binary'], 'OFF fresh before/after qualified integer health')
                m.require(reference['diagnostic_qualification'] == r['diagnostic_qualification'] and
                          reference['quiet_switches_absent'] == r['quiet_switches_absent'], 'same exact reviewed32K diagnostic qualification and quiet arm policy')
                r['reference'] = dict(path=str(reference_path), sha256=m.sha(reference_path))

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
                observation_started = time.monotonic()
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
                r.setdefault('mapping_observation_wall_seconds', {})[phase] = time.monotonic() - observation_started
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
                info = next(x for x in r['startup'] if x.startswith('INFO '))
                settings = dict(x.split('=', 1) for x in info.split()[1:] if '=' in x)
                r['effective_settings'] = settings
                m.require(all(settings.get(k) == v for k, v in dict(context='262144', kv='int8',
                    kv_resident='32768', spec='4', spec_min_p='0.00', expert_slots='144', pool_workers='5', pcie_frac='0.00').items()), 'effective model settings')
                m.require(int(settings['vram_free_mib']) >= 512, 'VRAM startup reserve')
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
                    if line.startswith('DONE '):
                        r['request_done_seen'] = True
                        break
                capture_maps('after_gen', entry, io.poll)
                r['after_gen_runtime_mapping_validation'] = validate_runtime(
                    {p:x for p,x in r['mapping_snapshots']['after_gen']['files'].items() if x['original_runtime']},
                    verified['health']['loaded_libraries'])
                # Finish an already completed request normally before judging its coverage.
                # Output/trace validation below cannot turn a valid early EOS into SIGTERM.
                save()
                r.setdefault('model_close', {}).update(quit_requested=True, quit_reason='completed GEN', quit_send_outcome='pending', quit_exit_outcome='pending')
                io.send(b'QUIT\n')
                r['model_close']['quit_send_outcome'] = 'sent'
                io.exit(30)
                r['model_close']['quit_exit_outcome'] = 'exit observed'

            def interaction(io, entry):
                owner_poll = io.poll
                try:
                    model_interaction(io, entry)
                except BaseException as failure:
                    r.setdefault('primary_model_failure', repr(failure))
                    r['model_failure_phase'] = 'after_gen' if r['gen_send_attempted'] else ('ready_before_gen' if r.get('ready_seen') else 'startup')
                    if r['gen_send_attempted'] and not r['mapping_snapshots'].get('after_gen', {}).get('complete'):
                        try:
                            capture_maps('after_gen', entry, owner_poll)
                        except BaseException as capture_error:
                            r['mapping_snapshots']['after_gen'].update(complete=False, unavailable_error=repr(capture_error))
                    if r.get('ready_seen') and (not r['gen_send_attempted'] or r.get('request_done_seen')):
                        close = r.setdefault('model_close', {})
                        close.update(quit_requested=True, quit_reason=('completed GEN post-completion gate failure' if r.get('request_done_seen') else 'READY pre-GEN gate failure'),
                                     quit_send_outcome='pending', quit_exit_outcome='pending')
                        save()
                        # Preserve the fault failure. Original owner poll keeps identity,
                        # wall/resource checks while allowing finite protocol shutdown.
                        io.poll = owner_poll
                        try:
                            io.send(b'QUIT\n', seconds=5)
                            close['quit_send_outcome'] = 'sent'
                        except BaseException as quit_error:
                            close['quit_send_outcome'] = repr(quit_error)
                        try:
                            io.exit(10)
                            close['quit_exit_outcome'] = 'exit observed'
                        except BaseException as exit_error:
                            close['quit_exit_outcome'] = repr(exit_error)
                        save()
                        if close['quit_exit_outcome'] == 'exit observed':
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
            r['validation'] = validate(r['request_protocol'], n=n)
            r['validation']['diagnostic_only'] = True
            r['validation']['performance_eligible'] = False
            r['diagnostic_request_timing'] = dict(prefill_tps=1000.0 * r['validation']['prompt_tokens'] / r['validation']['prompt_ms'], cold_decode_tps=1000.0 * r['validation']['generated_tokens'] / r['validation']['decode_ms'], prefill_ms=r['validation']['prompt_ms'], cold_decode_ms=r['validation']['decode_ms'], input_tokens=n, actual_batched_positions=n-1, generated_tokens=r['validation']['generated_tokens'], initialized_independent_process=True, restore_measured=False, settings_search=False)
            r['timing_sample'] = dict(sample_index=args.sample_index, campaign=r['campaign'], arm=batch,
                prefill_ms=r['validation']['prompt_ms'], decode_ms=r['validation']['decode_ms'],
                input_tokens=n, generated_tokens=r['validation']['generated_tokens'],
                raw_done=[x for x in r['request_protocol'] if x.startswith('DONE ')][0],
                raw_pp=[x for x in r['request_protocol'] if x.startswith('PP ')],
                performance_eligible=False, population_complete=False,
                mapping_observation_scope='READY before GEN and after DONE outside engine phase times; CPU/I/O/cache effects not calibrated')
            fill_matches = re.findall(r'^strata generate: pre-filled (\d+) of (\d+) slots from the profile .*; slot (\d+) verified$', se, re.M)
            resident_matches = re.findall(r'^strata generate: expert residency table: (\d+) resident experts$', se, re.M)
            m.require(len(fill_matches) == 1 and fill_matches[0][:2] == ('144', '144') and resident_matches == ['144'], 'actual full per-layer profile occupancy')
            m.require(r['effective_settings']['expert_slots'] == '144', 'actual cache capacity matches fill and residency')
            r['profile_fill_validation'] = dict(filled=144, capacity=144, residents=144, verified_slot=int(fill_matches[0][2]), per_layer_quota=3, geometry_provenance='Same full48x512 unique ranking/model; per-layer slot ranges and total occupancy imply each of48 layers fills its3 slots. Only one slot readback is the existing byte check, not all144 byte checks.')
            m.require('strata trace:' not in se and 'strata prefill expert' not in se and
                      'strata prefill route census' not in se and '[prefill-native-dequant-group-' not in se,
                      'quiet sample contains no trace/transfer/route/batch census')
            r['quiet_diagnostic_absence'] = True
            r['timer_overhead'] = validate_overhead(se, timing_enabled=False)
            m.require('strata prefill timing:' not in se, 'timer disabled')
            m.require(r['validation']['generated_tokens'] == 16, 'strict16 coverage after normal QUIT; early EOS remains normally closed but fails qualification')
            m.require(output_lines(r) == output_lines(diagnostic_pair[batch]), 'quiet strict16/T/LP5 equals exact Main qualified32K diagnostic for this arm')
            if reference is not None:
                def outputs(v):
                    return [x for x in v['request_protocol'] if x.startswith(('T ','LP '))]
                m.require({k:v for k,v in reference['effective_settings'].items() if k != 'vram_free_mib'} == {k:v for k,v in r['effective_settings'].items() if k != 'vram_free_mib'}, 'OFF/ON exact actual model settings; each observed free-memory reserve checked independently')
                r['reference_comparison'] = dict(output_ids_equal=reference['validation']['output_ids'] == r['validation']['output_ids'],
                    selected_logprob_top5_equal=outputs(reference) == outputs(r), scope='strict16 greedy T and selected LP5 only, no global capability inference')
                m.require(all(r['reference_comparison'][k] for k in ('output_ids_equal','selected_logprob_top5_equal')), 'strict fresh FUSED0 OFF/ON output equivalence')
                m.require(m.sha(Path(r['reference']['path'])) == r['reference']['sha256'], 'immutable OFF receipt remains stable')
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
            m.require(m.sha(publication_path) == r['publication_review']['sha256'] and
                      m.sha(diagnostic_review_path) == r['diagnostic_qualification']['review_sha256'], 'Main publication and32K diagnostic reviews stable')
            m.require(all(m.sha(Path(x['path'])) == x['sha256'] for x in r['observed_libraries'].values()), 'mapped runtime identities stable')
            m.require(all(m.sha(Path(p)) == h for p,h in r['receipt_pins'].items()), 'prerequisite receipts stable')
            m.require(m.sha(review_path) == r['source_applied_main_review_sha256'] and m.sha(controller_review_path) == r['controller_main_review_sha256'], 'reviews stable')
            m.require(r['validation']['mtp_counts'][1] > 0, 'actual speculative draft activity')
            final_head, _ = run('source-head-final', ['/usr/bin/git','rev-parse','HEAD'])
            final_patch, _ = run('source-working-diff-final', ['/usr/bin/git','diff','HEAD'])
            final_status, _ = run('source-tracked-status-final', ['/usr/bin/git','status','--porcelain=v1','--untracked-files=no'])
            m.require(final_head.strip() == r['published_head'] and final_patch == '' and final_status == '', 'published HEAD and clean tracked working tree stable')
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
            r['active'] = owner.active is not None
            r['passed'] = bool(r['complete'] and r.get('kernel_gate_passed') and r['boot_unchanged']
                               and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'))
            r['performance_eligible'] = False
            if 'validation' in r:
                r['validation']['performance_eligible'] = False
            r['finished_utc'] = m.utc()
            save()
        print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'kernel_gate_error', 'model_launch_attempted')}))
        return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
