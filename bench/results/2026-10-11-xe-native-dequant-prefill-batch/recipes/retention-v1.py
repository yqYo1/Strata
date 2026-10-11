"""Main-only retirement of the closed, published native-prefill boundary logs."""
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-native-dequant-prefill-batch'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
EXPECTED_OWNER = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
COMMIT = 'bdd23be6b2e4b823c8d5595a9ae90c614625aa67'
assert hashlib.sha256(O.read_bytes()).hexdigest() == EXPECTED_OWNER
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-native-dequant-prefill-retention-root-v1'
    m.require(not out.exists(), 'immutable cleanup output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O),
             committed_evidence=COMMIT, gpu_executed=False, model_executed=False,
             reset_executed=False, service_changed=False, deletion_started=False,
             deleted_prefix_count=0, deleted_logical_bytes=0, deleted_allocated_bytes=0)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, args):
        c, so, se = owner.run(label, args, ENV, W, wall=60, cpu=120,
                             text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c['exit_code'] == 0, 'normal closed ' + label)
        return so.read_text()
    def free_bytes():
        v = os.statvfs(B)
        return v.f_bavail * v.f_frsize
    def file_identity(p):
        v = p.lstat()
        m.require(stat.S_ISREG(v.st_mode) and p.resolve() == p, 'regular non-symlink path')
        return (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_blocks * 512)
    plan = None
    try:
        m.require(run('head', ['/usr/bin/git', 'rev-parse', 'HEAD']).strip() == COMMIT, 'published evidence commit')
        m.require(not run('source-status', ['/usr/bin/git', 'status', '--porcelain']).strip(), 'clean worktree before cleanup')
        m.require(run('object-format', ['/usr/bin/git', 'rev-parse', '--show-object-format']).strip() == 'sha1', 'Git blob hash format')
        tree = run('committed-tree', ['/usr/bin/git', 'ls-tree', '-r', 'HEAD', '--', str(R.relative_to(W))])
        blobs = {}
        for line in tree.splitlines():
            metadata, path = line.split('\t', 1)
            mode, kind, digest = metadata.split()
            m.require(kind == 'blob' and mode in ('100644', '100755'), 'regular committed archive')
            blobs[path] = digest
        def committed(p):
            key = str(p.relative_to(W))
            data = p.read_bytes()
            digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            m.require(blobs.get(key) == digest, 'byte-identical committed object ' + key)

        publication = B / 'xe-native-dequant-prefill-boundary-publish-root-v2/record.json'
        published = json.loads(publication.read_text())
        m.require(published['passed'] and not published['active'] and published['committed_revision'] == COMMIT,
                  'successful closed publication')
        manifest_path = R / 'archive-manifest.json'
        committed(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest['byte_exact_copies']:
            source, dest = Path(entry['source']), W / entry['archive']
            m.require(m.sha(source) == entry['sha256'] == m.sha(dest) and dest.stat().st_size == entry['bytes'], 'exact archived evidence')
            committed(dest)
        for entry in manifest['lossless_encoded_copies']:
            dest = W / entry['archive']
            decoded = base64.b64decode(json.loads(dest.read_text())['data'], validate=True)
            m.require(m.sha(dest) == entry['archive_sha256'] and len(decoded) == entry['decoded_bytes']
                      and hashlib.sha256(decoded).hexdigest() == entry['decoded_sha256']
                      and decoded == Path(entry['source']).read_bytes(), 'exact archived encoded patch')
            committed(dest)
        r['verified_byte_exact_copies'] = len(manifest['byte_exact_copies'])
        r['original_archive_manifest_sha256'] = m.sha(manifest_path)
        plan_path = R / 'retention-v1.json'
        committed(plan_path)
        plan = json.loads(plan_path.read_text())
        r['original_retention_plan_sha256'] = m.sha(plan_path)
        m.require(plan['status'].startswith('planned') and len(plan['files']) == 336, 'unexecuted reviewed deletion plan')
        m.require(plan['logical_bytes'] == 35324933 and plan['allocated_bytes'] == 36700160, 'reviewed byte budgets')
        excerpt = plan['retained_representative']
        representative = W / excerpt['path']
        m.require(m.sha(representative) == excerpt['sha256'] and representative.stat().st_size == excerpt['bytes'], 'retained failure excerpt')
        committed(representative)

        receipt_hashes = {}
        identities_checked = 0
        for receipt in sorted((R / 'receipts').glob('*.json')):
            committed(receipt)
            record = json.loads(receipt.read_text())
            m.require(not record['active'], 'original run inactive')
            receipt_hashes[receipt] = m.sha(receipt)
            for command in record['commands']:
                m.require(command['direct_child_reaped'] and command['session_empty'] and not command['survivors'], 'closed original run')
                for ident in command['owners']:
                    now = m.identity(ident['pid'])
                    m.require(now is None or now['start_ticks'] != ident['start_ticks'], 'original owned process absent')
                    identities_checked += 1
        r['original_owner_identities_checked_absent'] = identities_checked
        copied_sources = {Path(e['source']) for e in manifest['byte_exact_copies']}
        paths, identities = set(), {}
        for entry in plan['files']:
            p = Path(entry['path'])
            m.require(p.is_relative_to(B) and p.parent.name == 'commands' and p.suffix in ('.stdout', '.stderr'), 'only reviewed command logs')
            m.require(p not in paths and p not in copied_sources, 'unique non-evidence path')
            paths.add(p)
            retained = W / entry['retained_receipt']
            m.require(retained in receipt_hashes, 'retained original receipt')
            ident = file_identity(p)
            m.require(ident[2] == entry['bytes'] and ident[4] == entry['allocated_bytes'] and m.sha(p) == entry['sha256'], 'reviewed raw file identity')
            identities[p] = ident
        m.require(sum(e['bytes'] for e in plan['files']) == plan['logical_bytes']
                  and sum(e['allocated_bytes'] for e in plan['files']) == plan['allocated_bytes'], 'exact plan totals')
        r['available_bytes_before_unlink'] = free_bytes()
        r['deletion_started'] = True
        save()
        for entry in plan['files']:
            p = Path(entry['path'])
            m.require(file_identity(p) == identities[p] and m.sha(p) == entry['sha256'], 'unchanged file before unlink')
            p.unlink()
            entry['deletion_outcome'] = 'unlinked'
            r['deleted_prefix_count'] += 1
            r['deleted_logical_bytes'] += entry['bytes']
            r['deleted_allocated_bytes'] += entry['allocated_bytes']
            save()
        r['available_bytes_after_unlink'] = free_bytes()
        r['observed_available_bytes_change'] = r['available_bytes_after_unlink'] - r['available_bytes_before_unlink']
        r['available_space_change_is_not_exclusively_attributed_to_cleanup'] = True
        m.require(all(not p.exists() for p in paths), 'all planned logs absent')
        m.require(all(m.sha(p) == digest for p, digest in receipt_hashes.items()), 'original results unchanged')
        m.require(m.sha(representative) == excerpt['sha256'], 'failure representative unchanged')
        for entry in manifest['byte_exact_copies']:
            m.require(m.sha(Path(entry['source'])) == entry['sha256'] == m.sha(W / entry['archive']), 'retained source and evidence unchanged after cleanup')
        plan['status'] = 'executed; every reviewed file unlinked after committed evidence verification'
        plan['executed_utc'] = m.utc()
        plan['committed_evidence_verified'] = COMMIT
        plan['original_retention_plan_sha256'] = r['original_retention_plan_sha256']
        plan['available_space'] = {k: r[k] for k in ('available_bytes_before_unlink', 'available_bytes_after_unlink', 'observed_available_bytes_change', 'available_space_change_is_not_exclusively_attributed_to_cleanup')}
        plan['retained_representative']['consumer'] = 'Main and the next provider/lifecycle investigation; completed researchers use the committed excerpt and static metadata'
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
        if plan is not None and r['deletion_started']:
            plan['status'] = 'partial cleanup; original measurement statuses unchanged'
            plan['cleanup_error'] = repr(exc)
    finally:
        if plan is not None and r['deletion_started']:
            plan_path.write_text(json.dumps(plan, indent=2) + '\n')
            r['executed_retention_manifest_sha256'] = m.sha(plan_path)
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'deleted_prefix_count', 'deleted_logical_bytes', 'deleted_allocated_bytes', 'observed_available_bytes_change')}))
    raise SystemExit(0 if r['passed'] else 1)
