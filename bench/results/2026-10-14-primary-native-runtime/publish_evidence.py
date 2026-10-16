#!/usr/bin/env python3
"""Preserve exact closed receipts and compact diagnostics without altering status."""
import fcntl
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-native-runtime'
ROOTS = ['main-primary-native-build-root-v1', 'main-primary-native-runtime-root-v1',
         *['main-primary-native-engine-build-root-v' + str(i) for i in range(1, 14)],
         'main-primary-weight-index-host-root-v1', 'main-primary-native-kernels-root-v1',
         'main-primary-native-kernels-root-v2', 'main-primary-native-kernels-root-v3']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def store(relative, data):
    target = D / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = gzip.compress(data, mtime=0)
    target.write_bytes(encoded)
    assert gzip.decompress(encoded) == data
    return dict(path=relative, bytes=len(data), sha256=sha(data),
                gzip_bytes=len(encoded), gzip_sha256=sha(encoded))


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = {}
        receipts = []
        requested = {}
        diagnostics = []
        for root in ROOTS:
            path = B / root / 'record.json'
            data = path.read_bytes()
            record = json.loads(data)
            assert not record['active'], root
            assert not any(command.get('survivors') for command in record['commands']), root
            assert all(command.get('direct_child_reaped') and command.get('session_empty')
                       for command in record['commands']), root
            assert not any(re.search(r'(TOKEN|PASSWORD|SECRET)', key, re.I)
                           for key in record.get('environment', {})), 'private env cannot be published'
            records[root] = record
            receipt = store('receipts/' + root + '/record.json.gz', data)
            receipt.update(original_path=str(path), passed=record['passed'], complete=record['complete'])
            receipts.append(receipt)
            for file, digest in record.get('source_pins', {}).items():
                requested.setdefault((file, digest), []).append(root)
            for command in record['commands']:
                for name, identity in command.get('logs', {}).items():
                    # Build owner logs live alongside the reused build; other
                    # logs are in each private job's owned directory.
                    candidates = [B / root / 'owned' / name]
                    if root.startswith('main-primary-native-engine-build-root-v'):
                        attempt = int(root.rsplit('v', 1)[1])
                        suffix = '' if attempt == 1 else '-v' + str(attempt)
                        candidates.append(B / 'main-primary-native-build-root-v1' /
                                          ('engine-build-owned' + suffix) / name)
                    found = [path for path in candidates if path.is_file()]
                    assert len(found) == 1, (root, name, candidates)
                    log = found[0].read_bytes()
                    assert len(log) == identity['bytes'] and sha(log) == identity['sha256'], found[0]
                    text = log.decode(errors='replace')
                    if name.endswith('.stdout'):
                        # Per-case results and fault observations are compact;
                        # successful compiler listings have identities/recipes.
                        if 'build' not in command['label'] and not command['label'].startswith('configure'):
                            receipts.append(store('receipts/' + root + '/' + name + '.gz', log))
                    if command['exit_code'] != 0 and 'error:' in text:
                        errors = [line for line in text.splitlines() if 'error:' in line]
                        first = text.find('error:')
                        diagnostics.append(dict(root=root, command=command['label'], original_log=str(found[0]),
                            original_sha256=identity['sha256'], error_lines=errors,
                            window=text[max(0, first - 1500):first + 2000], exit_code=command['exit_code']))
                    elif command['exit_code'] != 0 and name.endswith('.stderr'):
                        app = [line for line in text.splitlines() if line and
                               not line.startswith((' ', '[', '<VALIDATION>'))]
                        if app:
                            first = text.index(app[0])
                            diagnostics.append(dict(root=root, command=command['label'],
                                original_log=str(found[0]), original_sha256=identity['sha256'],
                                application_errors=app, window=text[max(0, first - 1500):first + 2000],
                                exit_code=command['exit_code']))
        # Source fingerprints that were not committed at execution time must
        # resolve to Git or an exact source capture, including failed attempts.
        git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        variants = {}
        for file, _digest in requested:
            path = P / file
            variants.setdefault(file, {})[sha(path.read_bytes())] = ('working-source', path.read_bytes())
            for revision in [git_revision, '4fa478f50f8da396e651e3ae862f8b1849d49b54']:
                proc = subprocess.run(['git', 'show', revision + ':' + file], cwd=P, capture_output=True)
                if proc.returncode == 0:
                    variants[file][sha(proc.stdout)] = ('git:' + revision + ':' + file, proc.stdout)
        # One intermediate unused-parameter fix preceded the function annotation.
        file = 'src/core/pinned.cu'
        intermediate = (P / file).read_bytes().replace(b'[[maybe_unused]] bool clear_error()', b'bool clear_error()')
        variants[file][sha(intermediate)] = ('intermediate-pinned-parameter-annotation', intermediate)
        file = 'src/kernels/sycl/native_q2.cpp'
        before_guard = (P / file).read_bytes().replace(
            b"    bool finite = true;\n    for (int i = 0; i < 32; ++i) {\n"
            b"      finite &= sycl::isfinite(row[i]);\n"
            b"      maximum = sycl::fmax(maximum, sycl::fabs(row[i]));\n    }",
            b"    for (int i = 0; i < 32; ++i)\n"
            b"      maximum = sycl::fmax(maximum, sycl::fabs(row[i]));").replace(
            b"    // Even finite tiny inputs can overflow the reciprocal. Mark the block\n"
            b"    // invalid before any non-finite float-to-int conversion; never invent\n"
            b"    // apparently valid activation data for an unsupported input range.\n"
            b"    if (!finite || !sycl::isfinite(scale) || !sycl::isfinite(inverse)) {\n"
            b"      scales[b] = sycl::nan(0u);\n"
            b"      for (int i = 0; i < 34; ++i)\n"
            b"        block[i] = 0;\n      return;\n    }\n", b'')
        variants[file][sha(before_guard)] = ('intermediate-RN-scale-before-invalid-range-guard', before_guard)
        before_quantizer_fix = before_guard.replace(
            b"    // Match the CPU quantizer's rounded division. Multiplication by a rounded\n"
            b"    // reciprocal can change the scale by one bit even without fast math.\n"
            b"    const float scale = sycl::ext::intel::math::fdiv_rn(maximum, 127.f);",
            b"    const float scale = maximum * (1.f / 127.f);")
        variants[file][sha(before_quantizer_fix)] = ('intermediate-standard-exp-before-scale-correction', before_quantizer_fix)
        file = 'include/strata/sycl/native_q2.hpp'
        before_guard_contract = (P / file).read_bytes().replace(
            b'// Finite inputs with a representable reciprocal follow that contract. A block\n'
            b'// with non-finite input or scale/inverse is marked by a NaN scale and zero codes\n'
            b'// and sum, avoiding invalid float-to-int conversion. Callers must reject that\n'
            b'// marker; it is not a valid quantized activation or CPU parity promise.\n', b'')
        variants[file][sha(before_guard_contract)] = ('intermediate-RN-scale-contract-before-invalid-range-guard', before_guard_contract)
        source_index = []
        for (file, digest), consumers in sorted(requested.items()):
            assert digest in variants[file], (file, digest)
            origin, data = variants[file][digest]
            row = dict(path=file, sha256=digest, consumers=consumers, origin=origin)
            if not origin.startswith('git:'):
                row['capture'] = store('historical-source/' + digest + '.gz', data)
            source_index.append(row)
        # Keep one allocation warning and one unresolved loader lookup window.
        runtime = B / 'main-primary-native-runtime-root-v1/owned/runtime-debug.stderr'
        raw = runtime.read_bytes()
        text = raw.decode()
        windows = {}
        for needle in ['There are no valid references to handle 0', 'zelLoaderTranslateHandle',
                       'zelSetDriverTeardown', 'zeCommandListIsGraphCaptureEnabledExt']:
            at = text.index(needle)
            windows[needle] = text[max(0, at - 500):at + 1200]
        diagnostics.append(dict(root='main-primary-native-runtime-root-v1', original_log=str(runtime),
            original_sha256=sha(raw), windows=windows,
            interpretation='Known null-pool validator false positive; QUERY_FALSE; optional teardown hook; translate-handle lookup origin unresolved. API/numerical success is separate.'))
        (D / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        (D / 'receipts-index.json').write_text(json.dumps(receipts, indent=2) + '\n')
        (D / 'historical-source-index.json').write_text(json.dumps(source_index, indent=2) + '\n')
        summary = dict(roots={root: {key: records[root].get(key) for key in
                      ['passed', 'complete', 'active', 'GPU_executed', 'model_executed', 'performance_qualified',
                       'physical_256K_qualified', 'HA_exp_parity_qualified', 'error']}
                      for root in ROOTS},
                      owner_sessions_all_closed=True, source_fingerprints_all_resolved=True,
                      original_failed_status_preserved=True,
                      current_primary_model_performance='unmeasured', default_promotion_complete=False,
                      new_Niko_revision_checked=False, no_GPU_reset_unbind_or_service_change=True)
        (D / 'outcomes.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(closed_roots=len(records), exact_receipts=len(receipts),
                          resolved_source_fingerprints=len(source_index))), flush=True)


if __name__ == '__main__':
    main()
