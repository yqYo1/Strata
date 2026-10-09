"""Root read-only proof of actually compiled CPU target flags; no compile/run."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess

B = Path(__file__).parent
BUILD_RECEIPT = B / 'native-wrapper-matched-private-build-v1/record.json'
OLD_RECEIPT = B / 'decode-pool-phase-timing-v0141-private-build-v2/record.json'
OUT = B / 'native-wrapper-matched-build-flags-v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(root, build, target, fresh):
    def replace(s):
        return s.replace(str(build), '<build>').replace(str(root), '<source>')

    def normalize(args):
        kept = []; i = 0
        while i < len(args):
            if args[i] in ['-MT', '-MF']:
                assert i + 1 < len(args)
                i += 2; continue
            if args[i] in ['-MD', '-MMD']:
                i += 1; continue
            kept.append(replace(args[i])); i += 1
        return kept

    declared = {}
    for item in json.loads((build / 'compile_commands.json').read_text()):
        args = item.get('arguments') or shlex.split(item['command'])
        assert args.count('-o') == 1
        output = args[args.index('-o') + 1]
        key = (replace(item['file']), replace(output))
        assert key not in declared
        declared[key] = normalize(args)
    text = subprocess.check_output(['/usr/bin/ninja', '-C', str(build), '-t', 'commands', target],
                                   text=True, timeout=15)
    assert len(text) < 3 * 1024**2
    reachable = {}; objects = []; links = []
    for line in text.splitlines():
        args = shlex.split(line)
        if '-c' not in args:
            links.append(line); continue
        assert args.count('-o') == args.count('-c') == 1
        source, output = args[args.index('-c') + 1], args[args.index('-o') + 1]
        key = (replace(source), replace(output))
        assert key not in reachable
        reachable[key] = normalize(args)
        assert reachable[key] == declared[key]
        objects.append(output)
    entries = [line.split('\t')[3] for line in (build / '.ninja_log').read_text().splitlines()[1:] if line]
    compiled = [s for s in entries if s.endswith('.o')]
    if fresh:
        assert len(compiled) == len(set(compiled)) == len(objects)
        assert set(compiled) == set(objects), 'fresh target must actually compile each reachable object'
        assert all('/src/kernels/cpu/' in key[0] or '/ggml/' in key[0] or
                   key[0] == '<source>/sycl/tools/native_wrapper_parity.cpp' for key in reachable)
        assert any(key[0] == '<source>/sycl/src/kernels/cpu/iq_avx2.cpp' for key in reachable)
        assert any(key[0] == '<source>/src/kernels/cpu/native_expert.cpp' for key in reachable)
        assert not any('strata_kernels.dir/' in line or 'strata_core.dir/' in line for line in links)
    proof = {'reachable_compile_objects': len(reachable), 'fresh_compiled_objects': len(compiled),
             'compile_commands_sha256': sha(build / 'compile_commands.json'),
             'build_ninja_sha256': sha(build / 'build.ninja'), 'ninja_log_sha256': sha(build / '.ninja_log'),
             'reachable_commands_sha256': hashlib.sha256(text.encode()).hexdigest()}
    return reachable, proof


assert not OUT.exists()
new = json.loads(BUILD_RECEIPT.read_text())
old = json.loads(OLD_RECEIPT.read_text())
assert new['passed'] and new['complete'] and not new['active'] and not new['cleanup'] and not new['survivors']
assert old['passed'] and not old['active'] and old['compiled_engine']
assert sha(OLD_RECEIPT) == '3632b1514544ef2dd89046c112b823c75a27d0215aca2fce58c4b8df9e43d7f8'
arms = {}; proofs = {}
for arm in new['arms']:
    root, build = Path(arm['root']), Path(arm['build'])
    assert sha(build / 'compile_commands.json') == arm['files']['compile_commands.json']
    assert sha(build / 'build.ninja') == arm['files']['build.ninja']
    assert sha(build / '.ninja_log') == arm['files']['.ninja_log']
    assert sha(arm['binary']) == arm['binary_sha256']
    arms[arm['label']], proofs[arm['label']] = inspect(root, build, 'native_wrapper_parity', True)
assert set(arms) == {'T', 'H'}
missing = sorted(set(arms['T']) ^ set(arms['H']))
different = [key for key in arms['T'].keys() & arms['H'].keys() if arms['T'][key] != arms['H'][key]]
assert not missing and not different
old_build = Path(old['binary']).parent
assert sha(old_build / 'compile_commands.json') == old['compile_commands_sha256']
old_commands, old_proof = inspect(Path(old['root']), old_build, 'strata', False)
library_commands = {key: args for key, args in arms['T'].items()
                    if key[0] != '<source>/sycl/tools/native_wrapper_parity.cpp'}
assert all(key in old_commands and old_commands[key] == args for key, args in library_commands.items()), \
       'every CPU/ggml object must retain actual production baseline flags'
record = {'passed': True, 'gpu_executed': False, 'runtime_tested': False, 'adopted': False,
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checker_sha256': sha(__file__), 'build_receipt_sha256': sha(BUILD_RECEIPT),
          'baseline_build_receipt_sha256': sha(OLD_RECEIPT),
          'target_proofs': proofs, 'production_baseline_proof': old_proof,
          'matched_target_object_count': len(arms['T']),
          'baseline_matched_library_object_count': len(library_commands),
          'missing_objects': missing, 'different_flags': different,
          'normalization': 'Only source/build prefixes and dependency-output switches; compiler, ISA, precise math, macro, include, source and object flags retained.',
          'commands': {label: [{'source': key[0], 'object': key[1], 'argv': args}
                                for key, args in sorted(commands.items())] for label, commands in arms.items()}}
OUT.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: record[k] for k in ['passed', 'matched_target_object_count',
                 'baseline_matched_library_object_count', 'missing_objects', 'different_flags']}))
