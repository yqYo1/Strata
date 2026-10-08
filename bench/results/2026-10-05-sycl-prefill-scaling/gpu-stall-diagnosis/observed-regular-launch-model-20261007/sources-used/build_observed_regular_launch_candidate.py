"""Build a private property-only candidate for reviewed observed kernels.

No GPU enumeration/submission or production edit. Other launch properties,
arithmetic, kernel names, geometry, caller waits and all other objects remain.
"""
from pathlib import Path
import datetime
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build = root / 'build-sycl-refresh-20261007'
out = base / 'observed-regular-launch-build'
out.mkdir(mode=0o700)
binary = base / 'strata-observed-regular-launch-candidate'
assert not binary.exists()
flag = 'sycl::ext::oneapi::experimental::use_root_sync'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')


def run(label, argv, timeout=180):
    step = {'label': label, 'argv': argv, 'active': True}
    record['steps'].append(step)
    save()
    before = time.monotonic()
    with (out / (label + '.stdout')).open('wb') as stdout, (out / (label + '.stderr')).open('wb') as stderr:
        result = subprocess.run(argv, cwd=build, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
    step.update(active=False, exit_code=result.returncode, elapsed_seconds=time.monotonic() - before)
    save()
    assert result.returncode == 0, label + ': inspect saved stderr'


env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
started = time.monotonic()
record = dict(active=True, passed=False, gpu_tested=False, adopted=False,
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              controller_sha256=digest(Path(__file__)), sources=[], steps=[], archive_members={},
              scope='Only remove use_root_sync from44 reviewed observed property declarations across8 source files; no arithmetic/range/name/wait changes or blanket persistent-kernel rewrite; no GPU, parity/speed/capacity proof',
              manual_review={
                  'prefill/kernels.dp.cpp': 'Selected element/row ownership, local-memory reductions, subgroup shuffles and work-group barriers; routing uses independent subgroups; KV writes are separated by token/head/64-value group. The recurrence launch properties are retained.',
                  'native_ple_postops.dp.cpp': 'Gate and RMS reduce within one work-group; broadcast/conv/history own separate elements/channels. History kernel follows its producer in the same in-order queue.',
                  'native_qsa_indexer.dp.cpp': 'Each complete block owns its pooled row; only final block writes spare/block_pos; norm/rotation use subgroup and work-group local memory; tail kernel owns channel/slot. Host enabled atomic is not device coordination.',
                  'qsa_decode_attn.dp.cpp': 'Each chunk owns distinct scratch slots; chunk uses work-group local sq/sp/srow. The subsequent merge reads completed scratch through queue ordering and owns distinct query/head output.',
                  'qsa_select.dp.cpp': 'Only block_scores: independent query/block score per subgroup; local XOR reduction, no atomic/global waits. Other selector kernels and cluster paths retained.',
                  'iq_kernels.dp.cpp': 'Only FP16 flat/GU dequant and embedding: distinct256-value destinations, inline dq_dispatch conversion/codebook reads; no inter-group synchronization. Other IQ launches retained.',
                  'dequant_bf16.dp.cpp': 'group32 conversion owns disjoint32-value output, quantization metadata/codebook reads; shared template launcher flag removed with macro continuations retained.',
                  'verify_kernels.dp.cpp': 'Only observed uint32 gather_rows specialization: distinct strided output elements, read-only IDs/source. Other row formats and all persistent/global/doorbell functions retained.'},
              environment={k: env.get(k) for k in ('PATH', 'MKLROOT', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD')})
save()
try:
    receipt = json.loads((base / 'dequant-no-root-refresh-build/record.json').read_text())
    inputs = receipt['link_input_sha256']
    assert all(digest(Path(p)) == v for p, v in inputs.items())
    assert digest(build / 'strata') == 'ceff2d8a6d38fc4779a8b6969d4912efedea5dbd4e2c9229f04af49c680c0bd0'
    record['production_link_inputs_sha256'] = inputs
    record['production_binary_sha256'] = digest(build / 'strata')
    mapping = json.loads((base / 'observed-cooperative-source-targets.json').read_text())
    record['observed_mapping_sha256'] = digest(base / 'observed-cooperative-source-targets.json')
    targets = {}
    hashes = {}
    for row in mapping['rows']:
        assert len(row['source_candidates']) == 1
        candidate = row['source_candidates'][0]
        targets.setdefault(candidate['path'], set()).update(candidate['classes'])
        hashes[candidate['path']] = candidate['sha256']
    assert len(mapping['rows']) == 45 and len(targets) == 8
    commands = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', 'strata'], cwd=build, env=env, text=True, timeout=15).splitlines()
    replacements = {'libstrata_prefill.a': [], 'libstrata_kernels.a': []}
    total = 0
    for index, (relative, names) in enumerate(targets.items()):
        source = root / relative
        assert digest(source) == hashes[relative]
        original = source.read_text()
        declarations = list(re.finditer(r'(?:const\s+)?auto\s+(\w+)\s*=\s*sycl::ext::oneapi::experimental::properties\s*\{([^{}]*)\}', original))
        spans = {}
        for name in sorted(names):
            occurrences = list(re.finditer(r'\bclass\s+' + re.escape(name) + r'\b', original))
            assert occurrences
            for occurrence in occurrences:
                declaration = [p for p in declarations if p.end() < occurrence.start()][-1]
                between = original[declaration.end():occurrence.start()]
                assert len(re.findall(r'parallel_for\s*<', between)) == 1 and len(between) < 2000
                assert declaration.group(2).count(flag) == 1
                at = declaration.start(2) + declaration.group(2).index(flag)
                spans.setdefault(at, []).append(name)
        candidate = original
        for at in sorted(spans, reverse=True):
            assert candidate[at:at+len(flag)] == flag
            candidate = candidate[:at] + candidate[at+len(flag):]
        # Only the named tokens disappear. All other bytes, including macro
        # continuations, subgroup properties, kernel bodies and ranges survive.
        recovered = candidate
        for offset, at in reversed(list(enumerate(sorted(spans)))):
            position = at - offset * len(flag)
            recovered = recovered[:position] + flag + recovered[position:]
        assert recovered == original
        assert original.count(flag) - candidate.count(flag) == len(spans)
        private_source = out / 'sources' / relative
        private_source.parent.mkdir(parents=True, exist_ok=True)
        private_source.write_text(candidate)
        (private_source.with_suffix(private_source.suffix + '.original')).write_text(original)
        (private_source.with_suffix(private_source.suffix + '.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True), candidate.splitlines(keepends=True), fromfile=relative, tofile=relative+'.regular')))
        compiles = [shlex.split(line) for line in commands if line.endswith(' -c ' + str(source))]
        assert len(compiles) == 1
        argv = compiles[0]
        original_object = build / argv[argv.index('-o')+1]
        obj = out / 'objects' / original_object.name
        obj.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(source=relative, original_source_sha256=digest(source), original_object_sha256=digest(original_object),
                     candidate_source=str(private_source), candidate_source_sha256=digest(private_source),
                     original_compile_argv=argv.copy(), original_object=str(original_object), changed_classes=sorted(names),
                     removed_property_tokens=[dict(line=original.count('\n',0,at)+1,classes=spans[at]) for at in sorted(spans)])
        record['sources'].append(entry)
        for option, value in [('-o',obj),('-MT',obj),('-MF',Path(str(obj)+'.d')),('-c',private_source)]:
            assert argv.count(option) == 1
            argv[argv.index(option)+1] = str(value)
        run(f'compile-{index}-{source.name}', argv)
        entry['candidate_object_sha256'] = digest(obj)
        replacements['libstrata_prefill.a' if '/prefill/' in relative else 'libstrata_kernels.a'].append(obj)
        total += len(spans)
    assert total == 44
    record['removed_property_declarations'] = total
    private_archives = {}
    for name, objects in replacements.items():
        archive = out / name
        shutil.copyfile(build/name, archive)
        ar = '/usr/bin/ar'
        before_names = subprocess.check_output([ar,'t',str(build/name)],text=True).splitlines()
        assert len(set(before_names)) == len(before_names)
        replace = {obj.name:obj.read_bytes() for obj in objects}
        assert len(replace) == len(objects) and set(replace) <= set(before_names)
        run('replace-'+name,[ar,'r',str(archive),*[str(obj) for obj in objects]])
        run('index-'+name,['/usr/bin/ranlib',str(archive)])
        assert subprocess.check_output([ar,'t',str(archive)],text=True).splitlines() == before_names
        record['archive_members'][name] = []
        for member in before_names:
            old = subprocess.check_output([ar,'p',str(build/name),member])
            new = subprocess.check_output([ar,'p',str(archive),member])
            assert new == (replace[member] if member in replace else old)
            record['archive_members'][name].append(dict(member=member,replaced=member in replace,original_sha256=hashlib.sha256(old).hexdigest(),candidate_sha256=hashlib.sha256(new).hexdigest()))
        private_archives[name] = str(archive)
    link = [shlex.split(line) for line in commands if ' -o strata ' in line]
    assert len(link)==1 and link[0][:2]==[':','&&'] and link[0][-2:]==['&&',':']
    argv = link[0][2:-2]
    record['original_link_argv'] = argv.copy()
    assert all(argv.count(name)==1 for name in private_archives)
    argv = [private_archives.get(token,token) for token in argv]
    argv[argv.index('-o')+1] = str(binary)
    run('link',argv)
    assert all(digest(Path(p)) == v for p,v in inputs.items())
    assert all(digest(root/x['source'])==x['original_source_sha256'] and digest(Path(x['original_object']))==x['original_object_sha256'] for x in record['sources'])
    assert digest(build/'strata')==record['production_binary_sha256']
    record.update(passed=True,production_inputs_unchanged=True,candidate_binary_sha256=digest(binary),candidate_archive_sha256={name:digest(Path(path)) for name,path in private_archives.items()})
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k:record[k] for k in ['passed','production_inputs_unchanged','removed_property_declarations','candidate_binary_sha256','elapsed_seconds']},indent=2))
