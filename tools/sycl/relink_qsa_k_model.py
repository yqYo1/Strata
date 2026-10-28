#!/usr/bin/env python3
"""Own prospective compile-recipe/link-input qualification; never runs GPU code.

Reconfigure the immutable source in a fresh proof directory, require identical
effective commands, then relink pinned compiled inputs with GNU ld input tracing.
The previous build's incomplete link evidence remains explicitly incomplete.
GNU ld --trace: https://sourceware.org/binutils/docs/ld/Options.html
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_qsa_k_model as m

P, B, ROOT, RESULT = m.P, m.B, m.ROOT, m.RESULT
OUT = ROOT / 'link-contract-v1'
require, digest, write = m.require, m.digest, m.write


def compiled():
    row = json.loads((ROOT / 'record.json').read_text())
    commands = row.pop('commands_executed')
    require(row['passed'] and row['ownership_closed'] and not row['active'] and
            all(m.runtime.supervision.closed(c) for c in commands), 'closed original compile required')
    require(digest(ROOT / 'build/compile_commands.json') == row['compile_database_sha256'],
            'original effective commands changed')
    artifact = row['actual_compiler_inputs']
    require(digest(artifact['path']) == artifact['sha256'], 'actual dependency manifest changed')
    actual = json.loads(Path(artifact['path']).read_text())
    require(len(actual) == artifact['files'] and all(digest(p) == sha for p,sha in actual.items()),
            'compiled source changed')
    return row


def inputs(revision):
    pins = m.sources(revision)
    path = Path(__file__).resolve(); name = str(path.relative_to(P))
    raw = m.component.gen.git('show', revision + ':' + name)
    require(raw == path.read_bytes(), 'commit exact relink controller first')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def link_commands(build_dir, targets):
    raw = subprocess.check_output(['/usr/bin/ninja','-C',str(build_dir),'-t','commands',*targets],
                                  text=True,timeout=20)
    result = {}
    for line in raw.splitlines():
        args = shlex.split(line)
        if '-o' not in args or args[args.index('-o') + 1] not in targets:
            continue
        require(args[:2] == [':','&&'] and args[-2:] == ['&&',':'], 'unexpected CMake shell wrapper')
        args = args[2:-2]
        name = args[args.index('-o') + 1]
        require(name not in result, 'duplicate target link command')
        result[name] = args
    require(set(result) == set(targets), 'missing target link recipe')
    return result


def compile_commands(path, proof=False):
    rows = json.loads(path.read_text()); normalized = []
    original = m.origin()
    compiler = Path(original['compiler']['path']).resolve()
    allowed = [ROOT / 'source', *(Path(g['root']) for g in original['dependencies'])]
    for row in rows:
        args = shlex.split(row['command'])
        require(Path(args[0]).resolve() == compiler, 'unpinned compile driver')
        require('-c' in args and '-o' in args and args[args.index('-c') + 1] == row['file'],
                'unexpected effective compilation form')
        src = Path(row['file']).resolve()
        require(any(src.is_relative_to(d) for d in allowed), 'source escaped immutable snapshot/dependencies')
        if src == ROOT / 'source' / m.model.component.PREFIX:
            require(args.count('-fno-fast-math') == 1 and '-ffast-math' not in args and
                    '-Ofast' not in args and args.count('-ffp-contract=off') == 2 and
                    '-ffp-contract=fast' not in args and '-DSTRATA_ENABLE_SYCL=1' in args,
                    'production prefix math/dispatch compilation differs')
        text = json.dumps(row,sort_keys=True)
        if proof:
            text = text.replace(str(OUT / 'configure-proof'), str(ROOT / 'build'))
        normalized.append(text)
    require(len(rows) == original['configured_compile_entries'] == 232,
            'configured target/source coverage changed')
    return sorted(normalized)


def compiled_compiler():
    return m.origin()['compiler']['path']


def provider_inventory():
    """Freeze candidate providers before linking, including scripts and aliases.

    Actual ld --trace output selects the consumed subset. This deliberately
    includes standard linker scripts/archive stubs absent from the old SDK map.
    """
    paths = set()
    for root in (Path('/usr/lib/x86_64-linux-gnu'), Path('/usr/lib/gcc/x86_64-linux-gnu/13')):
        for pattern in ('*.so','*.a','*.o'):
            paths.update(p for p in root.glob(pattern) if p.is_file())
    paths.add(Path('/lib64/ld-linux-x86-64.so.2'))
    paths.add(Path('/usr/bin/ld'))
    # The SYCL driver's installed executables are separate Apache LLVM assets.
    sdk = Path(compiled_compiler()).parents[1]
    paths.update(p for p in (sdk / 'bin').iterdir() if p.is_file())
    return {str(p):dict(resolved=str(p.resolve()),sha256=digest(p)) for p in sorted(paths)}


def compact(row):
    return {k:v for k,v in row.items() if k not in ('controller_inputs','commands_executed')}


def admit(row):
    old = compiled()
    allowed = m.admit(old)
    require(inputs(row['source_commit']) == row['controller_inputs'], 'relink controller changed')
    providers = provider_inventory()
    require(providers == json.loads(Path(row['providers']['path']).read_text()),
            'prospectively frozen provider/alias inventory changed')
    require(digest(row['providers']['path']) == row['providers']['sha256'], 'provider manifest changed')
    require(digest(row['objects']['path']) == row['objects']['sha256'], 'compiled object inventory changed')
    objects = json.loads(Path(row['objects']['path']).read_text())
    require(len(objects) == row['objects']['files'] and all(digest(p) == sha for p,sha in objects.items()),
            'compiled object/archive changed')
    for path, sha in row['generated_build_inputs'].items():
        require(digest(path) == sha, 'generated compile/link rule changed: ' + path)
    require(link_commands(ROOT / 'build',m.TARGETS) == row['original_link_commands'], 'target recipe changed')
    allowed.update(objects)
    allowed.update({v['resolved']:v['sha256'] for v in providers.values()})
    return old, allowed


def prepare():
    record = m.component.gen.committed(RESULT); scope = record[m.KEY]
    require(not OUT.exists() and 'link_contract' not in scope, 'fresh prospective relink only')
    require(all(c['passed'] and c['ownership_closed'] and not c['fresh_faults'] for c in scope['same_build_checks']),
            'closed native qualification required')
    old = compiled();require(m.compact(old) == scope['build'], 'original build record differs')
    m.admit(old)
    revision = m.component.gen.git('rev-parse','HEAD').decode().strip()
    pins = inputs(revision)
    recipes = link_commands(ROOT / 'build',m.TARGETS)
    objects = {str(p.resolve()):digest(p) for pattern in ('*.o','*.a')
               for p in (ROOT / 'build').rglob(pattern)}
    objects.update({str(p):digest(p) for p in ROOT.glob('input-*.o')})
    require(len(objects) > 130, 'incomplete compiled object inventory')
    providers = provider_inventory()
    graph = {str(p):digest(p) for p in (ROOT / 'build').rglob('*.ninja')}
    graph.update({str(ROOT / 'build' / n):digest(ROOT / 'build' / n)
                  for n in ('CMakeCache.txt','compile_commands.json')})
    OUT.mkdir(mode=0o700)
    for name in ('tmp','cache','output'):
        (OUT / name).mkdir()
    write(OUT / 'providers.json',providers);write(OUT / 'objects.json',objects)
    argv = list(old['commands'][0][1]);argv[argv.index('-B') + 1] = str(OUT / 'configure-proof')
    env = dict(old['environment'],TMPDIR=str(OUT / 'tmp'),XDG_CACHE_HOME=str(OUT / 'cache'))
    links = []
    for name,args in [*recipes.items(),*((n,a) for n,a in old['commands'] if n in ('fixture','numerical'))]:
        args = list(args);args[args.index('-o') + 1] = str(OUT / 'output' / name)
        links.append([name,[*args,'-Wl,-t','-v','-save-temps=obj']])
    row = dict(stage='prepared',source_commit=revision,controller_inputs=pins,
        native_source=old['source_commit'], original_build_source=m.model.BASE,
        original_compile_database_sha256=old['compile_database_sha256'],
        generated_build_inputs=graph,original_link_commands=recipes,configure_command=argv,
        commands=links,environment=env,
        providers=dict(path=str(OUT / 'providers.json'),sha256=digest(OUT / 'providers.json'),files=len(providers)),
        objects=dict(path=str(OUT / 'objects.json'),sha256=digest(OUT / 'objects.json'),files=len(objects)),
        budgets=dict(wall=900,cpu=2400,rss_cap=12 << 30,text_cap=32 << 20,file_cap=512 << 20,total_cap=4 << 30),
        active=False,ownership_closed=True,passed=False,GPU_executed=False,model_executed=False,
        criteria='Prospective CPU-only configure/relink. Original compile232entries and exact production math flags;fresh CMake effective commands and target link recipes must match after build-directory normalization. Freeze compiled objects,SDK,MKL,system providers/scripts/aliases and graph before link. Every traced linker input is frozen or an actual owned compiler-generated output retained by save-temps;unexpected inputs fail. Reuse prior native gates only if all5relinked executable bytes match their original same-source identities. Different bytes require new qualification. Never relabel the earlier partial link evidence.',
        retention='One current model comparison generation;new relink outputs exist only for byte comparison/requalification. Retire duplicates,normal driver/link traces and generated proof/temp files after compact closure evidence;preserve original selected engine/twelve fixtures.')
    admit(row);compile_commands(ROOT / 'build/compile_commands.json')
    write(OUT / 'record.json',row);scope['link_contract'] = compact(row);write(RESULT,record)
    print('Prepared prospective configure/link closure;commit before execution.',flush=True)


def traced_inputs(stdout,stderr,allowed):
    # Save-temps keeps the compiler's internal products available for hashing;
    # bind them to explicit producer -o arguments in verbose owned tool commands.
    produced = set()
    for line in stderr.splitlines():
        if not line.startswith(' "'):
            continue
        args = shlex.split(line)
        if not args or not Path(args[0]).is_absolute():
            continue
        tool = Path(args[0]).resolve()
        require(str(tool) in allowed and digest(tool) == allowed[str(tool)], 'unfrozen executed linker/compiler tool')
        if '-o' in args:
            path = Path(args[args.index('-o') + 1])
            path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
            if path.is_relative_to(OUT) and path.is_file():
                produced.add(path)
    observed, generated = {}, {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        require(path.is_file(), 'unrecognized linker trace line: ' + line)
        sha = digest(path)
        if str(path) in allowed:
            require(sha == allowed[str(path)], 'resolved provider differs from pre-link inventory')
            observed[str(path)] = sha
        else:
            require(path in produced and path.is_relative_to(OUT), 'unfrozen linker input: ' + str(path))
            generated[str(path)] = sha
    require(len(observed) > 8, 'incomplete actual linker input trace')
    return dict(frozen_inputs=observed,compiler_generated_inputs=generated,uncovered_inputs=0)


def run():
    record = m.component.gen.committed(RESULT);scope = record[m.KEY]
    row = json.loads((OUT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and compact(row) == scope['link_contract'] and
            not (OUT / 'owned').exists(), 'fresh committed link contract only')
    old, allowed = admit(row)
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(OUT / 'record.json',dict(row,commands_executed=owner.commands if owner else []))
    try:
        row['stage'] = 'running';owner = m.runtime.supervision.Owner(OUT / 'owned',persist)
        entry,out,err = owner.run('configure',row['configure_command'],row['environment'],ROOT / 'build',**row['budgets'])
        require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0,'fresh proof configure failed')
        require(compile_commands(ROOT / 'build/compile_commands.json') ==
                compile_commands(OUT / 'configure-proof/compile_commands.json',proof=True),
                'fresh effective compile commands differ')
        proof = link_commands(OUT / 'configure-proof',m.TARGETS)
        require({k:[a.replace(str(OUT / 'configure-proof'),str(ROOT / 'build')) for a in v]
                 for k,v in proof.items()} == row['original_link_commands'], 'fresh link recipes differ')
        outputs, links = {}, {}
        previous = dict(old['binaries'],**old['fixtures'])
        for name,args in row['commands']:
            print('CPU prospective relink:',name,flush=True)
            entry,out,err = owner.run(name,args,row['environment'],ROOT / 'build',**row['budgets'])
            require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0,
                    'relink failed: ' + err.read_text(errors='replace')[-2500:])
            links[name] = traced_inputs(out.read_text(),err.read_text(),allowed)
            path = OUT / 'output' / name
            outputs[name] = dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path),
                matches_original=digest(path) == previous[name]['sha256'])
        admit(row)
        write(OUT / 'resolved-inputs.json',links)
        row.update(stage='closed',passed=True,outputs=outputs,
            actual_links=dict(path=str(OUT / 'resolved-inputs.json'),sha256=digest(OUT / 'resolved-inputs.json'),
                frozen_provider_counts={n:len(v['frozen_inputs']) for n,v in links.items()},uncovered_inputs=0),
            effective_commands_verified=True,native_qualification_reusable=all(v['matches_original'] for v in outputs.values()))
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(m.runtime.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist();scope['link_contract'] = compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','native_qualification_reusable','error')}),flush=True)
    require(row['passed'],'prospective build/link contract did not qualify')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare':prepare,'run':run}[sys.argv[1]]()
