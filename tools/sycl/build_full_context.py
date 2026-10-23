#!/usr/bin/env python3
"""Rebuild the unselected full-context diagnostic and streamed-KV fixtures; Main only.

Repository LICENSE. This controller never executes or selects an engine.
Each generated command plan must be committed before its next owned operation.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
import build_engine as build
import check_full_context_source as source
import measurement_guards as guards
import measurement_owner as supervision

P, B = build.P, build.B
ROOT = B / 'full-context-native-v1'
RESULT = source.RESULT
KEY = 'native_budget_build'
PREDECESSOR_KEY = 'native_streamed_build'
MACRO = '-DSTRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC=1'
PREFIXES = ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests', 'third_party',
            'bench/sycl', 'tools/sycl/build_full_context.py', *source.INPUTS]
SDK = B / 'unitrace-warmed-graph-window-v1/compiler-inputs.json'
SDK_SHA = '436cedc9dbd41810cd882e9040a129e6d8b7dbebb98806b00c4f2950b1dc1705'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def summary(record):
    compact = {k: v for k, v in record.items()
               if k not in ('project', 'dependencies', 'environment', 'commands', 'external_inputs')}
    compact['sources'] = [{k: s[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
                          for s in [record['project'], *record['dependencies']]]
    return compact


def sdk():
    guards.require(build.digest(SDK) == SDK_SHA, 'original SDK manifest changed')
    rows = json.loads(SDK.read_text())
    guards.require(len(rows) == 8697, 'SDK input count differs')
    for row in rows:
        guards.require(build.digest(row['path']) == row['sha256'], 'SDK input changed: ' + row['path'])
    return dict(path=str(SDK), sha256=SDK_SHA, files=len(rows))


def check_inputs(record):
    _, reference = source.baseline()
    guards.require(record['selected_original'] == reference, 'selected fallback changed')
    guards.require(build.pin(P, PREFIXES)['source_sha256'] == record['project']['source_sha256'],
                   'native/controller inputs changed')
    for dependency in record['dependencies']:
        actual = build.pin(Path(dependency['root']), ['.'])
        guards.require(actual == dependency, 'native dependency changed')
    guards.require(record['sdk'] == sdk(), 'SDK admission differs')
    for item in record['tools'].values():
        guards.require(build.digest(item['path']) == item['sha256'], 'build tool changed')



def mkl_inputs():
    root = Path('/opt/intel/oneapi/mkl/2026.1')
    paths = {p.resolve() for d in ('include', 'lib/cmake/mkl') for p in (root / d).rglob('*') if p.is_file()}
    ninja = (ROOT / 'build/build.ninja').read_text()
    stanza = ninja[ninja.index('build strata:'):].split('\n\n', 1)[0]
    line = re.search(r'^  LINK_LIBRARIES = (.+)$', stanza, re.M)
    guards.require(line is not None, 'missing actual executable link libraries')
    libraries = [Path(a) for a in shlex.split(line[1]) if a.startswith('/')]
    guards.require(len(libraries) == 4 and all(p.is_relative_to(root / 'lib') and p.is_file()
                                            for p in libraries), 'unexpected imported link assets')
    paths.update(p.resolve() for p in libraries)
    guards.require(len(paths) >= 230, 'MKL header/config/link closure unexpectedly small')
    return {str(p): build.digest(p) for p in sorted(paths)}


def actual_dependency_closure(record, output):
    expected = {}
    for s in [record['project'], *record['dependencies']]:
        expected.update({str((Path(s['root']) / n).resolve()): h for n, h in s['source_sha256'].items()})
    expected.update({str(Path(v['path']).resolve()): v['sha256'] for v in json.loads(SDK.read_text())})
    expected.update(record['external_inputs'])
    paths = {Path(line.strip()) for line in output.read_text().splitlines() if line.startswith('    ') and line.strip()}
    paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
    guards.require((P / 'src/program/generate.cpp').resolve() in paths and len(paths) >= 100,
                   'actual compiler dependency census incomplete')
    uncovered, generated = [], {}
    for p in sorted(paths):
        if str(p) in expected:
            guards.require(build.digest(p) == expected[str(p)], 'actual compiled input changed: ' + str(p))
        elif p.is_relative_to(ROOT.resolve()):
            generated[str(p)] = build.digest(p)
        else:
            uncovered.append(str(p))
    record['actual_dependency_closure'] = dict(files=len(paths), generated_files=len(generated),
        generated_sha256=generated, uncovered_files=len(uncovered), uncovered=uncovered)
    guards.require(not uncovered, 'compiled external inputs absent from prebuild pins: ' + repr(uncovered[:12]))


ENGINE_KEY = KEY
ENGINE_RECORD = ROOT / 'engine-record.json'
TARGETS = ('strata', 'sycl_native16_qsa_streamed_test', 'sycl_native16_qsa_test',
           'sycl_qsa_bf16_attention_test', 'prefill_expert_bank_plan_test')


def prepare(container):
    guards.require(KEY not in container, 'fresh streamed build only; inspect the previous attempt')
    previous = json.loads(ENGINE_RECORD.read_text())
    guards.require(summary(previous)==container[PREDECESSOR_KEY], 'predecessor receipt differs from committed outcome')
    guards.require(previous['stage']=='closed_link' and previous['passed'] and previous['complete'] and
        previous['ownership_closed'] and not previous['active'], 'closed predecessor required')
    guards.require(build.digest(previous['binary']['path'])==previous['binary']['sha256'], 'predecessor bytes differ')
    for fixture in previous['fixtures'].values():
        guards.require(build.digest(fixture['path'])==fixture['sha256'], 'predecessor fixture differs')
    # Recover the actual original inputs before overwriting this unselected
    # generation. Current native edits require a NEW build, never hash refresh.
    for item in [previous['project'], *previous['dependencies']]:
        recovered={n:hashlib.sha256(raw).hexdigest() for n,raw in
            build.blobs(Path(item['root']),item['revision'],item['source_sha256'])}
        guards.require(recovered==item['source_sha256'], 'predecessor source unrecoverable')
    full,reference=source.baseline()
    project=build.pin(P,PREFIXES)
    dependencies=[build.pin(Path(d['root']),['.']) for d in full['dependencies']]
    guards.require(all(a['source_sha256']==b['source_sha256'] for a,b in zip(dependencies,full['dependencies'])),
        'original dependencies changed')
    compiler=full['compiler']; guards.require(build.digest(compiler['path'])==compiler['sha256'],'compiler changed')
    generated=previous['actual_dependency_closure']['generated_sha256']
    for path,h in generated.items(): guards.require(build.digest(path)==h,'generated predecessor changed')
    argv=previous['configure_argv']
    guards.require(argv[argv.index('-S')+1]==str(P) and argv[argv.index('-B')+1]==str(ROOT/'build') and
        '-DCMAKE_CXX_FLAGS:STRING='+MACRO in argv,'unexpected diagnostic configuration')
    record=dict(source_commit=project['revision'],project=project,dependencies=dependencies,
        selected_original=reference,compiler=compiler,sdk=sdk(),tools=previous['tools'],
        configure_argv=argv,environment=previous['environment'],external_inputs=mkl_inputs(),
        generated_expected=generated,targets=list(TARGETS),
        predecessor=dict(source_commit=previous['source_commit'],binary=previous['binary'],
            receipt_sha256=build.digest(ENGINE_RECORD),inputs_recovered=True,
            source_groups=[dict(revision=s['revision'],count=s['count'],aggregate_sha256=s['aggregate_sha256'])
                for s in [previous['project'],*previous['dependencies']]],
            decision='Replace only the unselected diagnostic in place. Its original source/recipe recover;capacity checks failed closed. Original selected6e40 engine+12fixtures remain the validated fallback.'),
        command=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',*TARGETS,'-j','4'],
        dependency_command=['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps'],
        stage='prepared_streamed',passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,native_engine_built=False,production_selected=False,model_adoption_qualified=False,
        physical262144_qualified=False,
        budgets=dict(wall_s=1200,cpu_s=7200,rss_bytes=16<<30,text_bytes=8<<20,file_bytes=256<<20,total_bytes=4<<30),
        criteria='Exact committed native/controller/compiler/dependency/SDK/MKL/generated inputs;owned configure/build/dependency census exit0. Engine and both same-build fixtures hashed;actual dependency closure has zero uncovered files and only predeclared generated headers. No GPU execution or selection.',
        retention='Reuse the existing unselected comparison build directory;no third engine generation. Retain original selected13 assets. New generation is only for imminent traced streamed-Q8,full component/resident-oracle and capacity-model gates;retire normal build output after closure.')
    write(ENGINE_RECORD,record);container[KEY]=summary(record)
    container[PREDECESSOR_KEY]['superseded_by']=KEY
    container[PREDECESSOR_KEY]['original_binary_retired_on_rebuild']=True
    container['native_engine_build']['superseded_by']=KEY
    container['native_engine_build']['original_binary_retired_on_rebuild']=True
    write(RESULT,container)
    print('Prepared in-place streamed diagnostic build;commit before rebuild.',flush=True)


def admission(record,container):
    guards.require(RESULT.read_bytes()==build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        container[KEY]==summary(record),'commit exact streamed build plan first')
    check_inputs(record)
    guards.require(mkl_inputs()==record['external_inputs'],'MKL inputs changed')
    for path,h in record['generated_expected'].items():
        guards.require(build.digest(path)==h,'generated input differs from predeclared bytes')


def rebuild(container):
    record=json.loads(ENGINE_RECORD.read_text())
    guards.require(record['stage']=='prepared_streamed','no duplicate/retry build')
    admission(record,container); original=json.loads(ENGINE_RECORD.read_text());owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ENGINE_RECORD,dict(record,commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned-budget',persist)
    record['stage']='building_streamed';persist()
    try:
        b=record['budgets']
        for label,command in [('configure',record['configure_argv']),('compile-link',record['command'])]:
            entry,out,err=owner.run(label,command,record['environment'],P,wall=b['wall_s'],cpu=b['cpu_s'],
                rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,
                label+' failed: '+(err.read_text(errors='replace')+out.read_text(errors='replace'))[-8192:])
            if label=='configure': admission(original,container)
        for name in TARGETS:
            path=ROOT/'build'/name
            item=dict(path=str(path),bytes=path.stat().st_size,sha256=build.digest(path))
            if name=='strata':record['binary']=item
            else:record.setdefault('fixtures',{})[name]=item
        record['native_engine_built']=True
        entry,out,err=owner.run('actual-dependencies',record['dependency_command'],record['environment'],ROOT,
            wall=30,cpu=30,rss_cap=1<<30,text_cap=64<<20,file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'dependency census failed')
        actual_dependency_closure(record,out)
        guards.require(record['actual_dependency_closure']['generated_sha256']==record['generated_expected'],
            'new compiled generated input lacks a prebuild pin')
        admission(original,container)
        record.update(stage='closed_link',passed=True,complete=True)
    except BaseException as error:
        record.update(stage='build_failed',error=repr(error),passed=False,complete=False)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['passed']=bool(record['passed'] and record['ownership_closed'])
        record['complete']=record['passed'];persist();container[KEY]=summary(record);write(RESULT,container)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','native_engine_built','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(sys.argv[1:] in (['prepare'],['rebuild']),'prepare|rebuild')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(rebuild(container))
