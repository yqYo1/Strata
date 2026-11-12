#!/usr/bin/env python3
"""Main-owned CPU admission for exact expert copy runs; repository LICENSE.

No GPU work or model qualification. Source,recipe and dependency pins are
committed before compiling; a failed attempt is never refreshed or retried.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as owner

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT, KEY = B/'expert-copy-runs-host-v1', 'expert_copy_runs_host_v1'
RESULT = P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
SOURCES = ('include/strata/prefill/expert_bank_plan.hpp',
    'include/strata/prefill/layer_major_plan.hpp', 'include/strata/core/expert_layer_view.hpp',
    'tests/expert_bank_plan.cpp', 'tools/sycl/check_expert_copy_runs.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py')
require = guards.require


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P)


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row, indent=2)+'\n')
    container = json.loads(RESULT.read_text())
    container[KEY] = {k:v for k,v in row.items() if k != 'commands'}
    RESULT.write_text(json.dumps(container, indent=2)+'\n')


def current(row):
    require(json.loads(git('show','HEAD:'+str(RESULT.relative_to(P))))[KEY] ==
        {k:v for k,v in row.items() if k != 'commands'}, 'commit exact host plan first')
    require(str(Path(sys.executable).resolve()) == row['python']['path'] and
        sys.version == row['python']['version'], 'host controller interpreter changed')
    for name, expected in row['source_sha256'].items():
        require(sha(P/name) == expected and hashlib.sha256(
            git('show',row['source_commit']+':'+name)).hexdigest() == expected,
            'host input changed: '+name)
    require(sha(row['external_inputs']['path']) == row['external_inputs']['sha256'],
        'host dependency inventory changed')
    for path, expected in json.loads(Path(row['external_inputs']['path']).read_text()).items():
        require(sha(path) == expected, 'host compiler/dependency changed: '+path)
    require(sha(row['fallback']['path']) == row['fallback']['sha256'], 'validated engine changed')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh host attempt required')
    revision = git('rev-parse','HEAD').decode().strip()
    pins = {}
    for name in SOURCES:
        raw = git('show',revision+':'+name)
        require(raw == (P/name).read_bytes(), 'commit exact host source: '+name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    compiler = Path('/usr/bin/c++').resolve()
    version = subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0]
    require('13.3.0' in version, 'review changed host compiler before preparing')
    seed = json.loads((B/'coherent-prefill-native-v1/record.json').read_text())
    baseline = json.loads(git('show','HEAD:'+str(RESULT.relative_to(P))))['coherent_prefill_native_v1']
    require(seed['passed'] and seed['complete'] and seed['ownership_closed'] and not seed['active']
        and baseline['binaries'] == seed['binaries'] and baseline['recipe'] == seed['recipe'],
        'preserve committed coherent fallback')
    census = seed['actual_inputs']
    require(sha(census['path']) == census['sha256'], 'original compiler census changed')
    external = {p:h for p,h in json.loads(Path(census['path']).read_text())['compiled'].items()
        if p.startswith(('/usr/', '/opt/'))}
    # GCC uses its own C headers where the original icpx build used Clang's.
    for root in ('/usr/lib/gcc/x86_64-linux-gnu/13/include',
                 '/usr/include/c++/13', '/usr/include/x86_64-linux-gnu/c++/13'):
        for path in Path(root).rglob('*'):
            if path.is_file():
                name = str(path.resolve()); actual = sha(path)
                require(name not in external or external[name] == actual, 'SDK input changed')
                external[name] = actual
    programs = [compiler,Path(sys.executable).resolve()]
    for name in ('cc1plus','as','ld'):
        program = subprocess.check_output([str(compiler),'-print-prog-name='+name],text=True).strip()
        programs.append(Path(program).resolve() if '/' in program else Path('/usr/bin')/program)
    for name in ('libstdc++.so','libgcc_s.so.1','libgcc.a'):
        path = Path(subprocess.check_output([str(compiler),'-print-file-name='+name],text=True).strip()).resolve()
        require(path.is_file(), 'missing host link input: '+name); programs.append(path)
    external.update({str(p):sha(p) for p in programs})
    require(all(sha(p) == h for p,h in external.items()), 'installed compiler inputs differ from seed')
    ROOT.mkdir(mode=0o700)
    deps = ROOT/'compiler-inputs.json'; deps.write_text(json.dumps(external,sort_keys=True)+'\n')
    binary = ROOT/'plan-test'
    row = dict(stage='prepared',source_commit=revision,source_sha256=pins,
        compiler=dict(path=str(compiler),version=version),
        python=dict(path=str(Path(sys.executable).resolve()),version=sys.version),
        external_inputs=dict(path=str(deps),sha256=sha(deps),files=len(external)),
        fallback=seed['binaries']['strata'],
        recipe=dict(environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8'),
            compile=[str(compiler),'-O2','-std=c++20','-Wall','-Wextra','-Werror','-MD',
                '-MF',str(ROOT/'deps.d'),'-I'+str(P/'include'),str(P/'tests/expert_bank_plan.cpp'),
                '-o',str(binary)],test=[str(binary)]),
        plan='Compile and run committed production slot planner plus independent sequential-slot oracle. '
            'Require513capacity/full-fit configurations,eight windows each,hits/gaps/resets,scratch '
            'read equivalence,exact ticket generations and copy-byte/command conservation,22negative '
            'pending/stale/range/overflow cases,and all existing planner/source-view checks. '
            'Stop first source/compiler/dependency/owner/output/check failure;no retry or pin refresh.',
        retention='Temporary CPU fixture for the imminent copy-run device plan;no extra engine generation. '
            'Retire this executable/logs after the same-source CMake fixture supersedes it or candidate closes.',
        limits='CPU metadata/slot/generation logic only;no accepted provider copy,GPU bytes/lifetime,speed, '
            'numerical model,decode,physical262144/RESTORE or adoption qualification.',
        commands=[],passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False)
    save(row);print('Prepared exact copy-run CPU plan;commit before host.',flush=True)


def host():
    row = json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage'] == 'prepared', 'inspect existing host outcome;no rerun')
    frozen = copy.deepcopy(row)
    scope = owner.Owner(ROOT/'owned',lambda: (ROOT/'execution.json').write_text(json.dumps(
        dict(active=scope.active is not None,commands=scope.commands),indent=2)+'\n'))
    try:
        for label,argv in [('compile',row['recipe']['compile']),('test',row['recipe']['test'])]:
            current(frozen)
            event,out,err = scope.run(label,argv,row['recipe']['environment'],P,wall=120,cpu=240,
                rss_cap=2<<30,file_cap=16<<20,text_cap=4<<20,total_cap=32<<20)
            require(owner.closed(event) and event['exit_code'] == 0,
                label+': '+(out.read_text()+err.read_text())[-4000:])
            if label == 'compile':
                paths = shlex.split((ROOT/'deps.d').read_text().partition(':')[2].replace('\\\n',' '))
                dependencies = {str((P/p).resolve()) for p in paths}
                allowed = dict(json.loads(Path(row['external_inputs']['path']).read_text()),
                    **{str(P/n):h for n,h in row['source_sha256'].items()})
                require(all(p in allowed and sha(p) == allowed[p] for p in dependencies),
                    'actual CPU compiler dependency missing or changed: '+str(sorted(dependencies-allowed.keys())))
                row['compiler_census'] = dict(files=len(dependencies),uncovered=0,
                    sha256=hashlib.sha256(json.dumps(sorted(dependencies)).encode()).hexdigest())
            else:
                text = out.read_text(); require(not err.read_text(),'unexpected CPU test stderr')
                match = re.search(r'^expert_bank_upload_runs configurations=513 windows_each=8 '
                    r'logical_uploads=(\d+) commands=(\d+) runs=(\d+) run_experts=(\d+) negative_checks=22 PASS$',text,re.M)
                require(match and 'uploads=9 hits=3 negative_checks=18 PASS' in text and
                    'full_fit512=PASS boundary=PASS negative_checks=19 PASS' in text and
                    'fallback262144=original exact_boundary=PASS' in text and
                    text.endswith('expert_bank_plan PASS GPU=UNRUN model=UNRUN physical262144_RESTORE=UNQUALIFIED\n'),
                    'incomplete copy-run CPU coverage')
                uploads,commands,runs,experts = map(int,match.groups())
                require(commands == uploads-experts+runs and commands < uploads and runs > 0,
                    'copy-run conservation failed')
                row['checks'] = dict(configurations=513,windows_each=8,negative_cases=22,
                    logical_uploads=uploads,copy_commands=commands,runs=runs,run_experts=experts,
                    sequential_ticket_slot_generation_and_bytes_equal=True,existing_planner_checks=True)
        current(frozen)
        binary = ROOT/'plan-test';row.update(stage='closed',passed=True,complete=True,
            binary=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size))
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['commands'] = scope.commands
        row['active'] = scope.active is not None
        row['ownership_closed'] = not row['active'] and all(owner.closed(e) for e in scope.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed']);save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    require(row['passed'],'copy-run CPU gate failed')


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['host']),'prepare|host')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB);guards.no_model_owners([])
        {'prepare':prepare,'host':host}[sys.argv[1]]()
