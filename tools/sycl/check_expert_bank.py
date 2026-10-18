#!/usr/bin/env python3
"""Main-owned CPU admission/ticket checks before bounded expert-bank GPU work."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'moe-expert-bank-component-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/moe-expert-bank-current.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def query_limits():
    """Reuse the maintained query-only probe; submit no work or allocations."""
    root = B / 'moe-expert-bank-limits-v1'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.require(not root.exists(), 'inspect existing device-limit owner before retry')
        raw = subprocess.check_output(['git','show','HEAD:'+str(RESULT.relative_to(P))],cwd=P)
        record = json.loads(raw)
        guards.require(raw == RESULT.read_bytes() and record['complete'] and record['passed'] and
            record['ownership_closed'] and not record['active'], 'commit closed CPU plan first')
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
        pins = {}
        for name in ('tools/sycl/capability_inventory.cpp','tools/sycl/check_expert_bank.py',
                     'tools/sycl/measurement_environment.json','tools/sycl/measurement_guards.py',
                     'tools/sycl/measurement_owner.py'):
            data = subprocess.check_output(['git','show',revision+':'+name],cwd=P)
            guards.require(data == (P/name).read_bytes(), 'commit exact limit-query input: '+name)
            pins[name] = hashlib.sha256(data).hexdigest()
        settings = json.loads((P/'tools/sycl/measurement_environment.json').read_text())
        compiler = Path(settings['environment']['UR_ADAPTERS_FORCE_LOAD']).parents[1]/'bin/clang++'
        guards.require(digest(compiler) == settings['compiler_sha256'], 'compiler changed')
        for name, identity in settings['runtime_elfs'].items():
            guards.require(digest(name) == identity['sha256'], 'runtime changed: '+name)
        before = guards.driver_guard()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
        cursor = next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        env = dict(settings['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
        binary = root/'device-limits'
        query = dict(source_commit=revision,source_sha256=pins,environment_reference=
            'tools/sycl/measurement_environment.json',compiler=dict(path=str(compiler),sha256=digest(compiler)),
            complete=False,passed=False,ownership_closed=False,driver_before=before,
            allocation_and_kernel_submission=False,
            limits='Device properties only. Capacity is not an allocation, owner, performance or physical-context qualification.')
        owner = None

        def persist():
            (root/'record.json').write_text(json.dumps(dict(query,
                active=bool(owner and owner.active is not None),
                owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner = supervision.Owner(root/'owned-limits',persist)
        try:
            for label, command in [('compile-limits',[str(compiler),'-std=c++20','-O2','-fsycl',
                    str(P/'tools/sycl/capability_inventory.cpp'),'-o',str(binary)]),
                    ('query-limits',[str(binary)])]:
                entry,out,err = owner.run(label,command,env,P,wall=180,cpu=600,
                    rss_cap=4<<30,file_cap=16<<20,text_cap=8<<20,total_cap=32<<20)
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                    label+': '+(out.read_text()+err.read_text())[-4000:])
                if label == 'query-limits':
                    properties = {}
                    for line in out.read_text().splitlines():
                        key,sep,value = line.partition('=')
                        if sep and key in ('device','driver','global_mem_size','max_mem_alloc_size'):
                            guards.require(key not in properties, 'duplicate device property')
                            properties[key] = int(value) if key.endswith('_size') else value
                    guards.require(set(properties) == {'device','driver','global_mem_size','max_mem_alloc_size'}
                        and properties['device'] == 'Intel Arc B570 Graphics'
                        and 0 < properties['max_mem_alloc_size'] <= properties['global_mem_size'],
                        'unexpected or incomplete device limits')
                    query['properties'] = properties
            guards.require(all(digest(P/n)==h for n,h in pins.items()), 'limit-query inputs changed')
            query.update(complete=True,passed=True,binary=dict(path=str(binary),
                bytes=binary.stat().st_size,sha256=digest(binary)))
        except BaseException as error:
            query['error'] = str(error)
        finally:
            query['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            query['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
            query['fresh_faults'] = [s for s in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
            try:
                query['driver_after'] = guards.driver_guard()
            except Exception as error:
                query['driver_error'] = str(error)
                query['passed'] = False
            query['passed'] = bool(query['passed'] and query['ownership_closed'] and
                query['boot_unchanged'] and not query['fresh_faults'])
            persist()
            record['device_limits'] = query
            RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:query.get(k) for k in ('complete','passed','ownership_closed','properties','error')}))
        return 0 if query['passed'] else 1


def main():
    if sys.argv[1:] == ['limits']:
        return query_limits()
    guards.require(sys.argv[1:] == ['plan'], 'use check_expert_bank.py plan')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.require(not ROOT.exists(), 'inspect prior expert-bank job; do not restart')
        prior = None
        if RESULT.exists():
            raw = subprocess.check_output(['git','show','HEAD:'+str(RESULT.relative_to(P))],cwd=P)
            old = json.loads(raw)
            guards.require(raw == RESULT.read_bytes() and old['complete'] and old['ownership_closed'] and
                not old['active'], 'commit the closed preceding bank plan before replacement')
            prior = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
                path=str(RESULT.relative_to(P)),source_commit=old['source_commit'],passed=old['passed'])
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
        pins = {}
        for name in ('include/strata/prefill/expert_bank_plan.hpp',
                     'include/strata/prefill/layer_major_plan.hpp', 'tests/expert_bank_plan.cpp',
                     'tools/sycl/check_expert_bank.py', 'tools/sycl/measurement_guards.py',
                     'tools/sycl/measurement_owner.py'):
            raw = subprocess.check_output(['git','show',revision + ':' + name],cwd=P)
            guards.require(raw == (P/name).read_bytes(), 'commit exact bank plan inputs first: ' + name)
            pins[name] = hashlib.sha256(raw).hexdigest()
        compiler = Path('/usr/bin/c++')
        env = dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        version = subprocess.check_output([str(compiler),'--version'],env=env,text=True).splitlines()[0]
        ROOT.mkdir(mode=0o700)
        binary = ROOT/'plan-test'
        steps = [('compile-plan',[str(compiler),'-O2','-std=c++20','-Wall','-Wextra','-Werror',
                   '-I'+str(P/'include'),str(P/'tests/expert_bank_plan.cpp'),'-o',str(binary)]),
                 ('plan',[str(binary)])]
        record = dict(mode='plan',source_commit=revision,source_sha256=pins,
            compiler=dict(path=str(compiler),version=version,sha256=digest(compiler)),
            complete=False,passed=False,active=False,GPU_executed=False,checks=[],
            limits='CPU capacity/state planning only. No owner/queue/DMA completion, GPU staging, model speed, physical262144 or RESTORE qualification.')
        if prior:
            record['previous_result'] = prior
        owner = None

        def persist():
            (ROOT/'record.json').write_text(json.dumps(dict(record,
                active=bool(owner and owner.active is not None),
                owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner = supervision.Owner(ROOT/'owned-plan',persist)
        try:
            for label,command in steps:
                entry,out,err = owner.run(label,command,env,P,wall=120,cpu=240,
                    rss_cap=2<<30,file_cap=16<<20,text_cap=16<<20,total_cap=32<<20)
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                    label+': '+(out.read_text()+err.read_text())[-4000:])
                if label=='plan':
                    text=out.read_text()
                    guards.require('fallback262144=original exact_boundary=PASS' in text and
                        'short_tail_fallback_cases=3' in text and
                        'uploads=9 hits=3 negative_checks=15 PASS' in text and
                        'expert_bank_plan PASS GPU=UNRUN' in text,'incomplete expert-bank coverage')
                    record['checks']=[dict(case='admission_and_tickets',passed=True,
                        result_lines=text.splitlines())]
            guards.require(all(digest(P/n)==h for n,h in pins.items()),'bank plan inputs changed')
            record.update(complete=True,passed=True,binary=dict(path=str(binary),
                bytes=binary.stat().st_size,sha256=digest(binary)))
        except BaseException as error:
            record['error']=str(error)
        finally:
            record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed']=bool(record['passed'] and record['ownership_closed'])
            persist()
            RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
