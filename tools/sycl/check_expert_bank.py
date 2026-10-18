#!/usr/bin/env python3
"""Main-owned CPU admission/ticket checks before bounded expert-bank GPU work."""
import fcntl
import hashlib
import json
from pathlib import Path
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


def main():
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
