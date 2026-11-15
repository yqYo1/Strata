#!/usr/bin/env python3
"""Main-authored bounded graph-envelope support gate; no model or speed claim."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as build_tools
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from check_verifier_retirement import DELTA

P, B = build_tools.P, build_tools.B
ROOT = B/'mtp-graph-support-v1'
RESULT = build_tools.RESULT.with_name('mtp-graph-observer-current.json')


def evidence(text, profile, exit_code):
    """Successful execution and supported profiling are separate requirements."""
    final = re.findall(r'^graph envelope: profile=(\d) numerical=PASS guards=PASS aliases=PASS '
        r'checked_teardown=PASS support=(\w+) arithmetic_service=UNMEASURED hardware_saturation=UNMEASURED$', text, re.M)
    samples = []
    for line in text.splitlines():
        if line.startswith('graph sample: '):
            pairs = re.findall(r'([a-z_]+)=([0-9]+)', line)
            row = {k:int(v) for k,v in pairs}
            guards.require(len(pairs) == len(row) == 10, 'ambiguous graph sample')
            samples.append(row)
    if exit_code == 77 and profile:
        guards.require('support=UNKNOWN' in text, 'unsupported exit lacks explicit evidence')
        return dict(passed=False, support='UNKNOWN', samples=samples)
    guards.require(exit_code == 0 and final == [(str(profile), 'SUPPORTED' if profile else 'NOT_ENABLED')],
        'graph fixture did not close its numerical/alias/teardown contract')
    if profile:
        guards.require(len(samples) == 2 and [r['replay'] for r in samples] == [0,1], 'both replays required')
        for row in samples:
            guards.require(row['status'] == 0 and row['fields'] == 7 and
                row['timer_resolution'] > 0 and row['end'] > row['start'] and
                row['end'] - row['start'] >= row['timer_resolution'] and row['query_ns'] < 1000000000 and
                row['envelope_qualified'] == 1 and
                row['queue_delay_qualified'] == int(row['start'] >= row['submit']), 'invalid graph clock')
        guards.require(samples[1]['start'] >= samples[0]['end'], 'replay clocks overlap or regress')
    else:
        guards.require(not samples, 'ordinary graph exposed profiled samples')
    return dict(passed=True, support=final[0][1], samples=samples)


def main():
    guards.require(not sys.argv[1:], 'no options; fresh bounded support check only')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]); guards.driver_guard()
        compact = validation.committed(build_tools.RESULT)
        build_path = build_tools.ROOT/'record.json'
        full = json.loads(build_path.read_bytes())
        guards.require(full['passed'] and full['complete'] and full['ownership_closed'] and not full['active'] and
            all(full[k] == compact[k] for k in ('source_commit','binary','fixtures','compiler','configure_argv','targets')),
            'matching closed committed build required')
        guards.require('tools/sycl/check_mtp_observer.py' in full['project']['source_sha256'], 'build must pin this controller')
        settings = validation.committed(build_tools.SETTINGS)
        probe = validation.preload_qualification()
        runtime_pins = {}
        for name,item in settings['runtime_elfs'].items():
            old, expected = DELTA.get(name, (item['sha256'],item['sha256']))
            guards.require(old == item['sha256'], 'original runtime baseline differs')
            runtime_pins[name] = expected

        def unchanged():
            for source in [full['project'],*full['dependencies']]:
                for name,data in build_tools.blobs(Path(source['root']),source['revision'],source['source_sha256']):
                    expected = source['source_sha256'][name]
                    guards.require(hashlib.sha256(data).hexdigest() == expected and
                        build_tools.digest(Path(source['root'])/name) == expected, 'native source changed: '+name)
            for item in [full['binary'],*full['fixtures'].values()]:
                guards.require(build_tools.digest(item['path']) == item['sha256'], 'compiled artifact changed')
            for name,expected in {**runtime_pins,**full['fixture_external_headers']}.items():
                guards.require(build_tools.digest(name) == expected, 'runtime/header changed: '+name)
            guards.require(build_tools.digest(full['compiler']['path']) == full['compiler']['sha256'], 'compiler changed')

        unchanged()
        guards.require(not ROOT.exists(), 'inspect existing receipt; never automatically retry this check')
        ROOT.mkdir(mode=0o700)
        for name in ('tmp','cache'): (ROOT/name).mkdir()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
        cursor = next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        fixture = full['fixtures']['sycl_graph_envelope_test']
        env = dict(settings['environment'], TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
            LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
        record = dict(source_commit=full['source_commit'],build_record_sha256=build_tools.digest(build_path),
            fixture=fixture, environment=env, runtime_asset_delta={k:dict(qualified_sha256=v[0],candidate_sha256=v[1]) for k,v in DELTA.items()},
            checks=[],complete=False,passed=False,active=False,
            consumer='Immediate graph-support decision before MTP model attribution; retire successful full traces after recording the decision.',
            prospective_gate='Separate profile-off/on processes pass numerical guards, event provenance, two replay clocks and checked teardown with no fresh faults. Exit77 is UNKNOWN, not a pass; no retry.',
            limits='Tiny fixture only; no engine/model qualification, observer overhead, arithmetic service, hardware saturation or speed claim.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record,commands=owner.commands if owner else []),indent=2)+'\n')

        owner = supervision.Owner(ROOT/'owned',persist); persist()
        try:
            for profile in (0,1):
                command = [fixture['path'],'--profile='+str(profile)]
                entry,out,err = owner.run('profile-'+str(profile),command,env,P,wall=90,cpu=120,
                    rss_cap=2<<30,file_cap=64<<20,text_cap=64<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry), 'graph fixture ownership not closed')
                stdout,stderr = out.read_text(errors='replace'),err.read_text(errors='replace')
                row = dict(profile=profile,command=command,exit_code=entry.get('exit_code'),ownership_closed=True,
                    stdout_sha256=build_tools.digest(out),stderr_sha256=build_tools.digest(err))
                record['checks'].append(row)
                row['validation'] = validation.preload_evidence(stderr,entry['direct_identity']['pid'])
                row.update(evidence(stdout,profile,entry.get('exit_code')))
                if row['passed']:
                    # The qualified constructor deliberately probes invalid-null
                    # dispatch before this boundary; reject errors in the fixture.
                    boundary = re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',stderr,re.M)
                    guards.require(boundary is not None and not re.search(
                        r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain',stderr[boundary.end():]),
                        'unexpected runtime error in successful fixture')
                guards.require(row['passed'], 'graph support unknown or fixture failed; inspect owned output')
            unchanged()
            record['passed'] = True
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['complete'] = record['ownership_closed']
            record['fresh_faults'] = None
            try:
                after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                record['fresh_faults'] = [x for x in after.splitlines() if re.search(
                    r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
                record['driver_after'] = guards.driver_guard()
                record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            except Exception as error: record['final_probe_error'] = str(error)
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                record.get('boot_unchanged') and record['fresh_faults'] == [] and 'final_probe_error' not in record)
            persist(); RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__': sys.exit(main())
