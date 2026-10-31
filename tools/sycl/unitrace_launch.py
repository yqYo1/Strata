#!/usr/bin/env python3
"""Build a restricted launcher for the unchanged official non-counter collector."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import unitrace_profile as tool

P, B, RESULT = tool.P, tool.B, tool.RESULT
ROOT = B/'unitrace-session-owner-v1'
PATCHES = (
    ('  utils::SetEnv("ZES_ENABLE_SYSMAN", "1");',
     '  // Local non-counter launcher: preserve an explicitly disabled Sysman.\n'
     '  if (utils::GetEnv("ZES_ENABLE_SYSMAN") != "0")\n'
     '    utils::SetEnv("ZES_ENABLE_SYSMAN", "1");'),
    ('  std::string preload = utils::GetEnv("LD_PRELOAD");',
     '  // Validation belongs to the target, before its collector constructor.\n'
     '  std::string preload = utils::GetEnv("STRATA_UNITRACE_TARGET_PRELOAD");'),
    ('    preload = preload + ":" + lib_path;',
     '    preload = lib_path + ":" + preload;'),
)


def generate():
    source = tool.git(tool.PTI, 'show', tool.REV+':tools/unitrace/src/unitrace.cc').decode()
    for before, after in PATCHES:
        tool.guards.require(source.count(before)==1, 'official launcher patch context changed')
        source = source.replace(before, after)
    return source.encode()


def source_pins(revision):
    paths = ('tools/sycl/unitrace_launch.py', *tool.INPUTS)
    pins = {}
    for name in paths:
        data = tool.git(P,'show',revision+':'+name)
        tool.guards.require(data==(P/name).read_bytes(), 'commit exact owner input: '+name)
        pins[name] = tool.sha(data)
    return pins


def validate(record, collector):
    tool.guards.require(source_pins(record['source_commit'])==record['input_sha256'], 'owner input changed')
    tool.verify(collector)
    for name, item in collector['artifacts'].items():
        tool.guards.require(tool.sha(Path(item['path']).read_bytes())==item['sha256'], 'official artifact changed: '+name)
    tool.guards.require(generate()==(ROOT/'unitrace.cc').read_bytes() and
        tool.sha(generate())==record['generated_source_sha256'], 'owner generated source changed')


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tool.guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        collector = container['unitrace_build']
        tool.guards.require(collector['stage']=='built' and collector['passed'] and collector['ownership_closed']
            and not collector['active'] and not collector['gpu_qualified'], 'unqualified closed official collector build required')
        record = container.get('unitrace_session_owner')
        if action=='prepare':
            tool.guards.require(record is None and not ROOT.exists(), 'inspect existing owner; no blind retry')
            revision = tool.git(P,'rev-parse','HEAD').decode().strip()
            pins = source_pins(revision)
            tool.verify(collector)
            ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
            source = generate();(ROOT/'unitrace.cc').write_bytes(source)
            # Keep the original collector's include/generated closure and flags.
            command = list(collector['commands']['launcher'])
            command[command.index(str(tool.U/'src/unitrace.cc'))] = str(ROOT/'unitrace.cc')
            command[-1] = str(ROOT/'unitrace')
            (ROOT/'libunitrace_tool.so').symlink_to(collector['artifacts']['libunitrace_tool.so']['path'])
            record = dict(schema='strata-unitrace-session-owner-v1',stage='prepared',source_commit=revision,
                input_sha256=pins,generated_source_sha256=tool.sha(source),command=command,
                official_revision=tool.REV,collector=collector['artifacts']['libunitrace_tool.so'],
                environment=dict(collector['environment'],TMPDIR=str(ROOT/'tmp')),
                provenance='Local launcher-only changes to MIT Intel PTI4b6d742, preserving the official header/copyright. Collector unchanged. No XeStrata source.',
                changes=['Respect explicit ZES_ENABLE_SYSMAN=0 in the pre-exec launcher.',
                         'Read target validation preload from STRATA_UNITRACE_TARGET_PRELOAD; launcher itself must have no LD_PRELOAD.',
                         'Prepend collector before target validation library in LD_PRELOAD; actual constructor order/first-init proof still requires a bounded probe.'],
                constraints='Non-counter LevelZero only; XPTI/ITT/OpenCL/OMP/MPI/Perfetto off. No metric/Sysman/KMD request. All target environment fixed before exec; FollowChildProcess=1. Named session uses original launcher/control code, no new in-target mutation or IPC implementation.',
                consumer='Independent source/control-owner audit, then CPU loader/validation probe and known-work GPU/graph qualification before32K+ request-bounded production trace.',
                retention='One current collector plus this imminent launcher candidate; retire superseded CLI and normal output after qualification. Two model generations unchanged.',
                budgets=dict(wall_s=900,cpu_s=900,rss_bytes=8<<30,text_bytes=16<<20,file_bytes=128<<20,total_bytes=256<<20),
                passed=False,active=False,ownership_closed=True,gpu_qualified=False,model_qualified=False)
            container['unitrace_session_owner']=record
            RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
            print('Prepared launcher overlay; commit generated source hash and recipe before build.')
            return 0
        tool.guards.require(action=='build' and record and record['stage']=='prepared', 'committed prepared owner required; no retry')
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))), 'commit owner plan first')
        validate(record,collector)
        owner = None

        def persist():
            record['active']=bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner = tool.supervision.Owner(ROOT/'owned',persist)
        record['stage']='building';persist()
        try:
            entry, _, err = owner.run('launcher',record['command'],record['environment'],tool.PTI,
                wall=900,cpu=900,rss_cap=8<<30,text_cap=16<<20,file_cap=128<<20,total_cap=256<<20)
            tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,
                'launcher compile failed; inspect existing log '+str(err))
            validate(record,collector)
            record.update(stage='built',passed=True,binary=dict(path=str(ROOT/'unitrace'),
                sha256=tool.sha((ROOT/'unitrace').read_bytes()),bytes=(ROOT/'unitrace').stat().st_size))
        except BaseException as error:record.update(stage='failed',passed=False,error=repr(error))
        finally:
            record['ownership_closed']=owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
            record['passed']=bool(record['passed'] and record['ownership_closed'])
            persist();container['unitrace_session_owner']=record
            RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','gpu_qualified','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'use unitrace_launch.py prepare|build')
    raise SystemExit(main(sys.argv[1]))
