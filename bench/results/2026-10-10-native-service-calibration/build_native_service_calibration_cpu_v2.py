"""Root-owned bounded standalone production CPU build; never runs the model."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010')
G = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
I = Path('/opt/intel/oneapi/compiler/2026.1/bin')
PINS = {
    'sycl/tools/native-service-calibration/CMakeLists.txt': '6768f6e7d21c598c32999ef52e16d02b42cd63c545e430745d39ae19e9c131ac',
    'sycl/tools/native-service-calibration/native_service_calibration.cpp': 'a80db6439662c1d825decdd0873bcbb1c3e01ab1be25a7cc07f1a7d56daccdce',
    'sycl/tools/native-service-calibration/README.md': 'c392732567cfdfb45b0800e8bdf7bed386541b912d02b00f7ccb1e942f013e63',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def group_members(pgid):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            words = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            if int(words[2]) != pgid or words[0] == 'Z':
                continue
            rss = int(words[21])*os.sysconf('SC_PAGE_SIZE')
            members.append(dict(pid=int(entry.name), start_ticks=int(words[19]), rss_bytes=rss))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def main():
    assert __debug__
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, pin in PINS.items():
            assert sha(W/name) == pin, name
        out = B/'native-service-calibration-cpu-build-v2'
        out.mkdir(mode=0o700)
        build = W/'build-native-service-calibration-v1'
        assert build.is_dir(), 'existing qualified build for metadata-only rebuild'
        env = dict(PATH=str(I)+':/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        limits = dict(AS_each_bytes=12*1024**3, RSS_group_poll_bytes=8*1024**3,
                      CPU_soft_each_seconds=300, CPU_hard_each_seconds=301,
                      FSIZE_each_bytes=128*1024**2, NOFILE=256, CORE_bytes=0)
        record = dict(active=True, complete=False, passed=False,
                      scope='Standalone full production native CPU build and compile/import audit; no executable service/model/GPU runtime',
                      controller_sha256=sha(__file__), pins={str(W/p):h for p,h in PINS.items()},
                      environment=env, commands=[], cleanup=[], survivors=[], limits=limits,
                      deadline_seconds=600, text_budget_bytes=16*1024**2, peak_group_rss_bytes=0,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      gpu_work_submitted=False, model_opened=False, inference_run=False,
                      performance_eligible=False, adopted=False, full_lifecycle_passed=False)
        rp = out/'record.json'
        started = time.monotonic()
        proc = None
        def save():
            rp.write_text(json.dumps(record, indent=2)+'\n')
        def run(label, argv, cwd=W):
            nonlocal proc
            so, se = out/(label+'.stdout'), out/(label+'.stderr')
            command = dict(label=label, argv=argv, cwd=str(cwd), active=True)
            record['commands'].append(command)
            save()
            def child_limits():
                for kind, pair in ((resource.RLIMIT_AS,(12*1024**3,12*1024**3)),
                                   (resource.RLIMIT_CPU,(300,301)),
                                   (resource.RLIMIT_FSIZE,(128*1024**2,128*1024**2)),
                                   (resource.RLIMIT_NOFILE,(256,256)),(resource.RLIMIT_CORE,(0,0))):
                    resource.setrlimit(kind, pair)
            begin = time.monotonic()
            with so.open('wb') as stdout, se.open('wb') as stderr:
                proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                        preexec_fn=child_limits, start_new_session=True)
                command['owner'] = dict(pid=proc.pid, pgid=proc.pid,
                    start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
                save()
                while proc.poll() is None:
                    assert time.monotonic()-started < 600, 'build wall deadline'
                    assert sum(p.stat().st_size for p in out.glob('*.std*')) < 16*1024**2, 'build log budget'
                    members = group_members(proc.pid)
                    rss = sum(m['rss_bytes'] for m in members)
                    record['peak_group_rss_bytes'] = max(record['peak_group_rss_bytes'],rss)
                    assert rss <= 8*1024**3, 'build process-group RSS budget'
                    time.sleep(.10)
            command.update(active=False,exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin)
            assert proc.returncode == 0, label+' failed'
            assert not group_members(proc.pid), label+' process-group survivors'
            save()
            return so.read_text()
        save()
        try:
            record['compiler_pins'] = {str(I/n):sha(I/n) for n in ('icx','icpx')}
            run('compiler-c',[str(I/'icx'),'--version'])
            run('compiler-cxx',[str(I/'icpx'),'--version'])
            assert run('ggml-head',['/usr/bin/git','rev-parse','HEAD'],G).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
            assert run('ggml-status-before',['/usr/bin/git','status','--porcelain'],G)==''
            paths=run('ggml-tracked',['/usr/bin/git','ls-files'],G).splitlines()
            code_paths=[p for p in paths if p.endswith(('.c','.cc','.cpp','.cxx','.h','.hpp','.inl','.ipp','.tpp','.inc','.in','.cmake')) or Path(p).name=='CMakeLists.txt']
            ggml_pins={p:sha(G/p) for p in code_paths}
            source_paths=list(W.joinpath('include').rglob('*.hpp'))
            source_paths += [W/'src/artifact/native_role_plan.cpp']
            source_paths += list(W.joinpath('src/kernels/cpu').glob('*.cpp'))+list(W.joinpath('src/kernels/cpu').glob('*.inl'))
            source_paths += [W/'sycl/src/kernels/cpu/iq_avx2.cpp']
            source_pins={str(p.relative_to(W)):sha(p) for p in source_paths}
            record.update(ggml_source_pins=ggml_pins,production_source_pins=source_pins)
            save()
            run('configure',['/usr/bin/cmake','-S',str(W/'sycl/tools/native-service-calibration'),'-B',str(build),'-G','Ninja',
                '-DCMAKE_C_COMPILER='+str(I/'icx'),'-DCMAKE_CXX_COMPILER='+str(I/'icpx'),
                '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_ROOT='+str(W),'-DGGML_SOURCE_DIR='+str(G/'ggml')])
            run('build',['/usr/bin/cmake','--build',str(build),'--target','native_service_calibration','--parallel','6'])
            binary=build/'native_service_calibration'
            imported=run('direct-imports',['/usr/bin/readelf','-d',str(binary)])
            assert not any(s in imported.lower() for s in ('libsycl','libur_','libze_loader','libmkl','libiomp','libgomp')), 'unexpected runtime imports'
            compile_path=build/'compile_commands.json'
            commands=json.loads(compile_path.read_text())
            assert commands and all('-fsycl' not in c['command'] for c in commands), 'CPU compile closure'
            project=[c for c in commands if c['file'].startswith(str(W)+'/')]
            assert len(project)==11, 'eleven harness/production translation units'
            assert all('-march' not in c['command'] and '-fp-model=precise' in c['command'] for c in project), 'Strata flags'
            for command in project:
                if command['file'].endswith('/q2_avx2.cpp'):
                    assert '-DSTRATA_AVXVNNI=1' in command['command'], 'Q2 VNNI compile pin'
                if command['file'].endswith('/sycl/src/kernels/cpu/iq_avx2.cpp'):
                    assert '-DSTRATA_AVXVNNI=0' in command['command'], 'IQ wrapper VNNI pin'
            run('target-commands',['/usr/bin/ninja','-C',str(build),'-t','commands','native_service_calibration'])
            assert run('ggml-status-after',['/usr/bin/git','status','--porcelain'],G)==''
            assert all(sha(G/p)==h for p,h in ggml_pins.items()), 'ggml source mutation'
            assert all(sha(W/p)==h for p,h in source_pins.items()), 'production source mutation'
            assert all(sha(W/p)==h for p,h in PINS.items()), 'harness source mutation'
            record.update(complete=True,passed=True,binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary)),
                          compile_commands=dict(path=str(compile_path),bytes=compile_path.stat().st_size,sha256=sha(compile_path)),
                          compile_rule_count=len(commands),project_compile_rule_count=len(project),
                          import_audit_scope='direct NEEDED only; no device API runtime tripwire or payload smoke yet')
        except BaseException as error:
            record['error']=type(error).__name__+': '+str(error)
        finally:
            if proc is not None:
                members=group_members(proc.pid)
                if proc.poll() is None or members:
                    record['cleanup'].append(dict(signal='TERM',owned_process_group=proc.pid,members=members))
                    try: os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
                    members=group_members(proc.pid)
                    if proc.poll() is None or members:
                        record['cleanup'].append(dict(signal='KILL',owned_process_group=proc.pid,members=members))
                        try: os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                        proc.wait(timeout=5)
                record['survivors']=group_members(proc.pid)
            record.update(active=False,elapsed_seconds=time.monotonic()-started,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          logs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.glob('*.std*')})
            save()
        print(json.dumps(dict(record=str(rp),sha256=sha(rp),passed=record['passed'],error=record.get('error'),elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
