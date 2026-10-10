"""Root-only serialized precise IntelLLVM build/exactness/sanitizer gate."""
from pathlib import Path
import csv, datetime, fcntl, hashlib, json, os, resource, signal, subprocess, time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010')
S = W / 'sycl/tools/native-iq2s-nt1-index-spread'
G = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
HEAD = '22064c0b26232b0964afd70e7d9fe47535b39224'
GHEAD = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
CXX = Path('/opt/intel/oneapi/compiler/2026.1/bin/icpx')
CC = Path('/opt/intel/oneapi/compiler/2026.1/bin/icx')
ENV = {'PATH': str(CXX.parent) + ':/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def stat(pid):
    try:
        v = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': v[0], 'pgid': int(v[2]), 'start_ticks': int(v[19]), 'rss_bytes': int(v[21])*os.sysconf('SC_PAGE_SIZE')}
    except FileNotFoundError:
        return None

def main():
    lock = (B / 'owned-v0141-measurement.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == HEAD
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=W, text=True).strip()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=G, text=True).strip() == GHEAD
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=G, text=True).strip()
    pins = {n: sha(S/n) for n in ['CMakeLists.txt', 'iq2s_index_spread.cpp', 'README.md']}
    gpins = {n: sha(G/'ggml'/n) for n in ['src/ggml-cpu/arch/x86/quants.c', 'src/ggml-common.h', 'src/ggml-cpu/ggml-cpu.c', 'src/ggml-quants.c']}
    out = B / 'iq2s-register-index-cpu-validation-v1'
    out.mkdir(mode=0o700)
    rp, start = out/'record.json', time.monotonic()
    r = {'active': True, 'complete': False, 'passed': False, 'source_head': HEAD, 'dependency_head': GHEAD,
         'source_pins': pins, 'dependency_pins': gpins, 'controller_sha256': sha(__file__),
         'compiler_pins': {str(p): sha(p) for p in [CXX, CC]}, 'commands': [], 'cleanup': [], 'survivors': [],
         'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'previous_qualified_source_receipt_sha256': '36f1359468e6c383a749c8b36619e7ffd850c3b9e2598245ef66ac2c3cd299c8',
         'scope': 'standalone CPU synthetic three-arm exactness and sanitizer gate; no timings, model, GPU or adoption',
         'gpu_executed': False, 'model_opened': False, 'performance_eligible': False, 'adopted': False,
         'cpu_metadata': Path('/proc/cpuinfo').read_text(), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'variants': {}}
    def save():
        rp.write_text(json.dumps(r, indent=2)+'\n')
    def run(label, argv, env=ENV, wall=180, as_bytes=16<<30, rss_cap=3<<30, fsize=64<<20, cpu=180, placement=None):
        so, se = out/(label+'.stdout'), out/(label+'.stderr')
        entry = {'label': label, 'argv': argv, 'environment': env, 'wall_seconds': wall, 'AS_each_bytes': as_bytes,
                 'RSS_owned_poll_bytes': rss_cap, 'CPU_each_seconds': [cpu, cpu+1], 'FSIZE_each_bytes': fsize,
                 'NOFILE': 256, 'CORE_bytes': 0, 'cpu_placement': placement, 'owners': []}
        r['commands'].append(entry); save()
        proc, owned = None, {}
        t = time.monotonic()
        def observe():
            if proc is None:
                return []
            active = []
            for p in Path('/proc').iterdir():
                if not p.name.isdecimal(): continue
                s = stat(int(p.name))
                if not s or s['pgid'] != proc.pid or s['state'] == 'Z': continue
                key = (s['pid'], s['start_ticks'])
                if key not in owned:
                    try:
                        fd = os.pidfd_open(s['pid'])
                    except ProcessLookupError: continue
                    again = stat(s['pid'])
                    if not again or again['start_ticks'] != s['start_ticks']:
                        os.close(fd); continue
                    owned[key] = fd; entry['owners'].append(s)
                active.append(s)
            return active
        try:
            with so.open('wb') as f, se.open('wb') as e:
                def limits():
                    for kind, pair in [(resource.RLIMIT_AS, (as_bytes, as_bytes) if as_bytes is not None else (resource.RLIM_INFINITY, resource.RLIM_INFINITY)),
                                       (resource.RLIMIT_CPU, (cpu, cpu+1)), (resource.RLIMIT_FSIZE, (fsize, fsize)),
                                       (resource.RLIMIT_NOFILE, (256, 256)), (resource.RLIMIT_CORE, (0, 0))]:
                        resource.setrlimit(kind, pair)
                    if placement is not None: os.sched_setaffinity(0, {placement})
                proc = subprocess.Popen(argv, cwd=W, env=env, stdout=f, stderr=e, preexec_fn=limits, start_new_session=True)
                while True:
                    active = observe()
                    rss = sum(x['rss_bytes'] for x in active)
                    entry['peak_owned_rss_bytes'] = max(entry.get('peak_owned_rss_bytes', 0), rss)
                    assert rss < rss_cap, 'RSS limit'
                    assert time.monotonic()-t < wall, 'stage deadline'
                    assert time.monotonic()-start < 1200, 'total deadline'
                    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < 24<<20, 'text budget'
                    if proc.poll() is not None and not active: break
                    time.sleep(.05)
            entry['exit_code'] = proc.returncode
            assert proc.returncode == 0, f'{label} exit {proc.returncode}'
        finally:
            active = observe()
            for sig in [signal.SIGTERM, signal.SIGKILL]:
                if not active: break
                for s in active:
                    key = (s['pid'], s['start_ticks'])
                    if key not in owned: continue
                    try:
                        signal.pidfd_send_signal(owned[key], sig)
                        r['cleanup'].append({'label': label, 'pid': s['pid'], 'start_ticks': s['start_ticks'], 'signal': sig.name})
                    except ProcessLookupError: pass
                deadline = time.monotonic()+3
                while active and time.monotonic()<deadline:
                    time.sleep(.05); active = observe()
            r['survivors'].extend(active)
            if proc is not None and proc.poll() is not None: proc.wait()
            for fd in owned.values(): os.close(fd)
            entry.update(elapsed_seconds=time.monotonic()-t, stdout_sha256=sha(so), stderr_sha256=sha(se))
            save()
        return so
    save()
    try:
        run('compiler', [str(CXX), '--version'])
        for variant in ['release', 'asan-ubsan']:
            build = W / ('build-iq2s-register-index-'+variant+'-v1'); assert not build.exists()
            argv = ['/usr/bin/cmake', '-S', str(S), '-B', str(build), '-G', 'Ninja', '-DCMAKE_C_COMPILER='+str(CC), '-DCMAKE_CXX_COMPILER='+str(CXX), '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_SOURCE_DIR='+str(G/'ggml')]
            if variant == 'release': argv += ['-DCMAKE_BUILD_TYPE=Release']
            else: argv += ['-DCMAKE_BUILD_TYPE=Debug', '-DCMAKE_C_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer', '-DCMAKE_CXX_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer', '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined']
            run(variant+'-configure', argv)
            run(variant+'-build', ['/usr/bin/cmake', '--build', str(build), '--target', 'iq2s_index_spread', '--parallel', '2'], wall=420, cpu=360, rss_cap=4<<30)
            commands = json.loads((build/'compile_commands.json').read_text())
            assert all('-fp-model=precise' in x['command'] for x in commands)
            assert all(not any(flag in x['command'].split() for flag in ['-Ofast', '-ffast-math', '-fsycl', '-fassociative-math', '-funsafe-math-optimizations']) for x in commands)
            source_command = [x for x in commands if x['file'] == str(S/'iq2s_index_spread.cpp')]; assert len(source_command)==1
            assert all(x in source_command[0]['command'].split() for x in ['-mavx2', '-mfma', '-mf16c'])
            (out/(variant+'-compile_commands.json')).write_text((build/'compile_commands.json').read_text())
            (out/(variant+'-CMakeCache.txt')).write_text((build/'CMakeCache.txt').read_text())
            run(variant+'-target-commands', ['/usr/bin/ninja', '-C', str(build), '-t', 'commands', 'iq2s_index_spread'])
            binary = build/'iq2s_index_spread'
            imports = run(variant+'-imports', ['/usr/bin/readelf', '-d', str(binary)])
            assert not any(x in imports.read_text().lower() for x in ['libsycl', 'libur_', 'libze_loader', 'libmkl', 'libigc'])
            env = ENV if variant == 'release' else {**ENV, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'}
            log = run(variant+'-fixtures', [str(binary)], env=env, wall=180, cpu=150, as_bytes=512<<20 if variant=='release' else None, rss_cap=512<<20, fsize=2<<20, placement=min(r['cpu_affinity']))
            rows = list(csv.reader(log.read_text().splitlines()))
            assert rows[0][0]=='fp_environment' and len(rows[0])==4
            assert rows[1] == ['indices', '262144', '1024', 'pass']
            assert rows[2] == ['mixed_indices', '4194304', '1048576', '16', 'pass']
            expected = [(str(n), str(seed), str(mode)) for n in [256, 512, 2560, 8192] for seed in [1, 0x12345678, 0xdeadbeef] for mode in range(8)]
            cases = [x for x in rows if x[0]=='rows']
            assert len(rows)==101 and len(cases)==96 and [(x[1], x[2], x[3]) for x in cases]==expected
            assert all(len(x)==8 and x[4:6]==['16', 'traits_direct_index'] and x[-1]=='pass' for x in cases)
            assert rows[-2][0]=='fp_environment_end' and rows[-2][2:]==['control_unchanged','pass']
            assert rows[-1] == ['summary', 'synthetic_only', 'performance_false', 'adopted_false', 'pass']
            assert (out/(variant+'-fixtures.stderr')).stat().st_size==0
            previous = B/'iq2s-index-spread-cpu-validation-v2'/(variant+'-fixtures.stdout')
            prior = list(csv.reader(previous.read_text().splitlines()))
            assert cases == [x for x in prior if x[0]=='rows'], 'prior96profile checksums changed'
            r.setdefault('previous_profile_pins', {})[variant] = {'path':str(previous),'sha256':sha(previous),'all96profile_rows_identical':True}
            r['variants'][variant] = {'binary': str(binary), 'binary_sha256': sha(binary), 'binary_bytes': binary.stat().st_size, 'profiles': 96, 'GU_pairs': 1536, 'dot_calls': 9216, 'uniform_index_comparisons': 262144, 'mixed_active_index_comparisons': 4194304, 'mixed_vector_comparisons': 1048576, 'alignment_offsets': 16, 'complete_normal_exit': True, 'stdout_sha256': sha(log)}
        release = Path(r['variants']['release']['binary'])
        for label, symbol in [('control','(anonymous namespace)::direct_control(int, block_iq2_s const*, block_q8_K const*)'),('candidate','(anonymous namespace)::index_candidate(int, block_iq2_s const*, block_q8_K const*)'),('trait','ggml_vec_dot_iq2_s_q8_K'),('reference-quantizer','quantize_row_q8_K_ref')]:
            run('linked-'+label,['/usr/bin/objdump','-d','-C','--disassemble='+symbol,str(release)])
        run('linked-symbols',['/usr/bin/nm','-S','-C',str(release)])
        r['codegen_captured_not_interpreted']=True
        r.update(complete=True, passed=True)
    except BaseException as e:
        r['error'] = type(e).__name__+': '+str(e)
    finally:
        try:
            assert all(sha(S/n)==h for n,h in pins.items())
            assert all(sha(G/'ggml'/n)==h for n,h in gpins.items())
            assert all(sha(p)==h for p,h in r['compiler_pins'].items())
            assert sha(__file__)==r['controller_sha256']
            assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=G, text=True).strip()==GHEAD
            assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=G, text=True).strip()
            r['exit_pin_gate_passed']=True
        except BaseException as e:
            r['passed']=False; r['exit_pin_error']=type(e).__name__+': '+str(e)
        r.update(active=False, elapsed_seconds=time.monotonic()-start, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save()
    print(json.dumps({'record': str(rp), 'sha256': sha(rp), 'passed': r['passed'], 'error': r.get('error'), 'elapsed_seconds': r['elapsed_seconds']}))
    return 0 if r['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
