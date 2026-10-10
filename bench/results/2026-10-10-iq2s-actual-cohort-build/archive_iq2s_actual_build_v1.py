from pathlib import Path
import fcntl,hashlib,json,shutil,datetime
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010');A=W/'bench/results/2026-10-10-iq2s-actual-cohort-build'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);D=B/'iq2s-actual-cohort-cpu-build-v1';r=json.loads((D/'record.json').read_text());assert r['passed'] and r['complete'] and not r['active'] and not r['cleanup'] and not r['survivors'];assert not A.exists();A.mkdir(parents=True)
 build=Path(r['binary']['path']).parent;assert sha(build/'native_service_calibration')==r['binary']['sha256']
 shutil.copyfile(D/'record.json',A/'record.json')
 for p in [B/'build_iq2s_actual_cohort_cpu_v1.py',B/'make_iq2s_actual_build_v1.py',build/'compile_commands.json',build/'CMakeCache.txt']:
  shutil.copyfile(p,A/p.name)
 rows=[]
 for p in sorted(D.glob('*.std*')):
  item={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
  if p.stat().st_size:
   lines=p.read_text(errors='replace').splitlines();excerpt=lines if len(lines)<=24 else lines[:12]+['... bounded excerpt ...']+lines[-12:];name=p.name+'.excerpt.txt';(A/name).write_text('\n'.join(excerpt)+'\n');item['excerpt']=name
  rows.append(item)
 (A/'compact-log-identities.json').write_text(json.dumps(rows,indent=2)+'\n')
 p=B/'research-20261009/iq2s-real-cohort-three-arm-independent-review-round111.txt';assert sha(p)=='d65266bbb9ce14fb50d3b21502c45257af73893206a07f4cb2233169eb32186d';shutil.copyfile(p,A/p.name)
 for n in ['iq2s-actual-cohort-d68516f4','iq2s-custom-quantizer-d68516f4']:
  shutil.copyfile(B/'research-source-snapshots'/n/'manifest.json',A/(n+'-manifest.json'))
 (A/'REPORT.md').write_text('''# Actual-cohort IQ2S fixture: fresh CPU build only

Source d68516f4baae2f9ddd80d95494e6332532d4017a, IntelLLVM2026.1 precise, pinned GGML3cf03257f219afbe7334045ff7c6a06ac68c627d. The OFF-default isolated three-arm mode was enabled in a fresh private release build. Root-owned finite CPU build closed normally in29.0511s, no cleanup/survivors and all source pins unchanged. All37 common compile rules match the prior full production CPU calibration configuration after excluding only the explicitly enabled private macro; only the new isolated dot TU is additional. It uses AVX2/FMA/F16C/precise. Direct NEEDED contains no SYCL/UR/LevelZero/MKL/OpenMP imports; no runtime device tripwire has been run.

Independent read-only R111 reviewed all five changed files and found no definite source bug. Root separately verified the copied table/helper/direct/register body bytes. The initial research snapshot omitted the included production custom quantizer implementation; a separate immutable committed45-file Strata/GGML snapshot now supplies it. That gap remains a source/emission/runtime qualification task, not a hidden assumption of equivalence.

This is build evidence only. Actual weights were not opened, no executable inference/service/model was run, no speedup/actual-weight parity was measured, and no fullphysical262144 lifecycle or production default/decoder adoption is qualified. Fresh ASan/UBSan, OFF/negative/default mode checks, frozen manifest/1152 selected role content closure, actual384-expert Gate/Up bitwise results and linked custom quantizer evidence remain pending. Individual original failed/successful receipts remain unchanged. Full successful compiler repetition is represented by exact hashes and bounded excerpts.
''')
 shutil.copyfile(__file__,A/Path(__file__).name);p=A/'archive-file-identities.json';p.write_text(json.dumps({str(f.relative_to(A)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(A.rglob('*')) if f.is_file() and f!=p},indent=2)+'\n');print({'archive':str(A),'files':len(list(A.iterdir()))})
