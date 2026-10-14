#!/usr/bin/env python3
"""MAIN ONLY, after exclusive build/measurement ownership. Does not run the probe.

Build original exact helper plus the standalone candidate, without applying source.diff.
Use an empty output directory owned by this experiment. Existing binaries are not reused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

A = Path(__file__).resolve().parent
P = A.parents[3]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser = argparse.ArgumentParser()
parser.add_argument('--compiler',required=True)
parser.add_argument('--ggml-root',required=True,type=Path,help='.../ggml directory, with include/ and src/')
parser.add_argument('--cpu-archive',required=True,type=Path)
parser.add_argument('--base-archive',required=True,type=Path)
parser.add_argument('--out-dir',required=True,type=Path)
parser.add_argument('--sanitize',action='store_true')
args = parser.parse_args()
out = args.out_dir.resolve()
out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()): raise SystemExit('output directory must be empty')
pins = json.loads((A/'pins.json').read_text())
for path,wanted in pins['source_sha256'].items():
    if sha(path) != wanted: raise SystemExit(f'pin changed: {path}')
if args.ggml_root.resolve() != Path(pins['configured_dependency']).resolve():
    raise SystemExit('GGML dependency must match pins.json for this proposal')
manifest = json.loads((A/'manifest.json').read_text())
for filename,wanted in manifest['artifact_sha256'].items():
    if sha(A/filename) != wanted: raise SystemExit(f'artifact changed: {filename}')
flags = ['-std=c++20','-march=znver3','-mtune=znver3','-mavx2','-mfma','-mf16c',
         '-fno-fast-math','-fno-associative-math','-ffp-contract=off','-Wall','-Wextra',
         '-I'+str(P/'include'),'-I'+str(P/'third_party/ggml'),'-I'+str(args.ggml_root/'include')]
flags += ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if args.sanitize else ['-O3','-DNDEBUG']
source = [P/'src/kernels/cpu/iq_single_avx2.cpp',A/'probe_candidate.cpp',A/'fixture_bench.cpp']
commands = []
for s in source:
    commands.append([args.compiler,*flags,'-c',str(s),'-o',str(out/(s.stem+'.o'))])
commands.append([args.compiler,*flags,*[str(out/(s.stem+'.o')) for s in source],
    str(args.cpu_archive.resolve()),str(args.base_archive.resolve()),'-lpthread','-lm','-ldl',
    '-o',str(out/'fixture')])
receipt = {'status':'building','commands':commands,'cwd':str(P),'sanitized':args.sanitize,
    'compiler_sha256':sha(args.compiler),'source_sha256':{str(s):sha(s) for s in source},
    'archive_sha256':{str(s.resolve()):sha(s) for s in [args.cpu_archive,args.base_archive]},
    'source_diff_sha256':manifest['source_diff_sha256'],
    'limits':'CPU-only candidate/helper. Archives need independent Main source/compile provenance; prebuilt archives are unsanitized.'}
(out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
version = subprocess.run([args.compiler,'--version'],capture_output=True,text=True,check=True,timeout=15)
receipt['compiler_version'] = version.stdout+version.stderr
try:
    for i,cmd in enumerate(commands):
        with (out/f'compile-{i}.log').open('w') as log:
            subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    receipt.update(status='built_unrun',binary_sha256=sha(out/'fixture'))
except Exception as exc:
    receipt.update(status='build_failed',error=repr(exc))
    raise
finally:
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
