#!/usr/bin/env python3
"""Compile/run the host-only public-TLA FP32 D-copy ownership proof; MIT."""
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import sys

sys.dont_write_bytecode = True
import check_gdn_projection_fp32 as image

base, RESULT = image.base, image.RESULT
ROOT = image.ROOT / 'ownership-v3'
KEY = 'official_gdn_fp32_projection_ownership_v3'
FIXTURE = 'bench/sycl/gdn_projection_fp32.cpp'
require, digest, write = base.require, base.digest, base.write


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = base.compact(row); write(RESULT, data)


def prepare():
    data = base.committed(RESULT); prior = json.loads(image.RECORD.read_text())
    failed = data['official_gdn_fp32_projection_ownership_v2']
    require(failed['stage'] == 'failed' and not failed['passed'] and failed['ownership_closed'] and
            'subgroup accumulator/payload/destination' in failed['error'], 'preserve original descriptor-proof failure')
    require(KEY not in data and not ROOT.exists() and data[image.KEY] == base.compact(prior) and
            prior['passed'] and prior['compiler_inputs_attested'] and prior['ownership_closed'] and
            not prior['active'] and prior['codegen_member']['manual_FP32_store_review_passed'],
            'closed attested original object and direct FP32 codegen required')
    revision = base.basis.build.git(base.P, 'rev-parse', 'HEAD').decode().strip()
    sha = hashlib.sha256(base.basis.build.git(base.P, 'show', revision + ':' + FIXTURE)).hexdigest()
    require(digest(base.P / FIXTURE) == sha, 'commit exact host fixture')
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache'): (ROOT / part).mkdir()
    tla = '/home/yayoi/ghq/github.com/intel/sycl-tla/.worktree/strata-native16-87f/include'
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=Path(__file__).resolve()),
        native_sources={FIXTURE:sha},
        compiled_native_implementation='e618f3466432dca7ee7435a399e7eb92ecfc06c3',
        image_proof=dict(path=str(image.RECORD), sha256=digest(image.RECORD)),
        supersedes=dict(key='official_gdn_fp32_projection_ownership_v2', original_outcome='failed',
            reason='Enumerate one8x16 collective write rectangle per8-value atom descriptor,not descriptor coordinates as scalar destination cells.'),
        commands=[('compile',[base.COMPILER, '-O2', '-std=c++20', '-fno-sycl',
            '-fno-fast-math', '-ffp-contract=off', '-DCUTLASS_ENABLE_HEADERS_ONLY',
            '-DCUTLASS_ENABLE_SYCL', '-DSYCL_INTEL_TARGET',
            '-isystem', '/opt/intel/oneapi/compiler/2026.2/include', '-isystem', tla,
            '-MD', '-MF', str(ROOT / 'inputs.d'), '-Wl,-t', str(base.P / FIXTURE),
            '-o', str(ROOT / 'ownership')]), ('host-proof',[str(ROOT / 'ownership')])],
        environment=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                         TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        budgets=dict(wall=180,cpu=360,rss_cap=3<<30,text_cap=8<<20,file_cap=128<<20,total_cap=512<<20),
        plan=dict(scope='CPU-only coordinate/descriptor proof,not device masking/numerical/model qualification.',
            acceptance='Actual public TLA SG accumulator/payload/D-copy cells agree and uniquely cover128x128;all T1..8192,both N6144/10240,and four pitches conserve valid writes,descriptor bounds and spans;GPU_submissions0.',
            numerical_gate='Not exercised here;prospective guarded device oracle and checked same-Runtime event/queue retirement remain required.'),
        active=False,ownership_closed=True,commands_executed=[],passed=False,GPU_executed=False)
    save(row); print('Prepared host-only TLA ownership compile/run;commit before execution.',flush=True)


def run():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==base.compact(row) and row['stage']=='prepared' and
            not (ROOT/'owned-build').exists(), 'fresh committed ownership plan required')
    require(base.controls(row['source_commit'],controller=Path(__file__).resolve())==row['controller_inputs'],
            'host-proof controls changed')
    ref=row['image_proof'];require(digest(ref['path'])==ref['sha256'],'frozen image proof changed')
    prior=json.loads(Path(ref['path']).read_text()); original=image.admission(prior)
    allowed=image.build.admit(original)
    full,_=image.build.source.sources()
    for group in full['dependencies']:
        for n,sha in group['source_sha256'].items():
            path=str((Path(group['root'])/n).resolve());require(digest(path)==sha,'pinned header changed');allowed[path]=sha
    for n,sha in row['native_sources'].items():
        require(digest(base.P/n)==sha,'host fixture changed');allowed[str(base.P/n)]=sha
    owner=None
    def persist():
        write(ROOT/'record.json',dict(row,commands_executed=owner.commands if owner else [],active=bool(owner and owner.active)))
    def monitor(io,entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(base.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'host-proof artifact budget')
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        row['stage']='running';owner=base.supervision.Owner(ROOT/'owned-build',persist)
        entry,out,err=owner.run('compile',row['commands'][0][1],row['environment'],ROOT,interaction=monitor,**row['budgets'])
        require(base.supervision.closed(entry) and entry['exit_code']==0,
                'host compile failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-5000:])
        found=set()
        for line in (ROOT/'inputs.d').read_text().replace('\\\n','').splitlines():
            if not line.strip():continue
            left,colon,right=line.partition(':');require(colon and left.strip(),'invalid host depfile')
            found.update(str((ROOT/p).resolve()) for p in shlex.split(right))
        require(all(allowed.get(p)==digest(p) for p in found),'uncovered host compiler input')
        write(ROOT/'compiler-inputs.json',{p:allowed[p] for p in sorted(found)})
        row['actual_compiler_inputs']=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(found),uncovered=0)
        row['link_inputs']=base.traced_link_inputs(allowed,ROOT)
        row['binary']=dict(path=str(ROOT/'ownership'),sha256=digest(ROOT/'ownership'),bytes=(ROOT/'ownership').stat().st_size)
        entry,out,err=owner.run('host-proof',row['commands'][1][1],row['environment'],ROOT,interaction=monitor,**row['budgets'])
        require(base.supervision.closed(entry) and entry['exit_code']==0,
                'host ownership failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-3000:])
        result=json.loads(out.read_text());require(result['ownership']=='passed' and result['checked_shapes']==65536 and
            result['GPU_submissions']==0 and not result['physical262144_qualified'],'incomplete host ownership proof')
        row.update(stage='host-ownership-passed',passed=True,result=result)
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=False,ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)),commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None,'host proof owner unresolved');save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','result','error')}),flush=True)
    require(row['passed'],'host ownership proof failed')


if __name__=='__main__':
    with (base.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare()
        elif sys.argv[1:]==['run']:run()
        else:raise RuntimeError('check_gdn_projection_ownership.py prepare|run')
