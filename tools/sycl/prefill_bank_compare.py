#!/usr/bin/env python3
"""One-knob prefill bank comparison on the retained coherent fast engine.

Main-authored, repository LICENSE. Only startup prefill bank slots change;
native math, decode configuration and placement remain fixed.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode=True
import current_request_profile as prior

P,B,RESULT=prior.P,prior.B,prior.RESULT
ROOT,KEY=B/'prefill-bank512-compare-v1','prefill_bank512_compare_v1'
THIS,TEST='tools/sycl/prefill_bank_compare.py','tools/test_prefill_bank_compare.py'
# First process tests candidate capacity before spending time on controls.
BANKS=(512,384,384,512,512,384)
ORDER=[(32769,1)]*len(BANKS)
RUNS=3
require,sha,signature=prior.require,prior.sha,prior.signature
compact=prior.compact
configuration=prior.configuration
check_sample=prior.check_sample
check_native_environment=prior.check_native_environment
HOST_COMMAND=['/home/yayoi/.local/share/strata-sycl/venv/bin/python3','-B','-m','unittest',
    'tools.test_prefill_bank_compare','tools.test_current_request_profile','tools.test_request_profile_reader']


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    data=json.loads(RESULT.read_text());data[KEY]=compact(row)
    RESULT.write_text(json.dumps(data,indent=2)+'\n')


def native_context(row,plan):
    require(type(plan['index']) is int and 0<=plan['index']<len(BANKS)
        and plan['expected_bank_slots']==BANKS[plan['index']] and plan['arm']=='clean'
        and (plan['read'],plan['heads'])==ORDER[plan['index']],'declared bank arm required')
    return {k:row[k] for k in ('binary','native_source_commit','native_validation_probe','boot')}


def launch_plan(row,plan):
    return {k:v for k,v in native_context(row,plan).items() if k in ('binary','native_validation_probe')}


def environment(row,plan):
    native_context(row,plan)
    env=prior.environment(row,plan)
    env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS']=str(plan['expected_bank_slots'])
    return env


def request_route(text,plan):
    route=prior.fast_request_route(text,plan)
    banks=re.findall(r'^strata MoE expert bank: (.*)$',text,re.M)
    require(len(banks)==1,'one actual bank admission required')
    fields=dict(re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',banks[0]))
    slab=int(fields['host_slab_bytes']);device=int(fields['device_bytes'])
    require(slab>0 and slab%512==0 and device==slab//512*plan['expected_bank_slots'],
        'actual bank allocation must match declared slots and model stride')
    require('softmax=parallel-exp-ascending-sum' in text and 'softmax=subgroup' not in text,
        'fixed ordered-exp math required')
    require(route['fast_window']['prefill_slots']==plan['expected_bank_slots']
        and route['residual_gpu_rows']==plan['read']-1 and route['residual_host_bytes']==0,
        'no reduced-residency or bank-capacity fallback permitted')
    return route


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(plan['port']),
        '--prefill-exact-tokens',str(plan['read']),'--think','--runs',str(RUNS),
        '--profile-request-gates','--workload-id','prefill_bank_capacity_20261115',
        '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]


def current(row):
    require(prior.admission.committed_json(RESULT,P)[KEY]==compact(row),'commit exact prior bank stage')
    require(all(sha(P/n)==h for n,h in row['sources'].items()),'frozen bank source changed')
    admitted,seed,gate=prior.basis()
    require({k:v for k,v in admitted.items() if k!='controller_commit'}==
        {k:v for k,v in row['admission'].items() if k!='controller_commit'}
        and signature(prior.core.compact(seed))==row['configuration_evidence_sha256']
        and gate==row['small_probe_evidence_sha256'],'exact native/configuration qualification changed')
    contract,python=prior.runner.entry.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected'],'qualified Python changed')
    for plan in row['plans']:
        root=Path(plan['root'])
        require(plan['command']==command(plan)
            and json.loads((root/'config.json').read_text())==configuration(row,plan)
            and json.loads((root/'plan.json').read_text())==launch_plan(row,plan)
            and signature(environment(row,plan))==plan['environment_sha256']
            and all(sha(root/n)==h for n,h in plan['generated'].items()),'frozen generated bank inputs changed')
    require(prior.runner.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes'],'bank output budget exceeded')
    prior.runner.shared.models.model_pins(row['configuration'],row['model_assets'],full=False)
    return json.loads(Path(row['admission']['runtime_manifest']['path']).read_text()),python


def decision(processes):
    require(len(processes)==len(BANKS) and all(p['passed'] and p['complete'] and p['ownership_closed']
        and p['native_exit_code']==0 and not p['fresh_faults'] and len(p['samples'])==RUNS
        for p in processes),'six normal independent processes required')
    require(len({(p['native_identity']['pid'],p['native_identity']['start_ticks']) for p in processes})==len(BANKS),
        'native identity cannot be reused')
    pairs=((0,1),(3,2),(4,5))
    ratios=[processes[c]['subsequent_median_ms']/processes[a]['subsequent_median_ms'] for c,a in pairs]
    ttft=[statistics.median(s['ttft_s'] for s in processes[c]['samples'][1:])/
          statistics.median(s['ttft_s'] for s in processes[a]['samples'][1:]) for c,a in pairs]
    uploads=[statistics.median(r['fast_window']['requested_bank_H2D_bytes'] for r in p['routes']) for p in processes]
    return dict(paired_candidate_over_reference_prompt_time=ratios,paired_TTFT_time=ttft,
        all_pairs_favor512=all(v<1 for v in ratios),prefill_bank_requested_H2D_bytes=uploads,
        all_pairs_reduce_requested_bank_bytes=all(uploads[c]<uploads[a] for c,a in pairs),
        advance_to_length_and_decode_checks=all(v<1 for v in ratios),
        inference_adopted=False,sustained_decode_qualified=False,physical262144_qualified=False,
        hardware_saturation_qualified=False,
        next='Only a reproducible32K gain permits a separately committed64K/capacity and sustained-decode check. '
             'Keep384outside qualified shapes;do not combine kernel,chunk or decode changes.')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh bank plan only')
    admitted,seed,gate=prior.basis();revision=prior.admission.pti.git(P,'rev-parse','HEAD').decode().strip()
    sources=prior.source_pins(revision)
    for name in (THIS,TEST):
        require((P/name).read_bytes()==prior.admission.blob(P,revision,name),'commit actual bank controls')
        sources[name]=sha(P/name)
    contract,python=prior.runner.entry.python_manifest()
    row={k:copy.deepcopy(seed[k]) for k in ('configuration','environment_template','template','boot','driver','model_assets','native_validation_probe')}
    row['model_assets']=prior.runner.shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
    row.update(stage='prepared',source_commit=revision,sources=sources,admission=admitted,
        binary=admitted['binary'],native_source_commit=admitted['native_source_commit'],
        configuration_evidence_sha256=signature(prior.core.compact(seed)),small_probe_evidence_sha256=gate,
        python_contract=contract,python_selected=python['selected'],plans=[],processes=[],request_hashes={},
        host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,whole_bytes=512<<20,
        budgets=dict(wall=1500,cpu=9000,rss_cap=104<<30,text_cap=16<<20,file_cap=16<<20,total_cap=64<<20),
        plan=dict(unit='Only STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS384versus512;existing full-fit512 code. '
                      'Required benchmark-reader parameter is part of measurement admission,not a native change.',
            mechanism='Current whole-request profile has about10.31s copy union,28.66s kernel union and4ms overlap. '
                'Existing383permanent+one scratch bank can reupload experts across MoE windows;512full-fit can '
                'retain every expert within a layer. Larger VRAM allocation must actually admit all32768GPU residual rows. '
                'Requested copy-volume reduction is tested independently of speed;physical PCIe saturation unknown.',
            order='Six fresh processes512,384,384,512,512,384 at actual32769inputs;three fresh requests/process, '
                'exact24outputs and thinking on. Pairs0/1,3/2,4/5;first-use separate from subsequent-two medians.',
            constants='Same fast14508/dcb8 ELF,providers,safety,logical262144/resident20480K8V8,8192chunk, '
                'ordered-exp/bulk1/largeGU/source-view1/staging0,head2/dense1,MTPspec4,1256decode slots,5workers. '
                'No process environment mutation,decode-placement change,new kernel or inference math.',
            acceptance='Host controls then exact source/build/ELF/assets,actual requested/effective512or384slots, '
                'all residuals onGPU,unchanged target/MTP restoration,normal0/closed/no warning/fault. '
                'All3paired warmed prompt ratios<1 permits next shape/decode checks,without a minimum gain. '
                'Report TTFT,first-use and requested bank bytes separately;no speed-based sample exclusion.',
            stopping='Stop first admission/owner/source/token/route/validation/fault/budget failure. No retry or '
                'changed hashes. Commit each closed process before the next. No default/adoption/full-context or '
                'sustainedTG claim from this screen.'),
        retention='Retain individual18samples,actual bank/copy/placement/handoff outcomes and limits. Retire '
            'repeated routes/logs/caches/private receipts at closure;no new native build generation.')
    ROOT.mkdir(mode=0o700)
    for i,bank in enumerate(BANKS):
        root=ROOT/str(i);root.mkdir()
        for name in ('tmp','cache'):(root/name).mkdir()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,arm='clean',read=32769,heads=1,expected_bank_slots=bank,root=str(root),port=port)
        (root/'config.json').write_text(json.dumps(configuration(row,plan),indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(launch_plan(row,plan),indent=2)+'\n')
        plan.update(command=command(plan),environment_sha256=signature(environment(row,plan)),
            generated={name:sha(root/name) for name in ('config.json','plan.json')})
        row['plans'].append(plan)
    save(row);print('Prepared one-knob bank comparison;commit plan before host/native work.',flush=True)


def host():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='prepared' and not row['host_passed'],'one fresh host gate')
    owner=prior.runner.build.supervision.Owner(ROOT/'host')
    entry,out,err=owner.run('unit',HOST_COMMAND,dict(PATH='/usr/bin:/bin',LANG='C',PYTHONDONTWRITEBYTECODE='1'),P,
        wall=90,cpu=60,rss_cap=1<<30,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
    text=out.read_text()+err.read_text()
    good=prior.runner.build.supervision.closed(entry) and entry['exit_code']==0 and text.rstrip().endswith('OK')
    row['host_evidence']=dict(passed=bool(good),exit_code=entry['exit_code'],
        ownership_closed=bool(prior.runner.build.supervision.closed(entry)),command=HOST_COMMAND)
    row.update(host_passed=bool(good),stage='ready' if good else 'failed');save(row);require(good,text[-4000:])


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host','0','1','2','3','4','5'),'prepare|host|0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prior.admission.pti.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare':prepare()
        else:
            with (ROOT/(action+'-attempt.json')).open('x') as stream:json.dump(dict(action=action),stream)
            if action=='host':host()
            else:prior.runner.run(int(action),sys.modules[__name__])
