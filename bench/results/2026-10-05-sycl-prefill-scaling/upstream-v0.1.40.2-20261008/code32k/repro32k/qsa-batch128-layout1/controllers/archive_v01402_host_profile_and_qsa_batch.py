from pathlib import Path
import datetime, hashlib, json, shutil, sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
profile_out=repro/'host-api-profile'
qsa_out=repro/'qsa-batch128-layout1'
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(name):return json.loads((base/name/'record.json').read_text())
for name in ['unitrace-host-options-v01402','host-profile32k-v01402-sequence','host-profile32k-v01402-analysis','qsa-batch128-layout1-v01402-state-sequence','qsa-batch128-layout1-v01402-clean-sequence']:
    r=load(name);assert r['passed'] and not r['active'],name
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
assert not profile_out.exists() and not qsa_out.exists()
profile_out.mkdir();qsa_out.mkdir()
def copy(p,out,dst):
    assert p.is_file() and p.stat().st_size<20*1024**2,p
    q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
private={str(profile_out):[],str(qsa_out):[]}
def artifact(p,out):
    private[str(out)].append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
def run(name,out,comparison):
    d=base/name;r=load(name)
    assert r['healthy'] and r['completed'] and not r['active'] and r['exit_code']==0
    assert not r['new_fault_messages'] and not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
    for key in ['inferior','debugger']:
        old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
    assert 'EnableImplicitConvertionToCounterBasedEvents' not in r['environment'] and 'MKL_CBWR' not in r['environment']
    assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
    assert r['requests'][0]['measurement']['generated_tokens']==64
    c=r['requests'][0][comparison]
    assert all(c[k] for k in ['ids_equal','logprobs_equal'])
    if 'first_head_equal' in c:assert c['first_head_equal'] and not c['different_prefill_state_parts']
    for f in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt','host-summary.txt']:
        p=d/f
        if p.exists():copy(p,out,'runs/'+name+'/'+f)
    for p in (d/'probes').glob('*'):
        if p.is_file():copy(p,out,'runs/'+name+'/probes/'+p.name)
    for f in ['inferior-argv.json','inferior-environment.json']:
        p=d/'debugger'/f
        if p.exists():copy(p,out,'runs/'+name+'/debugger/'+f)
    for f in ['debugger/inferior.stderr','first-head.bin','prefill-state.bin']:
        p=d/f
        if p.exists():artifact(p,out)
    for p in d.glob('strata*.json'):artifact(p,out)
for phase in ['diagnostic','state','control']:
    run(f'owned-event-ack-no-root-host-profile-v01402-code32k-{phase}-r1',profile_out,'comparison_to_default_counter_control')
for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
    run(f'owned-qsa-batch128-layout1-v01402-code32k-{phase}-r{rep}',qsa_out,'comparison_to_default_counter_control')
for mode in ['qsa-default','qsa-batch128-layout1']:
    for rep in [1,2]:run(f'owned-{mode}-v01402-code32k-clean-r{rep}',qsa_out,'comparison_to_logged_control')
for out,names in [(profile_out,['host-profile32k-v01402-sequence','host-profile32k-v01402-analysis','unitrace-host-options-v01402']), (qsa_out,['qsa-batch128-layout1-v01402-state-sequence','qsa-batch128-layout1-v01402-clean-sequence','qsa-batch128-layout1-v01402-source-review'])]:
    for name in names:copy(base/name/'record.json',out,'analysis/'+name+'.json')
copy(base/'unitrace-build-system-cc/record.json',profile_out,'profiler-build/record.json')
for folder in ['configure','compile']:
    for p in (base/'unitrace-build-system-cc'/folder).iterdir():
        if p.is_file():copy(p,profile_out,'profiler-build/'+folder+'/'+p.name)
for p in (base/'unitrace-host-options-v01402').iterdir():
    if p.is_file() and p.name!='options':copy(p,profile_out,'options-audit/'+p.name)
for out,names in [(profile_out,['audit_unitrace_host_options_v01402.py','prepare_host_profile32k_v01402.py','run_owned_host_profile32k_v01402.py','run_host_profile32k_v01402_sequence.py','analyze_host_profile32k_v01402.py']), (qsa_out,['prepare_qsa_batch128_layout1_v01402.py','run_owned_qsa_batch128_layout1_v01402_code32k.py','run_qsa_batch128_layout1_v01402_state_sequence.py','prepare_qsa_batch128_layout1_v01402_clean.py','run_owned_qsa_batch128_layout1_v01402_code32k_clean.py','run_qsa_batch128_layout1_v01402_clean_sequence.py'])]:
    for name in names:copy(base/name,out,'controllers/'+name)
    copy(Path(__file__),out,'controllers/'+Path(__file__).name)
    (out/'private-artifacts.json').write_text(json.dumps(private[str(out)],indent=2)+'\n')
a=load('host-profile32k-v01402-analysis')
whole=a['phases']['whole_request']
(profile_out/'README.md').write_text('''# Host API profile on32K input

The unchanged private e82fc5 scheduling executable runs on Arc B57010GiB,
Ryzen5 5600X,128GiB RAM, kernel7.0.0-38, NEO26.31.39395.14 and
oneAPI2026.1.1. Each fresh process reads32,768 identical code-review tokens,
context33024,8192-token chunks, int8 KV, normal MTP4 and64 greedy outputs.
Default implicit counter conversion and no MKL CNR match the
[three completed controls](../registered-copy-default-counter-conversion/README.md).

The locally built [Intel PTI unitrace source](https://github.com/intel/pti-gpu/tree/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033)
is clean at the pinned commit. A CPU fixture compiles the actual collector
option derivation and checks host timing plus Chrome call logging: API tracing
is enabled; kernel tracing, device timing and metrics remain disabled.
Option-free unitrace would also enable device/kernel tracing and is not used.
Source hashes and generated callback guards are retained in options-audit.
The chosen command uses `-h --chrome-call-logging --output-dir-path ... -o ...`;
the wrapper execs the engine under the owned debugger without preloading GDB
or its helper. The actual target executable hash/PID identity and environment
are captured. This host mode does not install timestamp event-pool or kernel
append callbacks. No GPU event or hardware-counter trace is collected.

First logged/validated32K use, a quiet host profile, and a fresh unprofiled
control all match all66state parts,248,320first-head float bytes,64IDs and all
logprobs. All exit normally, complete owned cleanup and record no new xe fault.
Both quiet profile and control retain state/head capture; they are not clean
performance measurements. Their single-pair prompt durations differ by
about0.18%, decode by about2.93%. This does not establish negligible overhead
for other workloads.

The quiet trace has4,824,682 native host calls and zero GPU events. In the
approximately84.59-second externally observed request envelope, it contains
2,231,722 zeEventQueryStatus calls across three worker threads and503,650
zeCommandListAppendLaunchKernelWithArguments calls. The interval union of
native synchronization calls is about27.53s; status queries about38.78s;
append calls about10.86s. These overlap one another and GPU work. Their
inclusive concurrent sums must not be added or treated as CPU utilization,
GPU busy time or PCIe transfer duration. AppendMemoryCopy host duration
does not measure DMA completion or bytes transferred.

The pinned timer derives its epoch offset from BOOTTIME but stamps API calls
with MONOTONIC_RAW. Analysis adds the measured boot/raw difference (about1.49s)
before aligning to observed protocol times. Calibration was taken after the
job, so phase boundaries are approximate; +/-10ms sensitivity is recorded.
Native durations are unchanged. No raw trace is rewritten. See the complete
[analysis](analysis/host-profile32k-v01402-analysis.json) for counts, per-thread
intervals and limits. All status polling and almost all native kernel appends
occur before the observed last prefill progress message.

These measurements motivate separate launch batching and polling experiments;
they do not identify a dominant transfer/CPU/GPU bottleneck, prove general
hang prevention, meet PP1000/TG70, or validate full262,144-cell serving.
The executable remains private and unadopted. Public files retain exact argv,
environment, controllers, protocol, journal, state/head hashes and profiler
build receipts. Large API/state/head/Chrome payloads stay private with byte
counts and SHA256 hashes. No reset, rebind, reboot, package, service or global
setting is changed.
''')
seq=load('qsa-batch128-layout1-v01402-clean-sequence')
means=seq['clean_mean'];gain=seq['relative_change']
table='| Setting | Prefill token/s | Decode token/s |\n| --- | ---: | ---: |\n'
for mode in ['qsa-default','qsa-batch128-layout1']:
    for s in seq['steps']:
        if s['argv'][2]==mode:table+=f"| {mode}, run{s['argv'][-1]} | {s['measurement']['prefill_tok_s']:.3f} | {s['measurement']['decode_tok_s']:.3f} |\n"
    table+=f"| {mode}, mean | {means[mode]['prefill_tok_s']:.3f} | {means[mode]['decode_tok_s']:.3f} |\n"
(qsa_out/'README.md').write_text('''# Attention batch128/layout1 on32K input

This environment-only experiment uses the same private e82fc5 executable,
Arc B57010GiB / Ryzen5 5600X /128GiB host and32K input/configuration as the
[host-only profile](../host-api-profile/README.md). The two changed settings
are STRATA_PREFILL_ATTN_BATCH=128 and STRATA_PREFILL_ATTN_LAYOUT=1. Subgroup32,
workgroup256, KV format, ordered dot/reductions/accumulation/merge, kernel
source, queue properties, graphs, drains and all other settings are unchanged.
Layout1 transposes local query storage then reconstructs the same eight
operands; it avoids that branch's float-array to float4 pointer casts. No
SG16, tensor/XMX attention, approximate selection or retirement is enabled.
Attention scratch has four times as many per-query slots. The engine shares
a region sized to the largest phase workspace, so that is not by itself a
fourfold increase in total VRAM allocation. Both startup INFO records report
830MiB free before the request; those snapshots do not measure peak usage.
The [source review](analysis/qsa-batch128-layout1-v01402-source-review.json)
records disjoint per-query scratch, uniform barriers and the127-query tail of
the8191-token final chunk. Source review is not a general runtime UB proof.

The first logged/validated32K run and two fresh32K captured repeats match
all66prefill state parts, all248,320first-head floats,64 output IDs and every
logprob with the completed default32/layout0 control. These durations are not
speed evidence. Only after those gates, four fresh processes run a clean ABBA
comparison: default/128-layout1/128-layout1/default. All read32,768 tokens,
use8192-token chunks, context33024, int8 KV, normal MTP4 and64 greedy outputs.
Debug logs, validation, state/head dumps, transfer timing, profilers and extra
waits are absent. Both implicit-conversion override and MKL CNR are absent.
An uninterrupted owned GDB/PTY observer is common to both settings. Every
clean job matches all64IDs and all logprobs; all seven jobs exit normally,
complete owned cleanup and record no new xe fault.

'''+table+f'''
The two-run mean changes by{gain['prefill_tok_s']*100:+.3f}% for prefill and
{gain['decode_tok_s']*100:+.3f}% for decode. Two runs per setting cannot establish
a small improvement outside observed variation, and a prefill-only setting
does not by itself explain decode variation. See the
[clean sequence](analysis/qsa-batch128-layout1-v01402-clean-sequence.json)
for every duration, environment, gate and ordering. These are measurements
of private settings, not an upstream-alone equivalence claim.

No production executable or default setting changes. Full262,144-cell
occupancy/repeat/restore/clipped-tail/refusal/later-valid gates and PP1000/TG70
remain open. No general hang-prevention claim is made. Large API/state/head
payloads remain private with hashes; exact controllers, configuration,
fixture, protocol, process ownership and journal records are public.
''')
with (repro/'README.md').open('a') as stream:
    stream.write('''

The [host-only32K profile](host-api-profile/README.md) then passes three exact
full-state/head/output checks while capturing native host API counts without
GPU event instrumentation. Concurrent API durations do not identify PCIe or
CPU dominance. A separate [attention128/layout1 experiment](qsa-batch128-layout1/README.md)
passes three full-state/head controls and a clean32K ABBA comparison on the
unchanged private scheduling binary. Every performance input is32,768 tokens.
Full262,144-cell serving and production adoption remain separate pending gates.
''')
for directory in [profile_out,qsa_out,repro,parent/'code32k',parent]:
    p=directory/'manifest.json'
    r=json.loads(p.read_text()) if p.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    r.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append host-only32K profile and gated attention128/layout1 clean comparison.',files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p})
    p.write_text(json.dumps(r,indent=2)+'\n')
    for name,item in r['files'].items():
        f=directory/name;assert f.stat().st_size==item['bytes'] and digest(f)==item['sha256']
    print(directory.name,len(r['files']))
