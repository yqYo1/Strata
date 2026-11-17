#!/usr/bin/env python3
"""Owned same-build physical-fill diagnostic; Main, repository LICENSE.

Uses the coherent14508 executable and its actual native16 allocation contract.
Synthetic IDs test extent/completion, not task quality, decode speed or RESTORE.
No XeStrata source, native rebuild, new kernel or production selection change.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import coherent_original_reuse as original
import coherent_prefill_bench as reference
import compare_official_prefill as comparison
import qsa16_stream_bench as core
import ur_validation as validation

P, B, RESULT = reference.P, reference.B, reference.RESULT
ROOT, KEY = B/'coherent-context-model-v1', 'coherent_context_model_v1'
THIS = 'tools/sycl/coherent_context_model.py'
POLICIES = ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md')
CASES = (('short', 16385, 4, 900), ('full', 262144, 0, 2400))
MARKER = ('strata FULL_CONTEXT consumed_position=262144 prompt_tokens=262144 '
          'output_tokens=0 window_width=1 commit_count=1 commit_wait=PASS refill_drain=PASS')
require, sha, signature = original.require, original.sha, original.fingerprint
compact = core.compact
owner_module = comparison.build.supervision


def save(row):
    core.write(ROOT/'record.json', row)
    values = json.loads(RESULT.read_text()); values[KEY] = compact(row); core.write(RESULT, values)


def basis(expected=None):
    native, proof = original.verify_original(expected=expected)
    prior = json.loads((reference.ROOT/'record.json').read_text())
    require(validation.committed(RESULT, (reference.KEY,)) == compact(prior)
        and prior['stage'] == 'closed' and not prior['active']
        and all(prior[k] for k in ('passed','complete','ownership_closed'))
        and prior['binary'] == native['binaries']['strata']
        and prior['native_source_commit'] == original.SOURCE, 'closed coherent speed reference required')
    manifest = prior['runtime_manifest']; files = json.loads(Path(manifest['path']).read_text())
    require(sha(manifest['path']) == manifest['sha256'] and len(files) == manifest['files']
        and all(sha(p) == h for p,h in files.items())
        and comparison.build.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'original providers, boot or driver changed')
    comparison.shared.models.model_pins(prior['configuration'], prior['model_assets'], full=False)
    return native, proof, prior


def token_bytes(n):
    require(type(n) is int and n in (16385, 262144), 'unplanned diagnostic input extent')
    return ''.join(str(i % 4096)+'\n' for i in range(n)).encode()


def environment(row, case):
    env = comparison.environment(row, dict(root=case['root'], heads=1))
    probe = row['native_validation_probe']
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE='0', STRATA_TRACE='1',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0', STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        TMPDIR=case['root']+'/tmp', XDG_CACHE_HOME=case['root']+'/cache',
        LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    env.pop('UR_LOG_TRACING', None)
    for name in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
        env[name]='level:warning;flush:warning;output:stderr'
    require(env['STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS']=='384'
        and env['STRATA_SYCL_PREFILL_QSA_MATH']=='bf16-prefix-ordered-exp', 'preserve reference math/capacity')
    return env


def command(row, case):
    args = row['configuration']['args']
    require(all(flag not in args for flag in ('--serve','--tokens-file','--max-new','--consume-prompt')),
        'reference arguments contain a different entrypoint')
    for flag,value in (('--max-context','262144'),('--kv-resident','20480'),
        ('--expert-cache','1256'),('--spec','4'),('--suffix-draft','0'),('--prefill','8192')):
        require(args.count(flag)==1 and args[args.index(flag)+1]==value, 'fixed configuration differs: '+flag)
    result = [row['binary']['path'], *args, '--tokens-file', case['fixture']['path'],
              '--max-new', str(case['outputs'])]
    return result + (['--consume-prompt'] if case['name']=='full' else [])


def one(text, pattern, label):
    found = list(re.finditer(pattern, text, re.M))
    require(len(found)==1, 'missing/duplicate '+label)
    return found[0]


def evidence(stdout, stderr, row, case, pid):
    proof = validation.preload_evidence(stderr, pid)
    boundary = one(stderr, r'^strata UR validation: .*phase=before-main[^\n]*$', 'pre-main boundary')
    body = stderr[boundary.end():]
    require(proof['loader']==row['native_validation_probe']['loader']['path']
        and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', body),
        'provider activation or runtime validation failed')
    m = one(body, r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(\S+) '
        r'residual_element_bytes=(\d+) residual_total=(\d+) residual_gpu_rows=(\d+) '
        r'residual_gpu=(\d+) residual_host=(\d+) residual_staging=(\d+) layer_cache=(\d+);', 'residual ledger')
    fields = ('rows','chunk','dtype','element_bytes','total','gpu_rows','gpu_bytes','host_bytes','staging_bytes','bank_bytes')
    ledger = {k:(v if k=='dtype' else int(v)) for k,v in zip(fields,m.groups())}
    n = case['tokens']-1
    require(ledger['rows']==n and ledger['chunk']==8192 and ledger['dtype']=='bf16'
        and ledger['element_bytes']==2 and 0<=ledger['gpu_rows']<=n
        and ledger['total']==n*10240*2
        and ledger['gpu_bytes']==ledger['gpu_rows']*10240*2
        and ledger['host_bytes']==(n-ledger['gpu_rows'])*10240*2 and ledger['staging_bytes']==0,
        'actual BF16 row/byte conservation failed')
    bank = one(body, r'^strata MoE expert bank: requested_slots=(\d+) effective_slots=(\d+) staged=([01]) '
        r'host_slab_bytes=(\d+) device_bytes=(\d+) residual_gpu_rows=(\d+) allocation_limit_bytes=(\d+)', 'expert bank')
    requested, slots, staged, slab, size, gpu, limit = map(int,bank.groups())
    require(requested==384 and slots==(384 if staged else 512)
        and (not staged or gpu==n) and slab==512*2662400
        and size==slots*2662400==ledger['bank_bytes']>0
        and size<=limit and gpu==ledger['gpu_rows'] and ledger['gpu_bytes']<=limit,
        'actual bank or per-allocation bound differs')
    window = one(body, r'^strata MoE window configuration: requested_mode=10 effective_mode=(2|10) '
        r'maximum_rows=(\d+) mixed_fp16_bytes=(\d+)$', 'window policy')
    qsa = one(body, r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
        r'KV=active-logical-prefix transient_bytes=(\d+); source=resident-Mode0 score_pitch=33 '
        r'operand_layout=row-major softmax=parallel-exp-ascending-sum$', 'actual QSA math')
    prefix_bytes = ((n+15)//16)*16*2056
    require(int(qsa[1])==prefix_bytes<=limit
        and 'strata prefill: native16 BF16 HC residual/projection IO and MoE outputs;' in body
        and 'MiB free after releasing the standalone prompt buffers' in body,
        'actual typed prefix allocation or standalone release missing')
    p = one(stdout, r'^prefill\s+(\d+) tokens in ([0-9.]+) ms', 'batched extent')
    require(int(p[1])==n and float(p[2])>0, 'batched token extent differs')
    output = one(stdout, r'^output\s*:(.*)$', 'output IDs')
    ids = [int(t) for t in output[1].split()]
    require(len(ids)==case['outputs'] and all(0<=t<row['vocab_size'] for t in ids), 'output extent/IDs differ')
    markers = [line for line in stdout.splitlines() if line.startswith('strata FULL_CONTEXT ')]
    require(markers==([MARKER] if case['name']=='full' else []), 'final physical-slot commit marker differs')
    free = [int(v) for v in re.findall(r'^strata trace: (\d+) MiB free after ',body,re.M)]
    return dict(validation=proof,actual_batched_rows=n,generated_ids=len(ids),
        final_slot_consumed=case['name']=='full',allocation=ledger,
        bank=dict(slots=slots,staged=bool(staged),device_bytes=size,max_single_allocation_bytes=limit),
        window=dict(effective_mode=int(window[1]),maximum_rows=int(window[2]),mixed_fp16_bytes=int(window[3])),
        qsa_prefix_bytes=prefix_bytes,minimum_logged_free_mib=min(free) if free else None,
        prefill_ms_observed=float(p[2]),speed_eligible=False,
        scope='Standalone checked capacity/completion only; host-tail residency allowed. No serve allocation, RESTORE, task/numerical quality, ordinary provider teardown or sustained decode qualification.')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh coherent diagnostic only')
    native, proof, prior = basis()
    revision = validation.git('rev-parse','HEAD').decode().strip()
    sources = core.pins(revision)
    vocab = json.loads((Path(prior['configuration']['tokenizer'])/'vocab.json').read_text())
    require(set(range(4096))<=set(vocab.values()), 'synthetic IDs outside original vocabulary')
    row = {k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','model_assets',
        'native_validation_probe','boot','driver','runtime_manifest')}
    row.update(stage='prepared',source_commit=revision,sources=sources,
        policies={p:sha(P/p) for p in POLICIES},original_native_proof=proof,
        reference_evidence_sha256=signature(compact(prior)),
        binary=native['binaries']['strata_full_context_diagnostic'],
        host_binary=native['binaries']['full_context_diagnostic_host_test'],native_source_commit=original.SOURCE,
        vocab_size=max(vocab.values())+1,cases=[],outcomes={},host_passed=False,
        active=False,ownership_closed=True,passed=False,complete=False,production_selected=False,
        model_route_qualified=False,physical262144_RESTORE_qualified=False,
        ordinary_provider_teardown_qualified=False,
        plan=dict(question='Does the exact coherent fastest-owned implementation safely consume all262144 physical input slots?',
            unit='No native change. Standalone entrypoint plus required CACHE_SCRATCH_LEASE=0 is one inseparable diagnostic adaptation: lease1 requires serve. Other reference math/queue/chunk/384requested bank/decode flags/assets stay fixed. Length is the only variable between short/full cases;full has0output room.',
            order='Host parser refusal controls and same-build native bounds fixture, then one16385-input/4-output standalone case. Commit normal/quiet/fault-free closure before one262144-input/0-output consume-prompt case. No retries.',
            acceptance='Original2361source/dependency recovery,137objects/1470compiler/29link inputs and12ELF verification; current committed helpers, recipe, model/provider identities and boot/driver. Exact BF16 row/byte and max-allocation ledger; actual ordered-exp BF16 prefix, native16 producer/consumer, standalone release, n-1batched rows and exact checked final-slot marker. Normal0/reaped/empty owner, quiet validation and no fresh faults. Host tails/legacy512bank allowed and recorded, never called full GPU residency.',
            stopping='Stop first input/owner/budget/runtime/allocation/extent/validation/fault failure. Never reset/unbind or restart a failed arm. Full model hashes again after full case; existing checked hashes reused only with unchanged identities. No speed-based decision or inference adoption.',
            limits='Physical fill is a required narrow gate, not full-context state/numerical/RESTORE qualification or throughput evidence. bench.sh remains the performance entrypoint. Native14 has no old streamed-QSA/prefix-budget telemetry; do not reuse6e40parser assumptions.',
            retention='Keep compact allocation/extents/owner/fault outcomes and original source/fixture generator identities. Delete successful raw traces/JIT when cases close. Existing12coherentELFs and independent decode reference remain; no new generation.'),
        budgets=dict(cpu=18000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=256<<20))
    ROOT.mkdir(mode=0o700)
    for name,n,outputs,wall in CASES:
        root=ROOT/name;root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
        path=root/'tokens.txt';raw=token_bytes(n);path.write_bytes(raw)
        case=dict(name=name,tokens=n,outputs=outputs,wall=wall,root=str(root),
            fixture=dict(path=str(path),bytes=len(raw),sha256=sha(path)))
        case['command']=command(row,case);case['environment_sha256']=signature(environment(row,case));row['cases'].append(case)
    save(row)
    print('Prepared coherent short/full standalone diagnostic;commit plan before host or GPU run.',flush=True)


def current(row):
    require(validation.committed(RESULT,(KEY,))==compact(row)
        and core.pins(row['source_commit'])==row['sources']
        and all(sha(P/p)==h for p,h in row['policies'].items()), 'commit/freeze exact diagnostic inputs')
    native,proof,prior=basis(row['original_native_proof'])
    require(row['reference_evidence_sha256']==signature(compact(prior))
        and row['binary']==native['binaries']['strata_full_context_diagnostic']
        and row['host_binary']==native['binaries']['full_context_diagnostic_host_test']
        and row['native_source_commit']==original.SOURCE
        and all(row[k]==prior[k] for k in ('configuration','environment_template','model_assets',
            'native_validation_probe','boot','driver','runtime_manifest'))
        and len(row['cases'])==2, 'original native/config/model proof changed')
    for case,(name,n,outputs,wall) in zip(row['cases'],CASES):
        path=Path(case['fixture']['path']);raw=token_bytes(n)
        require((case['name'],case['tokens'],case['outputs'],case['wall'])==(name,n,outputs,wall)
            and path.read_bytes()==raw and sha(path)==case['fixture']['sha256']
            and len(raw)==case['fixture']['bytes'] and case['command']==command(row,case)
            and signature(environment(row,case))==case['environment_sha256'], 'generated workload/configuration changed')


def run(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row);before=copy.deepcopy(row)
    require(not row['active'] and row['ownership_closed'] and mode not in row['outcomes'], 'no retry or duplicate owner')
    require((mode=='host' and row['stage']=='prepared') or
        (mode=='short' and row['stage']=='ready' and row['host_passed']) or
        (mode=='full' and row['stage']=='short_closed' and row['outcomes']['short']['passed']), 'ordered gate not passed')
    root=ROOT/mode
    if mode=='host':root.mkdir()
    require(not (root/'execution.json').exists(), 'existing process receipt; inspect original owner')
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in text.splitlines() if s.startswith('-- cursor: '))
    outcome=dict(passed=False,complete=False,host_probe_accepted=False)
    owner=None
    def persist():
        row.update(active=bool(owner and owner.active),ownership_closed=bool(owner and owner.active is None
            and all(owner_module.closed(c) for c in owner.commands)))
        core.write(ROOT/'record.json',row)
        core.write(root/'execution.json',dict(outcome,active=row['active'],commands=owner.commands if owner else []))
    owner=owner_module.Owner(root/'owned',persist);row['stage']=mode+'_running';persist()
    try:
        if mode=='host':
            env=dict(row['environment_template']);env.pop('LD_PRELOAD',None)
            jobs=[('reader',[reference.model.PYTHON,'-B','-m','unittest','tools.test_coherent_context_model'],60),
                  ('native',[row['host_binary']['path']],30)]
            for label,argv,wall in jobs:
                entry,out,err=owner.run(label,argv,env,P,wall=wall,cpu=40,rss_cap=2<<30,
                    text_cap=1<<20,file_cap=1<<20,total_cap=8<<20)
                require(owner_module.closed(entry) and entry['exit_code']==0, 'host diagnostic control failed')
                if label=='native':require('full_context_request_and_terminal_bounds=PASS' in out.read_text()
                    and 'byte_order_and_bounds=PASS' in out.read_text(), 'same-build physical host gate absent')
            outcome.update(reader_controls=True,same_build_host_bounds=True,GPU_executed=False)
        else:
            case=next(c for c in row['cases'] if c['name']==mode)
            outcome.update(native_execution_attempted=True,GPU_execution_possible=True)
            entry,out,err=owner.run(mode,case['command'],environment(row,case),P,wall=case['wall'],**row['budgets'])
            outcome.update(native_exit_code=entry.get('exit_code'),stdout_sha256=sha(out),stderr_sha256=sha(err))
            require(owner_module.closed(entry) and entry['exit_code']==0, 'native capacity exit/ownership failed')
            outcome.update(evidence(out.read_text(),err.read_text(),row,case,entry['pid']),GPU_executed=True)
        current(before)
        if mode=='full':comparison.shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed']=bool(outcome['passed'] and outcome['complete'] and outcome.get('driver_after')==row['driver'])
        row['outcomes'][mode]=outcome
        row.update(stage=('ready' if mode=='host' else mode+'_closed') if outcome['passed'] else 'failed',
            passed=bool(mode=='full' and outcome['passed']),complete=bool(mode=='full' and outcome['passed']),
            host_passed=bool(row['host_passed'] or mode=='host' and outcome['passed']))
        persist();save(row)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','native_exit_code','ownership_closed','fresh_faults','error')}),flush=True)
    require(outcome['passed'], 'coherent capacity gate failed; preserve outcome before new plan')


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['host'],['short'],['full']), 'prepare|host|short|full')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        comparison.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run(sys.argv[1])
