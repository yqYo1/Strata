"""Main: commit closed six-case parity evidence and reviewed research, not speed claims."""
import base64
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
C = B / 'xe-compact-moe-model-controller-source-v2'
R = W / 'bench/results/2026-10-12-xe-compact-moe-gpu-parity'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('parity_publish_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0,str(C))
import admission as a
ENV = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
REPORTS = [
    'research-xe-eligible-prefill-compute-memory-microbenchmark-design-round146-v1.txt',
    'research-official-v0142-host-usm-stager-repair-contract-round146-v1.txt',
    'research-xe-current-decode-native-work-memory-ceiling-contract-round147-v1.txt',
    'research-xe-layer-major-fullcontext-live-ram-capacity-contract-round147-v1.txt',
    'research-xe-decode-work-service-join-source-design-round148-v1.txt',
    'research-xe-compact-on8192-64k-quality-capacity-contract-round148-v1.txt',
]

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out = B/'xe-compact-moe-gpu-parity-publish-root-v2'
    m.require(not out.exists() and R.is_dir() and not any(R.iterdir()),'new publication after original closed failure left only empty archive directory')
    out.mkdir()
    r = dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),
        recipe_sha256=m.sha(__file__),owner_sha256=m.sha(O),gpu_executed=False,
        model_executed=False,original_results_modified=False,performance_eligible=False)

    def save():
        (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')

    owner = m.Owner(out/'commands',save)
    r['commands'] = owner.commands

    def run(label,argv,cwd=W):
        e,so,_=owner.run(label,argv,ENV,cwd,wall=60,cpu=120,text_cap=8<<20,
                        total_cap=32<<20,rss_cap=2<<30,file_cap=64<<20)
        m.require(a.strong_closed_command(e) and e['exit_code']==0,'normal closed '+label)
        return so.read_text().strip()

    def write(name,value):
        target=R/name; target.parent.mkdir(parents=True,exist_ok=True)
        m.require(not target.exists(),'new compact archive')
        target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

    def copy(source,name):
        target=R/name; target.parent.mkdir(parents=True,exist_ok=True)
        m.require(not target.exists(),'new exact archive')
        target.write_bytes(source.read_bytes())
        m.require(m.sha(source)==m.sha(target),'byte-identical archive')
        return dict(source=str(source),archive=str(target.relative_to(W)),bytes=target.stat().st_size,sha256=m.sha(source))

    save()
    try:
        m.require(not run('initial-status',['/usr/bin/git','status','--porcelain']),'clean report worktree')
        identity=run('identity',['/usr/bin/git','config','--show-origin','--get-regexp',r'^user\.(name|email)$'])
        m.require(identity and all(v.startswith('file:/home/yayoi/.config/git/config') for v in identity.splitlines()),'global identity')
        m.require(run('origin',['/usr/bin/git','remote','get-url','--push','origin'])=='git@github.com:yqYo1/Strata.git','own fork only')
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        qroot=B/'xe-compact-moe-main-admission-converter-root-v1'
        conversion=json.loads((qroot/'record.json').read_text())
        a.require_closed_prerequisite(conversion)
        q=json.loads((qroot/'qualification.resolved.json').read_text())
        a.require_qualification(q,q['production_source_pins'])
        m.require(m.sha(qroot/'qualification.resolved.json')==conversion['qualification']['sha256'],'actual Main admission identity')
        records={}
        for n,variant in a.SEQUENCE:
            path=a.root_path(B,n,variant)/'record.json'
            value=json.loads(path.read_text()); a.require_normal_model(value,boot)
            m.require(value['qualification_sha256']==conversion['qualification']['sha256'] and
                value['source_pins']==q['production_source_pins'] and value['published_revision']=='39e395a103af36323262cdc0dbaea27cba81f3e4' and
                value['source_guard_phases']==['admission','ready','done','final'] and
                value['validation']['generated_tokens']==16 and value['performance_eligible'] is False and
                not value['new_fault_messages'] and value['health-before_protocol'][:6]==value['health-after_protocol'][:6],
                'exact closed bounded parity scope')
            for command in value['commands']+value['guard_commands']:
                m.require(not any('TOKEN' in key or 'PASSWORD' in key or 'SECRET' in key for key in command['environment']),
                    'explicit environment contains no credential keys')
            records[n,variant]=(path,value)
        comparisons=[]
        for n in (64,2049,32768):
            off,on=records[n,'off'][1],records[n,'on'][1]
            a.require_exact_variant_env(off['environment'],on['environment'])
            pair=a.compare_strict_outputs(off,on)
            historical=json.loads((B/f'xe-native-dequant-prefill-model-root-v6/n{n}-g16-off/record.json').read_text())
            legacy=a.compare_strict_outputs(historical,off)
            comparisons.append(dict(input_tokens=n,generated_tokens=16,actual_batched_positions=n-1,
                actual_chunk_limit=6144,fresh_off_vs_historical=legacy,compact_on_vs_fresh_off=pair,
                ready_free_vram_mib={arm:int(records[n,arm][1]['effective_settings']['vram_free_mib']) for arm in ('off','on')},
                normal_quit_both=True,kernel_faults=[],performance_eligible=False))
        m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],S)=='39e395a103af36323262cdc0dbaea27cba81f3e4' and
                  not run('source-status',['/usr/bin/git','status','--porcelain'],S),'published source unchanged')
        m.require(not any(R.iterdir()),'empty uncommitted archive from failed publisher')
        artifacts=[]
        original_path=B/'xe-compact-moe-gpu-parity-publish-root-v1/record.json'
        original=json.loads(original_path.read_text())
        m.require(original['passed'] is False and original['complete'] is False and not original['active'] and
            original['error']=="RuntimeError('normal closed secret-scan-n64-off')" and
            all(a.strong_closed_command(c) for c in original['commands']), 'original archive failure closed and retained')
        plain=out/'plaintext-receipts'; plain.mkdir()
        for (n,arm),(path,value) in records.items():
            target=plain/f'n{n}-g16-{arm}.json'
            target.write_bytes(path.read_bytes())
            m.require(m.sha(target)==m.sha(path),'scan exact plaintext destined for compressed archive')
        scan=out/'gitleaks-exact-plaintext-receipts.json'
        run('secret-scan-exact-plaintext',[str(S.parent/'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks'),
            'dir',str(plain),'--no-banner','--redact','--report-format','json','--report-path',str(scan)])
        m.require(json.loads(scan.read_text())==[],'no credentials in exact plaintext archived receipts')
        artifacts.append(copy(original_path,'receipts/original-gpu-parity-publication-failed-v1.json'))
        artifacts.append(copy(B/'xe-compact-moe-gpu-parity-publish-root-v1/gitleaks-n64-off.json','receipts/original-secret-scan-redacted.json'))
        artifacts.append(copy(B/'xe-compact-moe-gpu-parity-publish-v1.py','recipes/xe-compact-moe-gpu-parity-publish-v1.py'))
        write('publication-scan-scope-correction.json',dict(original_passed=False,original_result_modified=False,
            original_receipt_sha256=m.sha(original_path),original_scan_scope='Entire private model run including unrelated whole-boot kernel capture',
            reason='The two redacted generic-api-key detections were Loaded X.509 cert Build time autogenerated kernel key public certificate identifiers, in unrelated kernel-before raw capture lines913/935. Those raw boot entries are not archived.',
            corrected_scope='Actual six plaintext JSON receipts byte-identical to the compressed archive, plus the complete final archive. No rule exclusions or skip of archived plaintext.',
            raw_kernel_capture_modified=False,model_outcomes_modified=False))
        for (n,arm),(path,value) in records.items():
            raw=path.read_bytes(); packed=gzip.compress(raw,mtime=0)
            name=f'receipts/n{n}-g16-{arm}.json.lossless.json'
            write(name,dict(encoding='gzip+base64',source=str(path),decoded_bytes=len(raw),decoded_sha256=m.sha(path),
                original_passed=value['passed'],data_base64=base64.b64encode(packed).decode('ascii')))
            decoded=gzip.decompress(base64.b64decode(json.loads((R/name).read_text())['data_base64'],validate=True))
            m.require(decoded==raw,'lossless exact original receipt validation')
            artifacts.append(dict(source=str(path),archive=str((R/name).relative_to(W)),decoded_bytes=len(raw),
                                  decoded_sha256=m.sha(path),archive_sha256=m.sha(R/name)))
        for name in ('record.json','qualification.resolved.json','publication.bridge.resolved.json'):
            artifacts.append(copy(qroot/name,'admission/'+name))
        for path in [B/'xe-compact-moe-main-admission-converter-v1.py',Path(__file__)]:
            artifacts.append(copy(path,'recipes/'+path.name))
        # Keep the original report-publisher failure and its actual separate successful correction.
        artifacts.append(copy(B/'xe-compact-moe-results-representation-publish-root-v2/record.json','receipts/source-qualification-publication-correction.json'))
        reports=[]
        for name in REPORTS:
            item=copy(B/'research-20261009'/name,'research/'+name)
            reports.append(dict(path=item['source'],archive=item['archive'],sha256=item['sha256'],model='gpt-6-luna',main_full_report_review=True,implemented=False))
        prior=W/'bench/results/2026-10-11-xe-compact-moe-scratch-source-qualification/report-registry-v138.json'
        m.require(json.loads(prior.read_text())['research_completed']==580,'actual previous committed registry')
        write('report-registry-v139.json',dict(research_completed=580+len(reports),prior_registry_path=str(prior.relative_to(W)),
            prior_registry_sha256=m.sha(prior),new_reports=reports,scope='Read-only research full Main review; no implementation/measurement count added.'))
        arena=47962*(1<<20); kv=262144*25344
        write('main-research-corrections-v2.json',dict(original_reports_modified=False,source_only=True,
            R146_microbenchmark=dict(source_hashes_independently_matched=True,
                provenance='Hashes match current committed source; assignment did not itself list these hashes.',
                current_dequant_reference='Current nativebatch0 uses individual dequant wrappers. Grouped APIs are a separate conditional benchmark arm; do not label them current production execution.',
                event_timing='Kernel-event duration excludes separate transfers. A copy-event duration measures that command interval, not necessarily the complete host-pack/PCIe service.'),
            R147_memory=dict(actual_memtotal=134986948608,actual_representative_available=97499279360,
                expert_arena_bytes=arena,configured_context=262144,host_int8_kv_bytes=kv,
                residual_plane64k_bytes=65535*4*2560*4,residual_plane_full_bytes=262143*4*2560*4,
                current_config_known_owner_subtotal64k_bytes=arena+kv+65535*4*2560*4,
                current_config_known_owner_subtotal_full_bytes=arena+kv+262143*4*2560*4,
                correction='R147 50.885GiB subtotal assumes context65536 and shrinks the hostKV. Current context262144 allocates full6.1875GiB even for64K prompt; corrected known subtotal is55.525GiB. No64GB-host fit claim applies to this125.715GiB host.',
                full_context_fit_qualified=False,other_session_snapshot_runtime_owners_unmeasured=True),
            R148_route_scope=dict(
                explicit_unsupported_compact_request='Prefill::init returns an error before allocation when explicit compact opt-in is incompatible; it does not silently turn the request off.',
                short_prompt_clipping='Prompt-length clipping belongs to the known-prompt one-shot initialization route. Serve allocates fixed chunk storage before GEN; short requests use tails without changing the6144 chunk limit.',
                ring='Owned stage buffers clamp to16; fullstream uses16 and routed consumers use the requested8. Do not label every consumer16.',
                quality_permission='The report suggestion to request a numeric threshold creates no approval requirement. Existing authorization covers measurement; evaluate observed deltas and retain unresolved quality claims.',
                changed_chunk_quality_and_physical_fit_unqualified=True)))
        write('results.json',dict(source_revision='39e395a103af36323262cdc0dbaea27cba81f3e4',source_pins=q['production_source_pins'],
            comparisons=comparisons,artifacts=artifacts,arithmetic_precision_changed=False,default_compact_on=False,
            physical8192_fit_qualified=False,full262144_qualified=False,restore_qualified=False,
            broad_quality_qualified=False,performance_eligible=False,
            next_step='Resolve8192 compact-ON capacity and quality diagnostic, then64K quiet prefill and independent decode evaluation. Preserve full262144 occupancy check.'))
        (R/'README.md').write_text('''# Compact MoE GPU output and lifetime checks

At actual chunk6144, fresh OFF/ON checks at64,2049 and32768 input tokens all closed
normally with16 generated IDs and exact selected LP5. Fresh OFF also matched the
same-N retained historical OFF. Every process passed fresh integer GPU health,
runtime mapping/source guards and kernel fault checks. Arithmetic precision did
not change. These are diagnostics; their timings do not qualify performance.

The same configuration observed745MiB free VRAM OFF and1286MiB ON at READY.
This is a6144 observation, not8192 fit. The feature remains defaultOFF;8192,
64K performance, broad model quality, RESTORE and full262144 occupancy are unqualified.

[Individual results](results.json) link each original receipt. Receipt archives
use gzip+base64 with verified decoded byte count and SHA256. Source qualification
and exact controller source are in the preceding committed source report. Retain
private originals for current model/reference consumers; no cleanup is performed here.
[Research corrections](main-research-corrections-v2.json) preserve original reports.

本家の測定は箱出し設定だけではないが、本家単体の最適設定探索・独立検証は
未完了。[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を維持する。
''')
        scan=out/'gitleaks-report.json'
        run('secret-scan-archive',[str(S.parent/'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks'),
            'dir',str(R),'--no-banner','--redact','--report-format','json','--report-path',str(scan)])
        m.require(json.loads(scan.read_text())==[],'no credentials archived')
        run('stage',['/usr/bin/git','add','--',str(R.relative_to(W))])
        run('staged-check',['/usr/bin/git','diff','--cached','--check'])
        run('commit',['/usr/bin/git','commit','-m','docs(sycl): record compact MoE GPU parity and tuning prerequisites'])
        r['committed_revision']=run('head',['/usr/bin/git','rev-parse','HEAD'])
        run('push-own-fork',['/usr/bin/git','push','origin','HEAD'])
        m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean published boundary')
        r['complete']=True
    except BaseException as error:
        r['error']=repr(error)
    finally:
        r['active']=owner.active is not None
        r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc']=m.utc(); save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
