"""Main: evaluate the closed, immutable four-pair quiet32K campaign."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = B / 'xe-native-dequant-prefill-quiet-timing-controller-source-v3'
Q = B / 'xe-quiet32k-campaign-qualification-v3.json'
Q_SHA = 'a5cf2859a55fb2c6f582132f003c733fe0fc5cd2c6f1c71589187b780a3c2c12'
CONTROLLER_SHA = 'c1e636e7544fa5ef6a14f6ec294b6454171b19c7b91e241dc22c7fc02a88a5f8'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def need(value, reason):
    if not value:
        raise ValueError(reason)


def summary(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    return dict(n=len(values), values=values, mean=mean, median=statistics.median(values),
                sample_sd=sd, cv_percent=100 * sd / mean if mean else None,
                minimum=min(values), maximum=max(values))


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-quiet32k-balanced-main-analysis-root-v1'
    need(not out.exists(), 'immutable new Main analysis root')
    out.mkdir(mode=0o700)
    result = dict(active=True, complete=False, passed=False, started_utc=None,
                  recipe_sha256=sha(__file__), gpu_executed=False, model_executed=False,
                  cleanup_executed=False, original_receipts_modified=False)
    def save():
        (out / 'record.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        need(sha(Q) == Q_SHA and sha(C / 'run_root.py') == CONTROLLER_SHA,
             'qualified immutable manifest and actual qualified controller')
        q = json.loads(Q.read_text())
        identity = q['campaign_identity']
        for name, digest in identity['diagnostic_helper_pins'].items():
            need(sha(C / name) == digest, 'actual helper identity ' + name)
        sys.path.insert(0, str(C))
        spec = importlib.util.spec_from_file_location('quiet_controller', C / 'run_root.py')
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        result['started_utc'] = controller.m.utc()
        order = q['sample_order']
        order_sha = hashlib.sha256(json.dumps(order, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        need(order_sha == q['sample_order_sha256'] and len(order) == 8,
             'exact planned complete four-pair campaign')
        root = B / 'xe-native-dequant-prefill-quiet-timing-root-v3' / q['campaign']
        expected_paths = [root / f"sample-{x['sample_index']:03d}" / f"n32768-g16-{x['arm']}" / 'record.json' for x in order]
        need({str(p.parent) for p in expected_paths} == {str(p) for p in root.glob('sample-*/n*-g*-*')},
             'complete population without unplanned arms')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        samples, prior, output_reference = [], [], None
        qualified_runtime = identity['qualified_runtime_libraries']
        for ordinal, (item, path) in enumerate(zip(order, expected_paths)):
            controller.unaliased_namespace(path)
            need(path.is_file() and path.stat().st_size <= 8 << 20, 'bounded real receipt')
            r = json.loads(path.read_text())
            need(r['passed'] is True and r['complete'] is True and r['active'] is False
                 and not r.get('error') and not r.get('kernel_gate_error')
                 and r['commands'] and all(controller.closed_owned(c) and c['exit_code'] == 0 for c in r['commands']),
                 'actual anchored closed PASS ' + str(path))
            need(r['model_normal_close'] is True and r['kernel_gate_passed'] is True
                 and r['boot_unchanged'] is True and r['boot_id'] == boot
                 and not r['new_fault_messages'], 'same boot, normal QUIT and fresh no-fault evidence')
            need(r['campaign'] == q['campaign'] and r['sample_index'] == item['sample_index']
                 and r['batch'] == item['arm'] and r['sample_ordinal'] == ordinal
                 and r['pair_position'] == ordinal % 2 and r['first_arm'] == order[ordinal-ordinal%2]['arm']
                 and r['sample_order'] == order and r['sample_order_sha256'] == order_sha
                 and r['qualification_path'] == str(Q) and r['qualification_sha256'] == Q_SHA
                 and r['controller_sha256'] == CONTROLLER_SHA and r['campaign_identity'] == identity
                 and r['predecessor_receipts'] == prior, 'exact identity and immutable sequential order')
            need(r['published_head'] == q['published_head'] and r['source_pins'] == q['production_source_pins']
                 and r['binary'] == q['free_release_binary'] and r['working_diff_sha256'] == q['source_patch_sha256'],
                 'qualified source and binary')
            for key in ('input', 'model_inputs', 'argv', 'cpu_exclusion_proof', 'transfer_expectations',
                        'diagnostic_qualification', 'diagnostic_helper_pins', 'source_applied_main_review_sha256',
                        'controller_main_review_sha256', 'health_probe', 'quiet_switches_absent'):
                need(r[key] == identity[key], 'qualified configuration ' + key)
            need(r['environment'] == dict(identity['environment_off'],
                 STRATA_PREFILL_NATIVE_DEQUANT_BATCH='1' if item['arm'] == 'on' else '0')
                 and r['actual_environment_matches'] is True and r['quiet_diagnostic_absence'] is True
                 and {k:v for k,v in r['effective_settings'].items() if k != 'vram_free_mib'} == identity['effective_settings_without_vram']
                 and int(r['effective_settings']['vram_free_mib']) >= 512, 'quiet environment and actual memory reserve')
            need(controller.exact_health(r['health-before_protocol']) and controller.exact_health(r['health-after_protocol'])
                 and not r['runtime_error_lines'] and not r['startup_runtime_error_lines']
                 and not r['pre_gen_runtime_error_lines'], 'exact fresh health and no runtime error')
            need(controller.validate_runtime(r['observed_libraries'], qualified_runtime) == r['runtime_mapping_validation'],
                 'original loaded runtime identities')
            need(controller.validate_mapping_phases(r['mapping_snapshots'], qualified_runtime) == r['mapping_phase_validation'],
                 'stable complete core and compiler mapping identities')
            validated = controller.validate(r['request_protocol'], n=32768)
            need(validated == r['validation'] and validated['generated_tokens'] == 16, 'strict complete GEN16')
            diag = q['diagnostic_pair'][item['arm']]
            need(sha(diag['path']) == diag['sha256'], 'unchanged diagnostic reference')
            diagnostic = json.loads(Path(diag['path']).read_text())
            outputs = [x for x in r['request_protocol'] if x.startswith(('T ', 'LP '))]
            need(outputs == [x for x in diagnostic['request_protocol'] if x.startswith(('T ', 'LP '))]
                 and r['diagnostic_reference_comparison']['output_ids_equal'] is True
                 and r['diagnostic_reference_comparison']['selected_logprob_top5_equal'] is True,
                 'actual strict diagnostic token/logprob parity')
            output_reference = outputs if output_reference is None else output_reference
            need(outputs == output_reference, 'all eight identical tokens and selected LP5')
            if ordinal % 2:
                need(r['reference'] == prior[-1] and r['reference_comparison']['output_ids_equal'] is True
                     and r['reference_comparison']['selected_logprob_top5_equal'] is True,
                     'fresh same-index first-arm numerical join')
            else:
                need('reference' not in r, 'first arm has no reference')
            need(r['performance_eligible'] is False and r['full_model_quality_qualified'] is False
                 and r['full262144_qualified'] is False and r['adopted'] is False
                 and r['reset_executed'] is False and r['retry_count'] == 0
                 and r['warmup_executed'] is False, 'original limited qualification and no reset/retry')
            receipt = dict(path=str(path), sha256=sha(path))
            prior.append(receipt)
            sample = dict(receipt=receipt, sample_index=item['sample_index'], arm=item['arm'],
                          ordinal=ordinal, first_arm=r['first_arm'], prefill_ms=validated['prompt_ms'],
                          cold_decode_ms=validated['decode_ms'], prompt_tokens=32768, actual_batched_positions=32767,
                          generated_tokens=16, prefill_tps=32768000/validated['prompt_ms'],
                          cold_decode_tps=16000/validated['decode_ms'], ready_free_mib=int(r['effective_settings']['vram_free_mib']),
                          original_passed=True, strict16_T_LP5_equal=True, fresh_health_and_fault_gate=True,
                          model_normal_close=True, original_performance_eligible=False,
                          raw_done=r['timing_sample']['raw_done'])
            samples.append(sample)
        pairs = []
        for index in q['sample_indices']:
            pair = {s['arm']:s for s in samples if s['sample_index'] == index}
            need(set(pair) == {'off', 'on'}, 'one complete pair')
            off, on = pair['off'], pair['on']
            pairs.append(dict(sample_index=index, first_arm=off['first_arm'],
                prefill_time_on_change_percent=100*(on['prefill_ms']/off['prefill_ms']-1),
                prefill_tps_on_change_percent=100*(on['prefill_tps']/off['prefill_tps']-1),
                cold_decode_time_on_change_percent=100*(on['cold_decode_ms']/off['cold_decode_ms']-1),
                cold_decode_tps_on_change_percent=100*(on['cold_decode_tps']/off['cold_decode_tps']-1)))
        stats = {arm:{key:summary([s[key] for s in samples if s['arm'] == arm])
                      for key in ('prefill_ms', 'prefill_tps', 'cold_decode_ms', 'cold_decode_tps')}
                 for arm in ('off', 'on')}
        for entry in prior:
            need(sha(entry['path']) == entry['sha256'], 'immutable original receipt after analysis')
        need(sha(Q) == Q_SHA and sha(C / 'run_root.py') == CONTROLLER_SHA, 'immutable recipes after analysis')
        result.update(campaign=q['campaign'], qualification=dict(path=str(Q), sha256=Q_SHA),
            published_source_revision=q['published_head'], boot_id=boot, sample_order=order,
            samples=samples, paired_changes=pairs, summaries=stats,
            paired_prefill_time_change_percent=summary([p['prefill_time_on_change_percent'] for p in pairs]),
            paired_cold_decode_time_change_percent=summary([p['cold_decode_time_on_change_percent'] for p in pairs]),
            timing_comparison_qualified=True, complete_population=True,
            original_performance_eligible_flags_preserved=True, full_model_quality_qualified=False,
            full262144_qualified=False, restore_qualified=False, official_settings_search_complete=False,
            adopted=False, decision='Keep native dequant batching default OFF; evaluate prefill and decode separately. No adoption from this limited four-pair cold-process comparison.',
            limits=['Single canonical 32K prompt and GEN16; selected token/LP5 equivalence only.',
                    'Engine DONE phase times, independent cold processes, no warmup.',
                    'Mapping observations outside phase timers; CPU/I/O/cache effects not calibrated.',
                    'Balanced adjacent order reduces one order bias but does not remove variability.',
                    'Decode kernels unchanged by this prefill flag; no warm/restored sustained decode claim.',
                    'Configured physical context262144 is not a completed full-context validation.'],
            complete=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['active'] = False
        result['passed'] = bool(result['complete'] and not result.get('error'))
        if 'controller' in globals():
            result['finished_utc'] = controller.m.utc()
        save()
    print(json.dumps({k:result.get(k) for k in ('passed', 'error', 'timing_comparison_qualified', 'decision')}))
    raise SystemExit(0 if result['passed'] else 1)
