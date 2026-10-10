"""Root-only CPU preparation, preserving the original controller/build receipt."""
from pathlib import Path
import copy, datetime, fcntl, hashlib, json, runpy, subprocess

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/eval-b570-20261010')
SOURCE = B / 'xestrata-owned-comparison-v1/run_first_diagnostic.py'
BUILD_RECEIPT = B / 'xestrata-pristine-icpx-build-root-v1/record.json'
OUT = B / 'xestrata-diagnostic-contract-root-v3'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def ident(p):
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists()
    assert sha(SOURCE) == '199806ce1e12e3181b1a6a0e7b478ed08a655f52ff60efcf65e505334e045a6c'
    r = json.loads(BUILD_RECEIPT.read_text())
    assert r['passed'] and r['complete'] and not r['active'] and r['fork_pristine']
    assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True) == ''
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == r['source_commit']
    assert len(r['commands']) == 2
    for c in r['commands']:
        assert c['exit_code'] == 0 and c['observation_complete'] and c['session_empty'] and c['direct_child_reaped']
        assert not c.get('error') and not c['errors'] and not c['cleanup'] and not c['survivors']
    for key in ['binary_identity', 'device_binary_identity', 'compile_commands', 'cache_identity', 'build_json']:
        v = r[key]
        assert ident(Path(v['path'])) == v
    for v in r['source'].values():
        assert ident(Path(v['path'])) == v
    cc = json.loads(Path(r['compile_commands']['path']).read_text())
    assert len(cc) > 50
    for name in ['src/program/generate.cpp', 'src/prefill/prefill.cpp', 'src/prefill/gemm.cpp', 'src/kernels/xe/xmx_gemm.cpp']:
        rows = [x for x in cc if Path(x['file']).resolve() == W / name]
        assert len(rows) == 1
        assert '/opt/intel/oneapi/compiler/2026.1/bin/icpx' in rows[0]['command']
    compiler = json.loads(Path(r['build_json']['path']).read_text())['compiler']['version']
    assert compiler == '2026.1.1'
    text = SOURCE.read_text()
    assert text.count('int(v[2]) == 32767') == 1
    text = text.replace('int(v[2]) == 32767', 'int(v[2]) == 32768')
    assert text.count("build.get('compiler_version') == '2026.1'") == 1
    text = text.replace("build.get('compiler_version') == '2026.1'", "build.get('compiler_version') == '2026.1.1'")
    malformed = next(v for v in text.splitlines() if "record['new_fault_messages'] =" in v)
    replacement = "                record['new_fault_messages'] = [v['MESSAGE'] for v in rows if (((('0000:05:00.0' in v.get('MESSAGE', '') or re.search(r'\\bxe\\b', v.get('MESSAGE', ''))) and health.FAULT.search(v.get('MESSAGE', ''))) or ('strata' in v.get('MESSAGE', '') and 'segfault' in v.get('MESSAGE', ''))))]"
    text = text.replace(malformed, replacement)
    compile(text, 'root-reviewed-controller', 'exec')
    OUT.mkdir(mode=0o700)
    controller = OUT / 'run_first_diagnostic_v3.py'
    controller.write_text(text)
    source_sha = {name: v['sha256'] for name, v in r['source'].items()}
    source_sha['src/kernels/xe/xmx_gemm.cpp'] = sha(W / 'src/kernels/xe/xmx_gemm.cpp')
    adapter = dict(passed=True, active=False, completed=True, source_commit=r['source_commit'],
                   fork_sources_unchanged=True, license_mode=r['mode'], compiler_version=compiler,
                   ggml_commit=r['ggml_commit'], root=str(W), binary=r['binary'],
                   binary_sha256=r['binary_identity']['sha256'], source_sha256=source_sha,
                   original_build_receipt=ident(BUILD_RECEIPT), compile_commands=r['compile_commands'],
                   transformation='schema adapter; validated closed command ownership, source, binary and compiler identities; original unchanged')
    receipt = OUT / 'build-receipt-adapter.json'
    receipt.write_text(json.dumps(adapter, indent=2) + '\n')
    m = runpy.run_path(str(controller))
    root, binary, argv, tokens, baseline, profile = m['preflight'](receipt)
    valid = copy.deepcopy(baseline['requests'][0])
    assert m['validate'](valid)['fresh_complete_finite']
    tests = [{'case': 'canonical32768-with-32767-batched-position', 'passed': True}]
    cases = [
        ('wrong-PP-total', 'PP', 2, '32767'),
        ('wrong-last-PP-position', 'PP', 1, '32768'),
        ('wrong-DONE-prompt', 'DONE', 2, '32767'),
        ('wrong-DONE-read-n', 'DONE', 14, '32767'),
        ('reused-DONE', 'DONE', 8, '1'),
        ('nonfinite-prompt-time', 'DONE', 3, 'nan'),
        ('zero-decode-time', 'DONE', 4, '0'),
        ('negative-drafts', 'DONE', 6, '-1'),
        ('bad-finish', 'DONE', 5, 'cancelled'),
        ('resumed-input', 'RESUME', 1, '1'),
        ('reused-input', 'REUSED', 1, '1'),
    ]
    for label, tag, index, value in cases:
        v = copy.deepcopy(valid)
        pos = [i for i, line in enumerate(v['protocol']) if line.startswith(tag + ' ')][-1]
        fields = v['protocol'][pos].split(); fields[index] = value
        v['protocol'][pos] = ' '.join(fields)
        try: m['validate'](v)
        except (ValueError, IndexError): tests.append(dict(case=label, passed=True))
        else: raise AssertionError('accepted ' + label)
    for label in ['missing-DONE', 'duplicate-DONE', 'missing-PP', 'duplicate-RESUME', 'bad-token-id', 'missing-LP', 'nonfinite-LP', 'bad-LP-top-id']:
        v = copy.deepcopy(valid)
        if label == 'missing-DONE': v['protocol'] = [x for x in v['protocol'] if not x.startswith('DONE ')]
        elif label == 'duplicate-DONE': v['protocol'].append(next(x for x in v['protocol'] if x.startswith('DONE ')))
        elif label == 'missing-PP': v['protocol'] = [x for x in v['protocol'] if not x.startswith('PP ')]
        elif label == 'duplicate-RESUME': v['protocol'].append('RESUME 0')
        elif label == 'bad-token-id': v['ids'][0] = 248320
        elif label == 'missing-LP': v['logprobs'].pop()
        elif label == 'nonfinite-LP':
            fields = v['logprobs'][0].split(); fields[1] = 'nan'; v['logprobs'][0] = ' '.join(fields)
        elif label == 'bad-LP-top-id':
            fields = v['logprobs'][0].split(); fields[2] = '248320:-1'; v['logprobs'][0] = ' '.join(fields)
        try: m['validate'](v)
        except (ValueError, IndexError): tests.append(dict(case=label, passed=True))
        else: raise AssertionError('accepted ' + label)
    record = dict(passed=True, active=False, completed=True, gpu_executed=False, model_executed=False,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  controller=ident(controller), parent_controller=ident(SOURCE), build_adapter=ident(receipt),
                  original_build_receipt=ident(BUILD_RECEIPT), tests=tests, argv=argv, prompt_tokens=len(tokens),
                  profile=ident(profile), source_unchanged=True,
                  correction='PP total counts all32768 input tokens; last batched position32767 plus final prompt token; full compiler version2026.1.1; missing kernel-fault-filter closing parenthesis repaired without changing intended filter',
                  future_performance_criterion=dict(actual_input_tokens=65536, minimum_comparison_input=32768,
                                                   prefill_goal_tps=1000, decode_goal_tps=70,
                                                   equal_input_per_arm=True, fresh_no_reuse=True,
                                                   fixed_costs_separately_reported=True, physical_context_qualification=262144))
    (OUT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(passed=True, gpu_executed=False, cases=len(tests), adapter=str(receipt), controller=str(controller))))
