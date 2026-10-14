#!/usr/bin/env python3
"""One owned long-prompt observation; supplementary diagnostic, never stock speed."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent
RECOVERY = P / 'bench/results/2026-10-14-primary-compact-moe-recovery'
D = P / 'bench/results/2026-10-14-primary-model-baseline'
U = Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012')
BUILD = B / 'main-primary-routed-fp16-census-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
PHASES = ['embed+steps', 'hc read', 'gdn', 'qsa proj', 'qsa indexer', 'qsa select',
          'qsa attn', 'router+shared', 'host grouping', 'gather', 'wait copy', 'dequant',
          'gemm gate/up', 'gemm down', 'combine', 'ple', 'kv stage', 'gdn conv+gates',
          'gdn recurrence', 'gdn out proj']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_input(root):
    # Same tokenizer/template implementation as the server, with a fixed public
    # word-list recipe from the unchanged stock client. This is a new diagnostic
    # input, not byte-identical to the earlier timestamp-tagged stock requests.
    sys.path.insert(0, str(P / 'tools'))
    sys.path.insert(0, str(P))
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    spec = importlib.util.spec_from_file_location('stock_ab_input', U / 'tools/ab_engine.py')
    stock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stock)
    tpath = PACK / 'tokenizer'
    vocab = json.loads((tpath / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (tpath / 'merges.txt').read_text().split('\n'),
                    json.loads((tpath / 'token_type.json').read_text()))
    tpl = ChatTemplate(tpath / 'chat_template.jinja')
    words = 32768
    for _ in range(4):
        body = dict(messages=[dict(role='user', content='[strata route census v1]\n' + stock.long_prompt(words))],
                    max_tokens=1, temperature=0, reasoning_effort='none')
        rendered = tpl.render(body['messages'], enable_thinking=False)
        ids = tok.encode(rendered, parse_special=True)
        if 32769 <= len(ids) <= 36865:
            break
        words = max(1, int(words * 33792 / len(ids)))
    assert 32769 <= len(ids) <= 36865 and all(0 <= x < len(vocab) for x in ids)
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    path = root / 'tokens.txt'
    path.write_text(','.join(map(str, ids)) + '\n')
    return path, dict(actual_input_tokens=len(ids), batched_input_tokens=len(ids)-1,
        token_ids_sha256=sha(path), canonical_body_sha256=hashlib.sha256(canonical).hexdigest(),
        rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(), words=words,
        recipe='fixed public stock long_prompt word list, server template, thinking disabled',
        source_pins={str(f):sha(f) for f in [U/'tools/ab_engine.py', P/'tools/strata_tokenizer.py',
            P/'serve/frontend.py', tpath/'vocab.json', tpath/'merges.txt',
            tpath/'token_type.json', tpath/'chat_template.jinja']})


def main():
    mode = sys.argv[1]
    assert mode in ['census', 'phase', 'reference']
    O = B / ('main-primary-long-routed-' + mode + '-root-v2')
    build = json.loads((BUILD / 'record.json').read_text())
    gate = json.loads((B/'main-primary-routed-census129-pair-root-v1/record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    assert gate['passed'] and gate['complete'] and not gate['active']
    assert gate['binary_sha256'] == build['binaries']['strata']
    assert not gate['fresh_faults'] and not gate['driver_dump_exists'] and gate['boot_unchanged']
    prior = json.loads(gzip.decompress((RECOVERY/'stock-evidence/compact1/record.json.gz').read_bytes()))
    assert prior['passed'] and not prior['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        assert (DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists()
        for name, pin in build['source_pins'].items():
            assert sha(P/name) == pin
        engine = Path(build['build_path'])/'strata'
        assert sha(engine) == build['binaries']['strata']
        O.mkdir()
        token_file = O / 'tokens.txt'
        env = dict(prior['environment'])
        # Identical runtime policy to the completed stock arm. Keep detailed API
        # tracing, parameter-validation layers, sampler and preload absent. The
        # first small debug pair already exercised the new observation path.
        for key in ['STRATA_PREFILL_TIMING', 'STRATA_TRACE', 'STRATA_DBG_NAN',
                    'UR_ENABLE_LAYERS', 'LD_PRELOAD']:
            assert key not in env
        env.update(XDG_CACHE_HOME=str(O/'cache'), STRATA_WATCHDOG_S='480',
            STRATA_MEASUREMENT_ENGINE_START=str(O/'engine-start.json'),
            STRATA_MEASUREMENT_ENGINE_EXIT=str(O/'engine-exit.json'),
            STRATA_DUMP_FIRST_LOGITS=str(O/'first-logits.bin'))
        if mode != 'reference':
            env['STRATA_PREFILL_ROUTED_FP16_OBSERVE'] = '1'
        if mode == 'phase':
            env['STRATA_PREFILL_TIMING'] = '1'
        args = list(prior['engine_args'])
        args.remove('--serve')
        args += ['--tokens-file', str(token_file), '--max-new', '1']
        argv = ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(D/'engine_wrapper_v3.py'), *args]
        record = dict(active=True, passed=False, complete=False, commands=[], mode=mode,
            environment=env, engine_args=args, input=dict(preparation_complete=False), source_pins=build['source_pins'],
            binary_sha256=sha(engine), build_receipt_sha256=sha(BUILD/'record.json'),
            source_revision=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),
            source_diff_sha256=build['source_diff_sha256'], recipe_sha256=sha(Path(__file__)),
            wrapper_sha256=sha(D/'engine_wrapper_v3.py'),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=1800, raw_text_bytes=64<<20, RSS_bytes=88<<30,
                        job_bytes=1<<30, tensor_bytes=16<<20),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            restore_qualified=False, adopted=False, GPU_reset_or_service_change=False,
            scope='Supplementary fresh CLI long-prompt diagnostic, not unchanged bench.sh performance. Existing phase barriers perturb execution; exact route/payload meanings retained.')
        def persist():
            (O/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(O/'owned',persist)
        record['commands'] = owner.commands
        persist()
        def utility(label, argv):
            entry,out,err = owner.run(label,argv,build['environment'],P,wall=30,
                text_cap=64<<20,total_cap=1<<30)
            assert module.closed(entry) and entry['exit_code']==0
            return out.read_text()
        cursor = None
        try:
            actual_path, identity = make_input(O)
            assert actual_path == token_file
            record['input'] = dict(identity, preparation_complete=True)
            persist()
            cursor = re.search(r'-- cursor: (\S+)',utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])).group(1)
            record['journal_before_cursor'] = cursor
            entry,out,err = owner.run('long-'+mode,argv,env,P,wall=1800,cpu=2000,
                text_cap=64<<20,rss_cap=88<<30,total_cap=1<<30,file_cap=64<<20)
            assert module.closed(entry) and entry['exit_code']==0, (entry.get('error'),err.read_text()[-2000:])
            native = json.loads((O/'engine-exit.json').read_text())
            assert native['exit_code']==0 and native['binary_sha256']==record['binary_sha256']
            text = err.read_text()
            assert 'Strata SYCL cleanup:' not in text
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):',text)
            import array
            floats = array.array('f'); floats.frombytes((O/'first-logits.bin').read_bytes())
            assert len(floats)==248320 and all(math.isfinite(x) for x in floats)
            lines = [x.split('strata routed fp16 observation: ',1)[1] for x in text.splitlines()
                     if x.startswith('strata routed fp16 observation: ')]
            if mode=='reference':
                assert not lines
            else:
                assert len(lines)==1
                ro = json.loads(lines[0])
                assert ro['effective'] and ro['input_tokens']==identity['batched_input_tokens']
                assert ro['configured_chunk']==4096 and ro['compact_fp16']==1
                assert ro['layer_chunks']==48*ro['chunks']
                assert sum(ne*c for ne,c in ro['hist_ne_count'])==ro['gu_rows']==ro['down_rows']==48*10*identity['batched_input_tokens']
                assert sum(c for ne,c in ro['hist_ne_count'])==512*ro['layer_chunks']
                assert ro['gu_calls']==ro['down_calls']==ro['active_experts']
                assert ro['blob_h2d_attempted_host_copy_calls']==ro['blob_h2d_attempted_direct_pinned_copy_calls']+ro['blob_h2d_attempted_stager_copy_calls']
                assert ro['pf_timing_enabled']==(mode=='phase')
                if mode=='phase':
                    assert len(ro['pf_timeline_ms_by_phase'])==len(PHASES)
                    assert all(math.isfinite(x) and x>=0 for x in ro['pf_timeline_ms_by_phase'])
                    record['phase_names']=PHASES
                record['routed_observation']=ro
            record.update(native_exit=native,first_logits_sha256=sha(O/'first-logits.bin'),
                first_logits_finite=True,stdout_sha256=sha(out),stderr_sha256=sha(err))
            for name,pin in record['source_pins'].items():
                assert sha(P/name)==pin
            assert sha(engine)==record['binary_sha256']
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            try:
                if cursor:
                    entries=[json.loads(x) for x in utility('journal-after',
                        ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if x.strip()]
                    record['fresh_faults']=[x for x in entries if FAULT.search(str(x.get('MESSAGE','')))]
                    record['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False,complete=False)
            finally:
                record['active']=False
                persist()
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True,mode=mode,input=identity,
            routed_observation=record.get('routed_observation'),fresh_faults=0)),flush=True)


if __name__=='__main__':
    # Tokenizer/Jinja dependencies are already installed in the model's venv.
    # Re-exec before creating an output directory; no new packages are installed.
    interpreter = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
    if sys.executable != interpreter:
        os.execv(interpreter, [interpreter, str(Path(__file__).resolve()), *sys.argv[1:]])
    main()
