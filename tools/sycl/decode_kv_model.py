#!/usr/bin/env python3
"""Actual long/short/long handoff for the decode-only KV candidate.

Main-authored using the owned coherent model controller; repository LICENSE.
No speed, task-quality or physical full-context claim follows from this check.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import check_decode_kv as device
import coherent_prefill_model as previous

P, B, RESULT = device.P, device.B, device.RESULT
ROOT, KEY = B / 'decode-kv-model-v1', 'decode_kv_model_v1'
THIS = 'tools/sycl/decode_kv_model.py'
require = device.require


def kv_evidence(text):
    footers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = ', text, re.M))
    require([int(m[1]) for m in footers] == [32769, 82, 32769], 'exact native handoff inputs required')
    begin, requests = 0, []
    for index, footer in enumerate(footers):
        rows = []
        for line in text[begin:footer.start()].splitlines():
            if not line.startswith('strata verify KV_PLAN:'):
                continue
            m = re.fullmatch(r'strata verify KV_PLAN: phase=(prompt|decode) tokens=([1-9]\d*) '
                r'enabled=1 scalar_calls=(\d+) coalesced_calls=(\d+)', line)
            require(m is not None, 'actual enabled KV capture marker required')
            phase, tokens, scalar, cooperative = m.groups()
            tokens, scalar, cooperative = int(tokens), int(scalar), int(cooperative)
            require(not cooperative or phase == 'decode' and tokens in (2, 4),
                'cooperative planner escaped measured decode widths')
            rows.append(dict(phase=phase, tokens=tokens, scalar_calls=scalar,
                coalesced_calls=cooperative))
        if index in (0, 2):
            require(any(r['phase'] == 'decode' and r['tokens'] == 4 and r['coalesced_calls'] > 0
                for r in rows), 'each long handoff must actually capture cooperative decode4')
        requests.append(dict(input_tokens=int(footer[1]), captures=rows,
            observed_cooperative_widths=sorted({r['tokens'] for r in rows if r['coalesced_calls']})))
        begin = footer.end()
    require(not any(s.startswith('strata verify KV_PLAN:') for s in text[begin:].splitlines()),
        'unbound trailing KV capture')
    return dict(requests=requests, prefill_cooperative_calls=0,
        actual_decode4_handoffs=2, scope='Actual capture dispatch and state handoff,not model speed or default adoption.')


def controls():
    prompt = 'strata verify KV_PLAN: phase=prompt tokens=4 enabled=1 scalar_calls=12 coalesced_calls=0\n'
    single = 'strata verify KV_PLAN: phase=decode tokens=1 enabled=1 scalar_calls=12 coalesced_calls=0\n'
    decode = 'strata verify KV_PLAN: phase=decode tokens=4 enabled=1 scalar_calls=0 coalesced_calls=12\n'
    footer = lambda n: f'strata serve: prompt {n} tokens = fixture\n'
    text = prompt + single + decode + footer(32769) + footer(82) + prompt + decode + footer(32769)
    kv_evidence(text)
    kv_evidence(text.replace(decode, decode + decode.replace('tokens=4', 'tokens=2')))
    bad = [text.replace('enabled=1', 'enabled=0'), text.replace(decode, '', 1),
        text.replace(prompt, prompt.replace('coalesced_calls=0', 'coalesced_calls=1'), 1),
        text.replace(single, single.replace('coalesced_calls=0', 'coalesced_calls=1')),
        text.replace(decode, decode.replace('tokens=4', 'tokens=8')),
        text.replace('scalar_calls=0', 'scalar_calls=-1'),
        text.replace('prompt 32769 tokens', 'prompt 32768 tokens', 1), text + footer(82)]
    for value in bad:
        try:
            kv_evidence(value)
        except RuntimeError:
            continue
        raise RuntimeError('invalid KV handoff marker/phase/extent accepted')
    return dict(passed=True, positive=2, negative=len(bad), GPU_executed=False)


def controller():
    spec = importlib.util.spec_from_file_location('decode_kv_actual_model', previous.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.THIS = ROOT, KEY, THIS
    module.device = SimpleNamespace(**{**vars(device), 'context': lambda: device.controller().context()})
    # Keep the latest validated model configuration, not an obsolete native ELF.
    module.reference = SimpleNamespace(**{**vars(module.reference), 'ROOT': previous.ROOT, 'KEY': previous.KEY})
    original_route = module.route
    def route(text):
        result = original_route(text)
        result['decode_KV'] = kv_evidence(text)
        return result
    module.route = route
    return module


def prepare():
    module = controller(); module.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_SYCL_DECODE_KV_COALESCED='1', STRATA_SYCL_KV_COPY_GROUPS='0')
    row.pop('previous_preparation', None); row.pop('previous_control_failure', None)
    row['plan'] = ('Commit the exact controller/configuration/assets first. Host INFO/stage controls plus '
        'two positive/eight negative literal KV-capture controls,then one fresh330405engine process with '
        'thinking-enabled32769/82/32769inputs and32outputs each. Preserve the validated14508model '
        'configuration,262144logical/20480resident INT8KV,8192chunks,1256expert slots,MTPspec4,head2/dense1,'
        'large GU and all safety settings. Candidate1/copy-groups0 are immutable before exec. Every long '
        'handoff must actually capture cooperative decode4;prompt and other widths must never select it. '
        'Require all existing typed-QSA,GU/Down,MTPrestore/cache/recapture,normal0,quiet-validation,'
        'closed-owner,no-fresh-fault and15asset before/after checks. Stop first failure,no retry or pin refresh.')
    row['limits'] = ('Actual route/state handoff only. No sustainedTG,clean PP/TTFT,task/agent,physical '
        '262144/tails/RESTORE,ordinary teardown or default adoption. Width2requires observed model '
        'evidence separately. Repair API/native-segment journal binding before long-generation speed trials.')
    module.context().context(row['native_observer_admission']).save(row)
    print('Prepared actual decode-only KV model handoff;commit before host/model execution.', flush=True)


if __name__ == '__main__':
    require(Path(sys.prefix) == Path(previous.PYTHON).parent.parent, 'use the qualified model venv')
    require(len(sys.argv) == 2 and sys.argv[1] in previous.ACTIONS, '|'.join(previous.ACTIONS))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); device.common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1] == 'host':
            result = controls(); module = controller(); module.context().execute('host')
            row = json.loads((ROOT / 'record.json').read_text()); row['KV_host_controls'] = result
            module.context().context(row['native_observer_admission']).save(row)
        else:
            row = json.loads((ROOT / 'record.json').read_text())
            require(row['KV_host_controls']['passed'], 'literal KV handoff controls required')
            controller().context().execute('run')
