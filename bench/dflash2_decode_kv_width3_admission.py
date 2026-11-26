"""Probe width-three DFlash2 KV dispatch with a recovered benchmark prompt."""

import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_kv_admission as kv
import dflash2_decode_kv_width3_native as build
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer

bank, basis, primary, ranked = kv.bank, kv.basis, kv.primary, kv.ranked
require, validation = kv.require, kv.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_kv_width3_admission_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = kv.ORDER
CAPTURE = 'b28a9372bc1943812a87d170402435cc1c7425f0'
CAPTURE_KEY = 'dflash2_long_routing_holdout_v1'
WORKLOAD = 'varied-prose'
READ, OUTPUT, CHUNK = 33287, 48, 8192
PLAN = dict(order=list(ORDER), read_tokens=READ, output_cap=OUTPUT,
            scope='Same coherent binary and stock global expert placement. Switch only width-three KV at exec.',
            input='Recover the official thinking prompt byte-for-byte from the original calibration. '
                  'Send its rendered tokens through GEN with a bounded output cap.',
            checks='Require fresh layer-major prefill, width-three and scalar fallback captures, '
                   'quiet validation, closed ownership and matched token IDs and draft counters.',
            limits='Route and parity only. Sustained bench.sh timing, 262144 fill and RESTORE are separate.')


class PromptRecovered(Exception):
    def __init__(self, body):
        super().__init__('recovered benchmark request')
        self.body = body


def prompt(tokenizer):
    captured = bank.previous.historic(CAPTURE, CAPTURE_KEY)
    process, = captured['processes']
    require(build.grid.retained.closed(captured, 'closed') and process['ownership_closed']
            and not process['fresh_faults'] and process['native_exit_code'] == 0,
            'recover the closed varied-prose capture')
    samples = process['evidence']['samples']
    points = {s['run']: s['prompt_tokens'] for s in samples if s['phase'] == 'calibration'}
    observed = captured['routing_workload']
    require(set(points) == {200, 800} and points[800] > points[200] > 0
            and samples[-1]['phase'] == f'long-generation/{WORKLOAD}'
            and observed['read_tokens'] == samples[-1]['read_tokens'] == READ
            and observed['request_sha256'] == samples[-1]['request_sha256'],
            'unexpected original calibration or prompt identity')

    client = primary.speed.generation.suite.official.d.client

    class RecoverySuite(client.Suite):
        def request(self, phase, run, turn, messages, max_tokens, think=False, **kwargs):
            require((phase, run, turn, think) == (f'long-generation/{WORKLOAD}', 0, None, True),
                    'unexpected request from the benchmark generator')
            raise PromptRecovered(client.chat_body(messages, max_tokens, think, **kwargs))

    suite = RecoverySuite('', '', 262144, 1, True, log=lambda _: None, want_long_generation=True,
                          long_generation_only=True, generation_workload=WORKLOAD)
    suite.tokens_per_word = (points[800] - points[200]) / 600
    suite.overhead = max(0, points[200] - 200 * suite.tokens_per_word)
    try:
        suite.run_long_generation()
    except PromptRecovered as recovered:
        body = recovered.body
    else:
        raise ValueError('benchmark generator returned without a request')
    digest = sha256(json.dumps(body).encode()).hexdigest()
    require(digest == observed['request_sha256'], 'recovered request differs from the official body')

    template = ChatTemplate(tokenizer / 'chat_template.jinja')
    text = template.render(body['messages'], reasoning_effort=body['reasoning_effort'],
                           **body['chat_template_kwargs'])
    vocab = bank.previous.read(tokenizer / 'vocab.json')
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    cfg = bank.previous.read(tokenizer / 'tokenizer.json')
    merges = (tokenizer / 'merges.txt').read_text(encoding='utf-8').split('\n')
    tok = Tokenizer(tokens, merges, bank.previous.read(tokenizer / 'token_type.json'),
                    cfg['pre'], cfg['special_ids'])
    ids = tok.encode(text, parse_special=True)
    require(len(ids) == READ and text.endswith('<think>\n') and tok.decode(ids) == text,
            'thinking prompt bytes or actual input extent changed')
    proof = dict(archived_request_sha256=digest, native_wire_sha256=sha256(kv.wire(ids, OUTPUT)).hexdigest(),
                 rendered_sha256=sha256(text.encode()).hexdigest())
    return ids, proof


def routes(text, *, cooperative):
    expected = f'strata prefill policy: requested=throughput route=layer-major tokens={READ - 1} chunk={CHUNK}'
    require(text.splitlines().count(expected) == 1, 'require the actual fresh layer-major prefill route')
    captured = kv.kv_routes(text, cooperative=cooperative, widths=(3,))
    widths = {(r['phase'], r['width']) for r in captured}
    require({('prompt', 1), ('decode', 3)} <= widths
            and any(phase == 'decode' and width != 3 for phase, width in widths),
            'require a finalized prompt, one width-three capture and a scalar decode fallback')
    return captured


def inputs(revision):
    reference = build.grid.retained.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    require(build.grid.retained.closed(native, 'built') and native.get('policy', {}).get('passed')
            and native['policy']['ownership_closed'] and native['policy']['binary'] == native['binaries'][build.POLICY],
            'finish the coherent build and its same-build policy check')
    stock = ranked.restore_proof(reference, ranked.restore.KEY, ranked.restore.ORDER)
    profile = ranked.qualified_profile('stock', stock, None)
    source = ranked.experiment('blend').pins(revision)
    for name in (THIS, kv.THIS, kv.scorer.THIS, build.THIS, 'serve/frontend.py', 'tools/strata_tokenizer.py',
                 'tools/strata_bench.py', 'tools/long_generation_bench.py', 'tools/ab_engine.py'):
        path = bank.P / name
        require(path.read_bytes() == validation.git('show', revision + ':' + name),
                'commit the exact width-three admission input: ' + name)
        source[name] = bank.sha(path)
    cfg = ranked.configuration(dict(reference, qualified_profiles={'stock': profile}),
                               dict(arm='stock', root=str(ROOT)))
    args = cfg['args']
    for flag, value in (('--kv', 'int8'), ('--kv-resident', '20480'), ('--max-context', '262144'),
                        ('--prefill', str(CHUNK)), ('--spec', '4')):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == value, 'unexpected model option: ' + flag)
    require('--no-spec-split' in args, 'keep the verifier unsplit')
    ids, proof = prompt(Path(cfg['tokenizer']))
    binary = native['binaries']['strata']
    device = primary.speed.generation.model
    cases, environments = [], {}
    for index, arm in enumerate(ORDER):
        root = ROOT / arm
        env = kv.native_environment(reference, primary.curve.environment(reference, dict(root=str(root), index=index)))
        require(env.get('STRATA_SYCL_KV_COPY_GROUPS') == '0' and kv.FLAG not in env and kv.scorer.FLAG not in env,
                'unexpected inherited KV or QSA selector')
        env.update({kv.FLAG: '3' if arm == 'cooperative' else '0', kv.scorer.FLAG: '0', 'STRATA_TRACE': '1',
                    'LD_DEBUG': 'libs', 'UR_ENABLE_LAYERS': 'UR_LAYER_PARAMETER_VALIDATION',
                    'UR_LOG_VALIDATION': f'level:warning;flush:warning;output:file,{root / "validation.log"}'})
        env.pop('UR_LOG_TRACING', None)
        environments[arm] = env
        cases.append(dict(arm=arm, root=str(root), command=[binary['path'], '--serve', *args],
                          environment_sha256=bank.signature(env), limits=device.LIMITS))
    return dict(source_commit=revision, source_inputs=dict(files=len(source), sha256=bank.signature(source)),
                native_source_commit=native['native_source_commit'], build_sha256=bank.signature(build.builder.compact(native)),
                binary=binary, boot=reference['boot'], driver=reference['driver'], stock=profile,
                runtime_manifest=reference['runtime_manifest'], plan=PLAN, cases=cases, prompt=proof,
                read_tokens=READ, output_cap=OUTPUT), environments, (ids, OUTPUT), reference


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    record = bank.previous.read(bank.RESULT)
    record[KEY] = row
    bank.stage.write(bank.RESULT, record)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned interpreter and select one width-three admission stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one new admission cohort')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            row, _, _, _ = inputs(revision)
            for case in row['cases']:
                for name in ('tmp', 'cache'):
                    (Path(case['root']) / name).mkdir(mode=0o700, parents=True)
            save(dict(row, stage='prepared', outcomes={}, complete=False, passed=False, ownership_closed=True, active=False))
            return
        row = bank.previous.read(ROOT / 'record.json')
        pinned, environments, request, reference = inputs(row['source_commit'])
        require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in pinned.items())
                and row['ownership_closed'] and not row['active'] and row['stage'] in ('prepared', 'running')
                and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next admission case')
        kv.execute(row, environments[action], request, reference, action, read_routes=routes, publish=save)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
