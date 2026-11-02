#!/usr/bin/env python3
"""Current-engine clean/observed117K attribution; repository LICENSE.

Reuse owned supervision and resource readers in private policy contexts. An I/O
proxy checks actual routes and observers before each continuation. No imported
shared context, native source or process environment is mutated. Main owns tests.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import cache_map_prefill_speed as speed
import prefill_resources as maintained

entry, base, P, B, RESULT = speed.entry, speed.base, speed.P, speed.B, speed.RESULT
ROOT = entry.model.ROOT/'prefill-resources117489-v1'
KEY = 'cache_map_prefill_resources117489_v1'
SOURCE = 'tools/sycl/cache_map_prefill_resources.py'
require, digest = speed.require, speed.digest


class GateIO:
    """Observe the existing protocol without consuming or replacing its input."""
    def __init__(self, io, root, plan, check, checked):
        self.io, self.root, self.plan, self.check, self.checked = io, root, plan, check, checked
        self.gate_pending = None
        self.offset = None

    def __getattr__(self, name):
        return getattr(self.io, name)

    def line(self, deadline):
        line = self.io.line(deadline)
        if line.startswith('STRATA_PROFILE_GATE '):
            require(self.gate_pending is None, 'unacknowledged gate')
            self.gate_pending = json.loads(line.removeprefix('STRATA_PROFILE_GATE '))
        return line

    def send(self, data):
        require(data == b'continue\n' and self.gate_pending is not None, 'unexpected gate continuation')
        stage, request = ('begin', len(self.checked)) if self.offset is None else ('end', len(self.checked))
        require(self.gate_pending == dict(stage=stage, phase='prefill-only', run=request), 'unplanned gate order')
        path = self.root/'engine.stderr'
        end = path.stat().st_size
        if stage == 'begin':
            self.offset = end
        else:
            require(end > self.offset, 'missing request interval')
            part = path.read_bytes()[self.offset:end].decode()
            # This runs before forwarding continue to the original controller.
            self.checked.append(self.check(part, self.plan))
            self.offset = None
        self.gate_pending = None
        return self.io.send(data)


def qualification(revision):
    cohort = json.loads((speed.ROOT/'record.json').read_text())
    require(base.committed(RESULT)[speed.KEY] == speed.compact(cohort) and cohort['stage'] == 'closed'
            and cohort['passed'] and cohort['complete'] and cohort['host_passed']
            and cohort['ownership_closed'] and not cohort['active'] and len(cohort['processes']) == 3
            and cohort['decision'] == speed.decision(cohort['processes'])
            and cohort['decision']['all_process_medians_at_least1000'],
            'close the frozen three-process clean117K milestone first')
    native, providers, manifest, _ = speed.proof(cohort['source_commit'])
    require(cohort['binary'] == native['binaries']['strata'] and
            cohort['native_source_commit'] == native['source_commit'] and
            cohort['python_selected'] == manifest['selected'], 'same qualified native/frontend required')
    inv = cohort['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers
            and len(providers) == inv['files'], 'clean-cohort providers changed')
    recovered = {n:hashlib.sha256(raw).hexdigest() for n,raw in
                 speed.shared.gen.blobs(P,cohort['source_commit'],cohort['sources'])}
    current = speed.pinned_sources(revision)
    require(recovered == cohort['sources'] and all(current.get(n) == h for n,h in recovered.items())
            and set(current)-set(recovered) == {SOURCE}, 'only this resource adapter may extend clean inputs')
    return cohort, providers


def context():
    spec = importlib.util.spec_from_file_location('strata_current_resource_policy', maintained.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.qualification = ROOT, KEY, qualification
    original_environment = ctx.environment
    ctx.environment = lambda row, plan: environment(original_environment, row, plan)
    ctx.checked = {}

    def check(text, plan):
        route = speed.request_route(text, dict(plan, kernel='candidate'))
        if plan['observed']:
            ctx.observed(text, route)
        else:
            require(not any(v in text for v in ('strata trace:', 'strata lm profile:',
                'strata prefill cpu wait:', 'strata MoE progress:')), 'clean gate has observers')
        return route

    original_owner = ctx.build.supervision.Owner
    class Owner(original_owner):
        def run(self, *args, **kwargs):
            interact = kwargs.get('interaction')
            if interact is not None:
                plan = ctx.case_plan
                checked = ctx.checked.setdefault(plan['index'], [])
                require(not checked, 'fresh observed process only')
                kwargs['interaction'] = lambda io, command: interact(
                    GateIO(io, Path(plan['root']), plan, check, checked), command)
            return super().run(*args, **kwargs)
    supervision = SimpleNamespace(**vars(ctx.build.supervision)); supervision.Owner = Owner
    ctx.build = SimpleNamespace(**vars(ctx.build)); ctx.build.supervision = supervision
    original_finish = ctx.finish_process
    def finish(outcome, owner, cursor, boot):
        try:
            ctx.comparison.require_quiet_validation(Path(ctx.case_plan['root']))
            if outcome.get('complete'):
                require(len(ctx.checked.get(ctx.case_plan['index'], [])) == ctx.RUNS,
                        'missing pre-continuation checks')
                for sample in outcome['samples']: ctx.comparison.require_valid_client_ttft(sample)
        except BaseException as error:
            outcome.update(complete=False, error=repr(error))
        original_finish(outcome, owner, cursor, boot)
        if outcome.get('driver_after') != ctx.expected_driver:
            outcome.update(passed=False, error=outcome.get('error','post-run driver differs'))
    ctx.finish_process = finish
    original_save = ctx.save
    def save(row):
        if row.get('processes'):
            p = row['processes'][-1]
            checked = ctx.checked.get(p['index'], [])
            p['request_routes_checked_before_continue'] = len(checked)
            p['actual_native16_GDN3_routes_checked'] = all(v['GDN_barriers'] == 3 and
                v['GDN_FP16_storage'] and v['split_guard_checked'] for v in checked)
        original_save(row)
    ctx.save = save
    return ctx


def environment(original, row, plan):
    env = original(row, plan)
    require(all(env.get(k) == v for k,v in dict(STRATA_SYCL_ADAPT_SYNC='1',
        STRATA_MTP_BATCH_F16='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
        STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='0').items()),
        'preserve the current typed route and avoid the unsupported old chunk census')
    return env


def prepare(ctx):
    revision = speed.shared.gen.git('rev-parse','HEAD').decode().strip()
    cohort, providers = qualification(revision)
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh resource pair only')
    controls = dict(resource=maintained.selfcheck(), continuation=selfcheck())
    assets = ctx.shared.models.model_pins(cohort['configuration'],cohort['model_assets'],full=True)
    ROOT.mkdir(mode=0o700);base.write(ROOT/'runtime-inputs.json',providers)
    row = dict(stage='prepared', source_commit=revision, sources=speed.pinned_sources(revision),
        native_source_commit=cohort['native_source_commit'], binary=cohort['binary'],
        native_validation_probe=cohort['native_validation_probe'], model_assets=assets,
        configuration=copy.deepcopy(cohort['configuration']), environment_template=dict(cohort['environment_template']),
        python_contract=cohort['python_contract'], python_selected=cohort['python_selected'], template=cohort['template'],
        boot=cohort['boot'], driver=cohort['driver'],
        runtime_manifest=dict(path=str(ROOT/'runtime-inputs.json'),sha256=digest(ROOT/'runtime-inputs.json'),files=len(providers)),
        comparison_fingerprint=ctx.comparison.fingerprint(ctx.compact(cohort)),
        request_hashes=cohort['request_hashes'][str(speed.READ)], plans=[], processes=[],
        host_passed=False,complete=False,passed=False,active=False,ownership_closed=True,
        budgets=cohort['budgets'], controls=controls,inference_adopted=False)
    row['native_qualification_reuse'] = dict(native_source_commit=cohort['native_source_commit'],
        binary=cohort['binary'], clean_result_key=speed.KEY, native_result_key=entry.model.KEY,
        original_clean_inputs_unchanged=True, new_native_build=False,
        limits='Same current558-source/1452compiler-input/nine-ELF generation;all original clean275inputs recovered. No new mathematical fixture qualification inherited from observation.')
    row['plan'] = dict(question='Which host regions and coarse GPU intervals consume the current safe117K native16 path?',
        order='Two fresh processes,clean then observed;four exact-count thinking-on requests each. First use separate;subsequent-three medians are the process units. Same body vector as the closed three-process recheck.',
        constants='Current552b engine,eight unchanged math fixtures,GDN16/MTP16,ADAPT_SYNC1,zero adaptive swaps;all remaining current clean configuration and runtime safety preserved.',
        observer_differences=maintained.OBSERVERS,
        acceptance='All source/model/provider/native/owner/clock/body/read/reuse/route/final-validation/fault gates pass. Observed requests additionally pass all48planned/retired windows and CPU-v4 accounting,540GDN chunks,QSA and transition checks. Observed/clean warmed-time ratio<=1.15 admits attribution;one pair is a cost screen,not a statistical cost or gain claim.',
        stopping='Actual native16/GDN3/bank/split routes and observed CPU/census/phase gates are checked before every continuation. Stop first failure without retry,replacement,exclusion or hash refresh. Quiet validation is checked again after owned teardown;client TTFT remains independently valid. Commit each process before the next. Full15model hashes before/after pair.',
        retention='Keep individual samples,aggregate job/host/phase conservation and decisions. Retire successful raw output at closure. One warmed-window census may remain outside Git only for an explicitly selected MoE design,with finite budget and review point. Both current builds remain only while the existing native proof still consumes the fallback.',
        limitations='Nonexclusive GPU queue spans include host gaps;they are not additive device service or a full critical path. Command profiling remains0because this engine refuses it with production GU/Down fusion. Requested bytes are not physical PCIe/VRAM traffic;hardware ceilings and saturation remain unknown. QSA per-row and nested CPU-status details are not qualified. Old resource totals and worktree CPU-pool edits are not current evidence. No speed gain,sustained TG,task/agent,physical262144/RESTORE or inference adoption.')
    for i, observed in enumerate((False,True)):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for part in ('tmp','cache'):(root/part).mkdir()
        config=copy.deepcopy(row['configuration']);config['log']=str(root/'engine.stderr')
        base.write(root/'config.json',config)
        base.write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=speed.READ,heads=1,observed=observed,root=str(root),
            generated={n:digest(root/n)for n in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-exact-tokens',str(speed.READ),'--think','--runs',str(ctx.RUNS),
                '--profile-request-gates','--workload-id','native_gdn_20261018','--fail-fast-engine-error',
                '--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=ctx.comparison.fingerprint(ctx.environment(row,plan));row['plans'].append(plan)
    ctx.save(row)
    print('Prepared current-engine clean/observed117K pair;commit before probes.',flush=True)


def selfcheck():
    # Use real protocol parsing with a temporary log and no hardware or child.
    import tempfile
    class IO:
        def __init__(self, lines): self.lines, self.sent, self.pending = iter(lines), [], bytearray()
        def line(self, deadline): return next(self.lines)
        def send(self, data): self.sent.append(data)
    def line(stage, run=0): return 'STRATA_PROFILE_GATE '+json.dumps(dict(stage=stage,phase='prefill-only',run=run))
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);path=root/'engine.stderr';path.write_text('prefix\n');checked=[]
        io=IO([line('begin'),line('end')]);gate=GateIO(io,root,{},lambda text,plan: text,checked)
        gate.line(1);gate.send(b'continue\n');path.write_text('prefix\nactual\n');gate.line(1)
        gate.send(b'continue\n');require(checked==['actual\n'] and len(io.sent)==2,'actual gate window')
        require(gate.pending is io.pending, 'protocol state shadows the transport buffer')
        rejected=0
        for wrong in (line('end'),line('begin',1)):
            target=IO([wrong]);proxy=GateIO(target,root,{},lambda *_: {},[]);proxy.line(1)
            try: proxy.send(b'continue\n')
            except RuntimeError: rejected+=1
            else: raise RuntimeError('bad gate accepted')
            require(not target.sent,'rejected gate continued')
        target=IO([line('begin'),line('end')]);proxy=GateIO(target,root,{},lambda *_: (_ for _ in ()).throw(RuntimeError('forced route failure')),[])
        proxy.line(1);proxy.send(b'continue\n');path.write_text('prefix\nactual\nmore\n');proxy.line(1)
        try:proxy.send(b'continue\n')
        except RuntimeError:rejected+=1
        else:raise RuntimeError('failed route continued')
        require(len(target.sent)==1,'failed route forwarded continuation')
    return dict(passed=True, valid_gate_pair=1, rejected=rejected, GPU_executed=False)


if __name__ == '__main__':
    args=sys.argv[1:];require(len(args)==1 and args[0]in('selfcheck','prepare','host','0','1'),
        'use selfcheck|prepare|host|0|1')
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        ctx=context()
        if args==['selfcheck']:print(json.dumps(selfcheck()),flush=True)
        elif args==['prepare']:prepare(ctx)
        elif args==['host']:ctx.host()
        else:
            row=json.loads((ROOT/'record.json').read_text())
            ctx.case_plan=row['plans'][int(args[0])];ctx.expected_driver=row['driver']
            ctx.run(int(args[0]))
