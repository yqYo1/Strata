"""Build the checked MoE Gate-boundary observer on the measured V2 source."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_build as upload

bank, builder, require = upload.bank, upload.builder, upload.require
KEY = 'prefill_moe_gate_clock_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CLOCK = 'bench/sycl/prefill_moe_gate_clock.inc'
MEASURE = 'bench/sycl/prefill_moe_gate_clock_measure.py'
UNIT, WINDOW = upload.upload.UNIT, upload.upload.WINDOW
INCLUDE = 'src/prefill/prefill_moe_gate_clock.inc'


def replace(source, before, after):
    before, after = before.encode(), after.encode()
    require(source.count(before) == 1, 'the measured MoE clock anchor changed')
    return source.replace(before, after)


def clock_projection(source, clock):
    unit = source[UNIT]
    signature = '''bool native_proj16(Gemm& gm, const core::WeightRef* w, const uint16_t* x, uint16_t* y,
                   int64_t rows, const std::string& name, std::string& err) {'''
    unit = replace(unit, signature, '#include "prefill_moe_gate_clock.inc"\n' +
                   signature.replace('std::string& err)', 'std::string& err, PrefillMoeGateClock* clock = nullptr)'))
    dequant = '''        strata::kernels::iq_dequant_f16(w->native_type,
            static_cast<const uint8_t*>(w->native_data) + size_t(r) * row_bytes,
            count * k, gm.scratch(), gm.stream());'''
    unit = replace(unit, dequant, '''        if (clock && r == 0) clock->mark(PrefillMoeGateClock::dequant_begin);
''' + dequant + '''
        if (clock && r == 0) clock->mark(PrefillMoeGateClock::dequant_end);''')

    window = source[WINDOW]
    window = replace(window, '''    owner.status_observed = ~uint32_t(0);
    {
        hp::StatusScope status_profile;''', '''    PrefillMoeGateClock gate_clock;
    owner.status_observed = ~uint32_t(0);
    {
        hp::StatusScope status_profile;''')
    window = replace(window, '''            if (cudaMemcpyAsync(&owner.status_observed, status, sizeof(uint32_t), cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
                cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = "prefill MoE accumulator: status readback/drain"; return false;
            }''', '''            gate_clock.mark(PrefillMoeGateClock::copy_begin);
            const auto status_copy = cudaMemcpyAsync(&owner.status_observed, status, sizeof(uint32_t),
                                                     cudaMemcpyDeviceToHost, m.cs);
            gate_clock.mark(PrefillMoeGateClock::wait_begin);
            if (status_copy != cudaSuccess || cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = "prefill MoE accumulator: status readback/drain"; return false;
            }
            gate_clock.mark(PrefillMoeGateClock::wait_end);''')
    window = replace(window, '''        auto* input = mixed_piece(at, count);
        if (!native_proj16(m.gemm, wsg, input, sgate, count, v.name("ffn_gate_shexp.weight"), err) ||''',
        '''        auto* input = mixed_piece(at, count);
        auto* clock = at == 0 && gate_clock.enabled ? &gate_clock : nullptr;
        if (clock) clock->mark(PrefillMoeGateClock::projection_begin);
        if (!native_proj16(m.gemm, wsg, input, sgate, count, v.name("ffn_gate_shexp.weight"), err, clock) ||''')
    window = replace(window, '''    if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "prefill MoE accumulator: HC writes drain"; return false; }
    if (upload_census''',
        '''    if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "prefill MoE accumulator: HC writes drain"; return false; }
    gate_clock.emit(layer, first_row, rows, std::min(m.T, rows));
    if (upload_census''')
    return {UNIT: unit, WINDOW: window, INCLUDE: clock}


def retained_v2_projection(expected=None):
    native, foundation, source, recipe, proof = upload.project(expected)
    parent = bank.previous.read(upload.ROOT / 'record.json')
    source_hashes = bank.stage.hashed(source)
    require(parent['stage'] == 'built' and parent['complete'] and parent['passed']
            and parent['ownership_closed'] and not parent['active']
            and parent['projection'] == proof and parent['recipe'] == recipe
            and parent['source_files'] == source_hashes and parent['foundation'] == foundation
            and parent['external_inputs'] == native['external_inputs']
            and parent['driver'] == builder.guards.driver_guard()
            and builder.compact(parent) == bank.basis.validation.committed(bank.RESULT, (upload.KEY,))
            and bank.sha(parent['binaries']['strata']['path']) == parent['binaries']['strata']['sha256'],
            'recover the exact V2 native source and preserve the measured engine')
    return native, foundation, source, recipe, proof, parent


def recipe_at(recipe, root):
    old, new = str(upload.ROOT), str(root)
    return dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                environment={key: val.replace(old, new) for key, val in recipe['environment'].items()},
                targets=recipe['targets'])


def project(revision, expected=None):
    native, foundation, source, recipe, proof, parent = retained_v2_projection(expected)
    clock = dict(bank.stage.source.blobs(bank.P, revision, [CLOCK]))[CLOCK]
    require(INCLUDE not in source, 'the diagnostic include already exists in the V2 projection')
    overlay = clock_projection(source, clock)
    source.update(overlay)
    proof = dict(proof, clock_source_commit=revision,
                 clock_parent_build_sha256=bank.signature(builder.compact(parent)),
                 overlays=proof['overlays'] | bank.stage.hashed(overlay))
    return native, foundation, source, recipe_at(recipe, ROOT), proof


def context(revision):
    ctx = upload.context()
    ctx.ROOT, ctx.KEY = ROOT, KEY
    ctx.project = lambda expected=None: project(revision, expected)
    ctx.scope = 'V2 status-copy, checked wait, status-to-Gate handoff and first dequant host intervals.'
    ctx.retention = 'Preserve the V2 prefill and coherent decode references through the observation bracket.'
    return ctx


def prepare(ctx, revision, controls):
    root = ctx.ROOT
    require(not root.exists() and ctx.KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing prefill build')
    require(not (upload.GATE_ROOT / 'build').exists(), 'retire the completed Gate engine')
    native, foundation, source, recipe, proof = ctx.project()
    parent = bank.previous.read(upload.ROOT / 'record.json')
    controls = {*parent['controller_sources'], *controls}
    pins = bank.stage.source.pin(bank.P, sorted(controls))
    require(pins['revision'] == revision and controls == pins['source_sha256'].keys(),
            'commit the observation controls and generating source before preparation')
    row = dict(stage='prepared', source_commit=revision, native_source_commit=revision,
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               foundation=foundation, projection=proof, recipe=recipe, external_inputs=native['external_inputs'],
               build_tools={name: bank.sha(name) for name in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=parent['driver'], retained_decode_reference=upload.upload.decode_reference(recover=True),
               active=False, passed=False, complete=False, ownership_closed=True, commands=[],
               scope=ctx.scope, retention=ctx.retention)
    root.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = root / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ctx.check(row, committed=False)
    ctx.save(row)


def run_stage(ctx, revision, action, controls):
    builder.ROOT, builder.KEY, builder.check, builder.save = ctx.ROOT, ctx.KEY, ctx.check, ctx.save
    if action == 'projection':
        _, _, sources, recipe, proof = ctx.project()
        print(dict(source_files=len(sources), overlays=proof['overlays'], recipe_sha256=bank.signature(recipe)))
    elif action == 'prepare':
        prepare(ctx, revision, controls)
    elif action == 'host':
        bank.stage.host(root_path=ctx.ROOT, check_build=ctx.check, save_record=ctx.save)
    else:
        row = bank.previous.read(ctx.ROOT / 'record.json')
        builder.TARGETS = row['recipe']['targets']
        builder.execute(action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('projection', 'prepare', 'configure', 'compile', 'host'), 'select one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        row = bank.previous.read(ROOT / 'record.json') if action in ('configure', 'compile', 'host') else None
        revision = row['source_commit'] if row else bank.basis.validation.git('rev-parse', 'HEAD').decode().strip()
        run_stage(context(revision), revision, action, (THIS, CLOCK, MEASURE))
