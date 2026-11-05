#!/usr/bin/env python3
"""Fresh host and small-device gates for the scoped centering engine.

Main-authored, repository LICENSE. Reuse the maintained exact/pool and traced
phase runners in private module contexts. Never reuse a retired ELF or inherit
the previous model's qualification. Main serializes every native operation.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import build_q6_centering_model as model
import iq3_host_v2 as host_template
import check_verify_window_phase as device_template

P, B, RESULT, base = model.P, model.B, model.RESULT, model.base
SELF = Path(__file__).resolve()
HOST_ROOT, HOST_KEY = B / 'q6-centering-model-host-v1', 'q6_centering_model_host_v1'
DEVICE_ROOT = B / 'q6-centering-model-phase-device-v1'
DEVICE_KEY = 'q6_centering_model_phase_device_v1'
require = model.require


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def built():
    native = model.context().verify()
    require(native['centering_native_recipe_checks']['passed']
            and native['centering_native_recipe_checks']['q6_and_full_verifier_compiled']
            and native['native_sources']['include/strata/kernels/q6_packed_center.hpp']
                == '2a6df6a2fdc1e8e8ad17cd6eeda6e2b9cb57be62e2ab2ee7cb1e63d1b702ff75',
            'fresh complete centering build and actual helper input required')
    return native


def host_context():
    ctx = load('strata_centering_host_checks', host_template.__file__)
    ctx.ROOT, ctx.KEY, ctx.SELF = HOST_ROOT, HOST_KEY, SELF
    ctx.build = SimpleNamespace(ROOT=model.ROOT,
        component=model.source.original.component)
    ctx.proof = built
    prepare = ctx.prepare

    def prepare_host():
        prepare()
        row = json.loads((HOST_ROOT / 'record.json').read_text())
        row['plan'] = ('Fresh same-build exact0,pool0,exact1,pool1 children for the '
            'Q6-centering model generation. Require384single/80pair bitwise GGML/canary '
            'controls,80production calls,decode/prefill routing and28real-pool cases '
            'with the original checksum. Immutable IQ3 selectors before exec;common '
            'AUDIT1,SINGLE_EXACT0,CPU_SERVICE0. Isolated1MiB output budget. Stop first '
            'failure;no retry,rebuild,GPU/model execution or speed qualification.')
        row['qualification_basis'] = dict(build_key=model.KEY,
            host_runner='tools/sycl/iq3_host_v2.py',
            CPU_sources='Unchanged563-file baseline CPU inputs;only7Q6/phase source files differ.',
            policy_gate='q6_centering_policy_host_v1 at dc2299;not full-verifier GPU math.')
        ctx.save(row)

    ctx.prepare = prepare_host
    return ctx


def qualified_host(native):
    ctx = host_context()
    checked = json.loads((HOST_ROOT / 'record.json').read_text())
    ctx.unchanged(checked)
    require(checked['stage'] == 'closed' and checked['passed'] and checked['complete']
            and checked['ownership_closed'] and not checked['active']
            and not checked['GPU_executed'] and not checked['native_rebuilt']
            and checked['binaries'] == native['binaries']
            and checked['native_source_commit'] == native['source_commit']
            and [(c['kind'], c['arm'], c['passed'], c['exit_code'])
                 for c in checked['results']] ==
                    [('exact', 0, True, 0), ('pool', 0, True, 0),
                     ('exact', 1, True, 0), ('pool', 1, True, 0)]
            and checked['pool_checksum'] ==
                model.source.original.component()['correctness']['pool_checksum'],
            'closed same-build exact arithmetic/phase/pool gate required')


def device_context():
    ctx = load('strata_centering_phase_checks', device_template.__file__)
    native_context = model.context()
    ctx.build = SimpleNamespace(ROOT=model.ROOT, KEY=model.KEY,
        compact=native_context.compact, inputs=model.inputs,
        selected=model.template.selected)
    ctx.ROOT, ctx.KEY, ctx.SELF = DEVICE_ROOT, DEVICE_KEY, SELF
    prior_built = ctx.built

    def qualified_build():
        native = prior_built()
        require(native['centering_native_recipe_checks']['passed'],
                'full native centering recipe checks required')
        qualified_host(native)
        return native

    ctx.built = qualified_build
    return ctx


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] in ('host', 'device')
            and sys.argv[2] in ('prepare', 'run'),
            'usage: q6_centering_checks.py host|device prepare|run')
    ctx = host_context() if sys.argv[1] == 'host' else device_context()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': ctx.prepare, 'run': ctx.run}[sys.argv[2]]()
