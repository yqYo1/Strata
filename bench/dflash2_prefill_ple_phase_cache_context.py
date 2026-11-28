"""Check the standalone final slot with the dedicated prefill PLE cache."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_phase_cache as phase
import dflash2_prefill_ple_early_capacity as earlier

bank, require = phase.bank, phase.require
basis = earlier.basis
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_phase_cache_context_v1'
ROOT = bank.B / KEY.replace('_', '-')
PLAN = dict(order=list(earlier.PLAN['order']),
    scope='Use the coherent phase-cache build on the original standalone short and full-context token fixtures.',
    gates='Require same-build host bounds, the original 16385-input handoff and all 262144 input slots. '
          'Verify the actual 4194304-row prefill cache and unchanged 1048576-row ordinary cache in both model cases. '
          'Require the original copy-writer placement, per-layer KV evidence and expected output counts.',
    limits='Check physical context with the selected cache. Live RESTORE, agent workloads and decode speed '
           'have separate gates.')


def prerequisites(_reference_context):
    baseline = phase.baseline()
    native = bank.previous.read(phase.BUILD_ROOT / 'record.json')
    measured = bank.previous.read(phase.ROOT / 'record.json')
    sealed = basis.validation.committed(bank.RESULT, (phase.KEY,))
    require(phase.wait.retained.closed(measured, 'closed')
            and basis.compact(measured) == sealed
            and measured['decision'] == phase.capacity.decision(measured['processes'])
            and measured['decision']['warm_cache_mechanism_advances']
            and all(measured[k] == baseline[k] for k in (
                'binary', 'native_source_commit', 'native_build_sha256', 'host_fixture_sha256',
                'capacity_proof_sha256', 'boot', 'driver', 'runtime_manifest',
                'native_validation_probe', 'dflash2_assets')),
            'finish the exact same-build dedicated-cache comparison first')
    require(native['binaries']['strata'] == measured['binary']
            and all(name in native['binaries'] for name in
                    ('strata_full_context_diagnostic', 'full_context_diagnostic_host_test')),
            'use the physical-context executables from that coherent build')
    return native, baseline, bank.signature(sealed)


def experiment():
    standalone = basis.core.source.device.load_module('dedicated_ple_context', earlier.__file__)
    standalone.ROOT, standalone.KEY, standalone.PLAN = ROOT, KEY, PLAN
    standalone.prerequisites = prerequisites
    ctx = standalone.experiment()
    base_environment, base_evidence, base_pins = ctx.environment, ctx.evidence, ctx.source_pins

    def environment(row, case):
        env = base_environment(row, case)
        require(phase.SELECTOR not in env
                and env.get('STRATA_PREFILL_PLE_RECORD_AUDIT', '0') == '0',
                'unexpected PLE cache or record observer at standalone startup')
        return dict(env, **{phase.SELECTOR: str(phase.capacity.LARGE_ROWS)})

    def evidence(stdout, stderr, row, case, pid):
        result = base_evidence(stdout, stderr, row, case, pid)
        expected = f'{phase.START}long_prefill={phase.capacity.LARGE_ROWS} ordinary={phase.capacity.DEFAULT_ROWS}'
        require([line for line in stderr.splitlines() if line.startswith(phase.START)] == [expected]
                and phase.TICKET not in stderr, 'standalone prefill cache startup differs')
        phase.check_first_cache(stderr, 'large', 1)
        return dict(result, phase_ple_cache=dict(prefill_rows=phase.capacity.LARGE_ROWS,
                                                 ordinary_rows=phase.capacity.DEFAULT_ROWS,
                                                 observed_request_routes=1))

    def pins(revision):
        sources = base_pins(revision)
        for name in (THIS, phase.THIS, phase.capacity.THIS, phase.corpus.THIS, phase.geometry.THIS):
            source_path = bank.P / name
            require(source_path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact dedicated-cache context input: ' + name)
            sources[name] = bank.sha(source_path)
        return sources

    ctx.environment, ctx.evidence, ctx.source_pins = environment, evidence, pins
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON
            and action in ('prepare', 'host', *earlier.physical.ORDER), 'select a committed capacity stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        control = experiment()
        control.prepare() if action == 'prepare' else control.run(action)
