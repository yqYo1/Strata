"""Recover the owned coherent engine and its closed speed reference.

Main-authored, repository LICENSE. Kept separately from retired one-run physical
capacity controllers; callers must qualify their own source, route and outcome.
"""
import json
from pathlib import Path

import coherent_original_reuse as original
import coherent_prefill_bench as reference
import compare_official_prefill as comparison
import qsa16_stream_bench as core
import ur_validation as validation

P, B, RESULT = reference.P, reference.B, reference.RESULT
POLICIES = ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md')
MARKER = ('strata FULL_CONTEXT consumed_position=262144 prompt_tokens=262144 '
          'output_tokens=0 window_width=1 commit_count=1 commit_wait=PASS refill_drain=PASS')
require, sha, signature = original.require, original.sha, original.fingerprint
compact = core.compact


def basis(expected=None):
    native, proof = original.verify_original(expected=expected)
    prior = json.loads((reference.ROOT/'record.json').read_text())
    require(validation.committed(RESULT, (reference.KEY,)) == compact(prior)
        and prior['stage'] == 'closed' and not prior['active']
        and all(prior[k] for k in ('passed','complete','ownership_closed'))
        and prior['binary'] == native['binaries']['strata']
        and prior['native_source_commit'] == original.SOURCE, 'closed coherent speed reference required')
    manifest = prior['runtime_manifest']; files = json.loads(Path(manifest['path']).read_text())
    require(sha(manifest['path']) == manifest['sha256'] and len(files) == manifest['files']
        and all(sha(p) == h for p,h in files.items())
        and comparison.build.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'original providers, boot or driver changed')
    comparison.shared.models.model_pins(prior['configuration'], prior['model_assets'], full=False)
    return native, proof, prior
