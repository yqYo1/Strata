"""Freeze failed model comparisons and healthy owned cleanup without adopting candidates."""
from pathlib import Path
import datetime
import difflib
import hashlib
import json
import sys

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
helper = (base / 'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads', 1)[0]
exec(compile(helper, str(base / 'archive_lazy_full_and_dequant.py'), 'exec'))
directory = parent / 'expanded-fp16-cache-model-20261007'
assert not directory.exists()
directory.mkdir()

def read(name):
    return json.loads((base / name / 'record.json').read_text())

def gone(item):
    identity = process_identity(item['pid'])
    assert not identity or identity['start_ticks'] != item['start_ticks']

runs = []
pairs = [
    ('owned-expanded-fp16-refresh-128-2048', 'post-expanded-fp16-128-health', 128, 24, 256),
    ('owned-expanded-fp16-no-root-refresh-128-2048', 'post-expanded-fp16-no-root-128-health', 128, 14, 256),
    ('owned-expanded-fp16-no-root-refresh-zero-2048', 'post-expanded-fp16-no-root-zero-health', 0, 3, 1280),
    ('owned-upstream-refresh-2048-recheck', 'post-upstream-refresh-2048-recheck-health', 0, 17, 256),
]
for name, health_name, capacity, layer, token in pairs:
    run = read(name)
    health = read(health_name)
    assert not run['active'] and not run['completed'] and not run['healthy']
    assert run['requests'] == [] and 'SIGABRT' in run['error']
    assert not run['cleanup']['inferior_survived'] and not run['cleanup']['gdb_survived']
    gone(run['inferior']); gone(run['debugger'])
    assert health['healthy'] and health['boot_id'] == run['boot_id']
    assert health['started_utc'] > run['finished_utc']
    for step in health['steps']:
        assert step['exit_code'] == 0 and not step['timed_out'] and not step['still_alive']
        assert not Path('/proc', str(step['pid'])).exists()
    save_folder(base / name, directory, name)
    save_folder(base / health_name, directory, health_name)
    runs.append({'run': name, 'cache_request': capacity, 'binary_sha256': run['binary_sha256'],
                 'stopped_layer_zero_based': layer, 'chunk_offset': token,
                 'output_requests_completed': 0, 'cleanup': run['cleanup'],
                 'new_fault_messages': run['new_fault_messages'],
                 'subsequent_logged_exact_word_probe_passed': health['healthy']})

short = read('owned-expanded-fp16-refresh-zero-short')
assert short['healthy'] and not short['active'] and len(short['requests']) == 4
assert all(all(v['equality'].values()) for v in short['requests'])
gone(short['inferior']); gone(short['debugger'])
save_folder(base / 'owned-expanded-fp16-refresh-zero-short', directory, 'cache-zero-short')
for name in ['expanded-fp16-cache-refresh-build', 'expanded-fp16-cache-no-root-refresh-build']:
    assert read(name)['passed'] and read(name)['production_inputs_unchanged']
    save_folder(base / name, directory, name)
query = json.loads((base / 'dequant-cooperative-limit-query/query-record.json').read_text())
assert query['passed'] and not query['active'] and not query['new_fault_messages']
assert query['stdout'].count('max_workgroups_installed_query 1152') == 2
for path in sorted((base / 'dequant-cooperative-limit-query').iterdir()):
    if path.is_file() and path.name != 'query' and path.suffix not in ('.o', '.d'):
        copy(path, directory, Path('cooperative-limit-query') / path.name)
for name in ['expanded-fp16-failure-analysis', 'expanded-fp16-no-root-failure-analysis',
             'expanded-fp16-no-root-zero-failure-analysis']:
    save_folder(base / name, directory, name)
sources = ['expanded_fp16_layer_cache.hpp', 'build_expanded_fp16_cache_refresh.py',
           'build_expanded_fp16_cache_no_root_refresh.py', 'build_dequant_cooperative_limit_query.py',
           'run_dequant_cooperative_limit_query.py', 'run_owned_expanded_fp16_refresh_zero_short.py',
           'run_owned_expanded_fp16_refresh_128_2048.py', 'run_owned_expanded_fp16_no_root_refresh_128_2048.py',
           'run_owned_expanded_fp16_no_root_refresh_zero_2048.py', 'run_owned_upstream_refresh_2048_recheck.py',
           'probe_after_expanded_fp16_128.py', 'probe_after_expanded_fp16_no_root_128.py',
           'probe_after_expanded_fp16_no_root_zero.py', 'probe_after_upstream_refresh_2048_recheck.py',
           'run_expanded_fp16_refresh_2048_matrix.py', 'archive_expanded_fp16_model_results.py']
for name in sources:
    copy(base / name, directory, Path('sources-used') / name)
for name in ['root-sync-property-source-audit.json', 'prefill-2048-chunk-label-correction.json',
             'expanded-fp16-refresh-128-2048-checkpoint-analysis.json',
             'expanded-fp16-no-root-refresh-128-2048-checkpoint-analysis.json',
             'expanded-fp16-no-root-refresh-zero-2048-checkpoint-analysis.json']:
    copy(base / name, directory, Path(name))
original = root / 'sycl/src/prefill/prefill.cpp'
candidate = base / 'expanded-fp16-cache-prepared/prefill.candidate.cpp'
diff = ''.join(difflib.unified_diff(original.read_text().splitlines(keepends=True),
                                   candidate.read_text().splitlines(keepends=True),
                                   fromfile='production/prefill.cpp', tofile='private/prefill.candidate.cpp'))
(directory / 'sources-used/prefill-cache.patch').write_text(diff)
assessment = {
    'scope': 'Private optional cache model failures, actual-kernel cooperative resource query and unchanged refreshed control recheck; not clean timing or full capacity',
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'runs': runs, 'cache_zero_context128_four_requests_passed': True,
    'cache_zero_short_release_pairs': short['release_pairs'],
    'no_manual_reset_rebind_reboot_or_service_package_change': True,
    'automated_driver_ccs_reset_logged_first_failed_run': True,
    'first_reset_ordering': 'After 60-second watchdog and near owned cleanup; not evidence that reset initiated the stall',
    'cache_adopted': False, 'dequant_property_candidate_adopted': False,
    'clean_0_128_256_matrix_executed': False, 'all262144_cell_capacity_suite_passed': False,
    'reference_original_prefill_sha256': digest(original),
    'private_prefill_sha256': digest(candidate),
    'private_header_sha256': digest(base / 'expanded_fp16_layer_cache.hpp'),
    'actual_kernel_query': {'kernel_count': 2, 'local_size': [1, 1, 32],
        'max_cooperative_groups_each': 1152, 'observed_original_gu_groups': 12800,
        'observed_original_down_groups': 6400, 'gpu_kernels_submitted_by_query': 0,
        'installed_api': 'kernel_queue_specific::max_num_work_groups',
        'observed_driver_api': 'zeKernelSuggestMaxCooperativeGroupCount',
        'online_root_specific_api_available_in_installed_headers': False},
    'main_stack_symbolization': {'library_build_id': 'a150ba51180689c9ab166533b5d6503441b9e1c1',
        'normal_launch_cache128_main_stack_return_address': '0x7fec5e0c6674',
        'library_text_virtual_address': '0xbc570', 'runtime_text_start': '0x7fec5d9f4570',
        'symbolized_elf_address': '0x78e674',
        'matching_debug_symbol': 'NEO::CommandStreamReceiver::baseWaitFunction(unsigned long volatile*, NEO::WaitParams const&, unsigned long)',
        'method': 'Reconstruct saved main-thread frame-chain return address and use matching build-ID debug ELF; automatic GDB unwind stops in VDSO',
        'gpu_completion_tag_values_captured': False},
    'interpretation': 'Unnecessary cooperative launch settings exceed the measured limit and need correction. Their removal alone does not prevent model stalls. Disabled-cache and unchanged-control failures exclude optional VRAM allocation as a sufficient explanation. Specific runtime completion failure and any additional application misuse remain unresolved.',
}
(directory / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
(directory / 'README.md').write_text('''# Optional FP16 cache model checks on 2026-10-07

The private cache is rebuilt against the updated upstream executable on Arc
B570 10 GiB / Ryzen 5600X / 128 GiB RAM, kernel 7.0.0-38-generic, NEO
26.31.39395.14 and oneAPI 2026.1.1. Production source and binary are unchanged.
All first model checks retain flushed Level Zero/UR logs, parameter validation,
original per-phase waits/progress, normal MTP and owned GDB/PTY cleanup.

With cache disabled, four context-128 requests match every ID, logprob and
all 248,320 first-head floats. Six release/restore pairs complete; the process
exits normally without a new xe fault. Active cache128 allocates 1,258,291,200
bytes in the context-4096/int8-KV/2048-input test, but the first request stops
at layer24, token offset256, after two dequant launches enter completion wait.
The 60-second serve watchdog aborts it. A xe CCS engine reset appears near
cleanup, after the watchdog; this does not show that a reset began the stall.

A combined candidate retaining the same cache and changing only the two IQ
dequant launches to regular launches stops at layer14/offset256. Disabling
the cache in that exact executable also stops, at layer3/offset1280. Finally,
the unchanged ceff updated-upstream control stops at layer17/offset256 during
the same fully logged recheck. These last three intervals contain no new xe
fault. No failed run completes an output request or restores its released
decode cache. Both owners are gone after every cleanup. Four subsequent
logged H2D/kernel/D2H probes pass three rounds of 16,384 exact words each,
without invoking a reset, rebind or reboot. Small probes do not establish
healthy long model execution. See [assessment](assessment.json) and the
individual terminal records and bounded stderr tails.

The saved main-thread frame chain from the regular-launch cache128 run
contains a return address in NEO CommandStreamReceiver::baseWaitFunction,
resolved with the exact matching build-ID debug ELF. The VDSO instruction
and registers alone do not identify GPU completion tag values or why the
wait never completed. Optional cache capacity is not a sufficient explanation:
the zero-cache candidate and unchanged control also fail. No general hang
prevention or cache parity/speed result follows from these failures.

The [actual IQ-object query](cooperative-limit-query/query-record.json)
submits no GPU kernels. Both original dequant kernels report 1,152 maximum
cooperative work-groups for local size32; the original stalled GU/down
launches use 12,800/6,400 groups with UR_KERNEL_LAUNCH_FLAG_COOPERATIVE.
The installed query maps to zeKernelSuggestMaxCooperativeGroupCount. Its
API differs from the current online root-group-specific query; the first
failed compile and corrected build are preserved. The
[Level Zero programming guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#cooperative-kernels)
requires cooperative group counts to respect the queried maximum. These
dequant bodies use no root-group synchronization, so the setting needs
correction, although its removal alone did not prevent the observed waits.
The [source audit](root-sync-property-source-audit.json) finds 317 property
occurrences in 47 sycl/src files and no textual root-group helper call.
This is not a complete call-graph or resource audit, and does not justify
blanket removal from persistent or atomic synchronization kernels.

The cache owner, bounds, producer/consumer ordering and original arithmetic
remain reviewable in [the header](sources-used/expanded_fp16_layer_cache.hpp)
and [the prefill diff](sources-used/prefill-cache.patch). Clean 0/128/256
timings are not run: the prepared controller's acceptance prerequisite failed.
Both candidates remain private and unadopted. Repeated restoration and all
262144-cell CLI/serve, clipped-tail, refusal and later-valid gates remain open.

Earlier upstream/dequant receipts incorrectly describe this 2048-input test
as two chunks. The configured maximum is1024, but the initial chunk is256:
actual prefill sizes are256,1024,767 (2047 tokens), with the final input in
decode. The [correction](prefill-2048-chunk-label-correction.json) preserves
all144 successful prior trace entries: 48 layers times three chunks, offsets
0/256/1280. Timings and test conditions are unchanged; prior frozen manifests
are retained. Binaries, whole heads and large logs remain private with hashes;
the manifest covers every archived file except itself.
''')
print(json.dumps(manifest(directory), indent=2))
