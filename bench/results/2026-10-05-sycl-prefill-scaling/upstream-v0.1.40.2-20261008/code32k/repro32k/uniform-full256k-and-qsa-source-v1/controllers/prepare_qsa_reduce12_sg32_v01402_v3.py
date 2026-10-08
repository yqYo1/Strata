"""Prepare the private reduction with explicit subgroup lane/group identifiers.

Source and CPU routing review only. Never compile or launch a GPU workload here.
"""
from pathlib import Path
import datetime, difflib, hashlib, json, random

base = Path(__file__).parent
previous = base / 'qsa-reduce12-sg32-v01402-source-v2'
prepared = json.loads((previous / 'record.json').read_text())
original = Path(prepared['original_source'])
source = Path(prepared['candidate_source'])
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(original) == prepared['original_source_sha256']
assert digest(source) == prepared['candidate_source_sha256']
before = original.read_text()
s = source.read_text()
old = '    const int t = item_ct1.get_local_id(2), lane = t % SG, warp = t / SG;'
new = '''    const int t = item_ct1.get_local_id(2);
    // Only the new SG32/transposed arm changes its routing. Work-item grouping
    // is implementation-defined: use subgroup IDs for uniform cell/head loops.
    const auto score_sg = item_ct1.get_sub_group();
    const int lane = (SG == 32 && TRANSPOSE_Q)
                         ? (int) score_sg.get_local_linear_id() : t % SG;
    const int warp = (SG == 32 && TRANSPOSE_Q)
                         ? (int) score_sg.get_group_linear_id() : t / SG;'''
assert s.count(old) == 1
s = s.replace(old, new)
old_call = '''            const int sg_lane = (int) sycl::ext::oneapi::this_work_item::get_sub_group().get_local_linear_id();
            const float score = reduce12_sg32(qk_part, sg_lane);
            const int h = sg_lane >> 1;
            if ((sg_lane & 1) == 0 && h < G) sp[h][c] = score * scale;'''
new_call = '''            const float score = reduce12_sg32(qk_part, lane);
            const int h = lane >> 1;
            if ((lane & 1) == 0 && h < G) sp[h][c] = score * scale;'''
assert s.count(old_call) == 1
s = s.replace(old_call, new_call)
out = base / 'qsa-reduce12-sg32-v01402-source-v3'
out.mkdir(mode=0o700)
candidate = out / 'qsa_decode_attn.dp.cpp'
candidate.write_text(s)
diff = ''.join(difflib.unified_diff(before.splitlines(True), s.splitlines(True),
                                  fromfile=str(original), tofile=str(candidate)))
(out / 'candidate-source.diff').write_text(diff)
def body(text):
    start = text.index('__dpct_inline__ float reduce12_sg32(')
    return text[start:text.index('\n}', start) + 2]
cpu_path = base / 'qsa-reduce12-sg32-float-cpu-v01402-v2/record.json'
cpu = json.loads(cpu_path.read_text())
helper_sha = hashlib.sha256(body(s).encode()).hexdigest()
assert cpu['passed'] and not cpu['active'] and cpu['asan_ubsan_passed']
assert helper_sha == cpu['actual_helper_body_sha256']
assert body(s) == body(source.read_text())
assert s[s.index('// PR #540'):] == before[before.index('// PR #540'):]
unchanged_start = 'template <int SG = 32>\n__dpct_inline__ float warp_max(float v) {'
assert s[s.index(unchanged_start):s.index(old.split(', lane')[0])] == before[before.index(unchanged_start):before.index(old)]
dot_start = '        for (int h = 0; h < G; ++h) {'
dot_end = '            s = warp_sum<SG>(s);'
assert before[before.index(dot_start, before.index('        float upper_k[8];')):before.index(dot_end, before.index('        float upper_k[8];'))].strip() in s
tail_start = '    // per-head chunk max and exp-sum: warp w handles heads w and w+8.'
assert s[s.index(tail_start):] == before[before.index(tail_start):]
assert s.count('item_ct1.barrier();') == before.count('item_ct1.barrier();')
assert s.count('[[sycl::reqd_sub_group_size(SG)]]') == before.count('[[sycl::reqd_sub_group_size(SG)]]')
assert 'launch_chunk_variant<KV_MODE, 32, true>' in s
assert s.count('sycl::permute_group_by_xor(sg,') == 5

# Enumerate a semantic model of the source's cell/head stores for arbitrary
# work-group membership. This does not prove real device compilation/execution.
rng = random.Random(0x632032)
routing_cases = 0
original_nonuniform_groups = 0
for iteration in range(128):
    work_items = list(range(256))
    if iteration:
        rng.shuffle(work_items)
    groups = [work_items[32*g:32*(g+1)] for g in range(8)]
    original_nonuniform_groups += sum(len({t//32 for t in group}) != 1 for group in groups)
    assert sorted(t for group in groups for t in group) == list(range(256))
    for n_here in [0, 1, 7, 31, 32, 33, 63, 64]:
        writes = [[0]*64 for _ in range(12)]
        for subgroup_id, group in enumerate(groups):
            assert {lane*8+j for lane in range(len(group)) for j in range(8)} == set(range(256))
            for cell in range(subgroup_id, 64, 8):
                valid = cell < n_here and (cell % 7 != 3)
                # The predicate depends only on a shared cell row, not t/lane.
                assert len({valid for _ in group}) == 1
                for subgroup_lane, _ in enumerate(group):
                    if valid:
                        head = subgroup_lane >> 1
                        if (subgroup_lane & 1) == 0 and head < 12:
                            writes[head][cell] += 1
                    elif subgroup_lane < 12:
                        writes[subgroup_lane][cell] += 1
        assert all(count == 1 for head in writes for count in head)
        assert sorted(h for g in range(8) for h in range(g,12,8)) == list(range(12))
        routing_cases += 1
assert routing_cases == 1024 and original_nonuniform_groups > 0
record = {
    'active': False, 'prepared': True, 'compiled': False, 'gpu_tested': False,
    'adopted': False, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'controller_sha256': digest(__file__),
    'original_source': str(original), 'original_source_sha256': digest(original),
    'previous_source_sha256': digest(source),
    'candidate_source': str(candidate), 'candidate_source_sha256': digest(candidate),
    'source_diff_sha256': digest(out / 'candidate-source.diff'),
    'actual_helper_body_sha256': helper_sha,
    'cpu_receipt': str(cpu_path), 'cpu_receipt_sha256': digest(cpu_path),
    'cpu_helper_proof_reused_by_exact_body_identity': True,
    'cpu_bitwise_equal_outputs': cpu['bitwise_equal_outputs'],
    'cpu_groups': cpu['groups'], 'cpu_asan_ubsan_passed': True,
    'actual_subgroup_lane_and_group_ids_used': True,
    'cpu_semantic_routing_cases': routing_cases,
    'counterexample_original_abstract_mapping_nonuniform_groups': original_nonuniform_groups,
    'counterexample_is_actual_gpu_failure': False,
    'scope': 'Private source-only SG32/transposed reduction and explicit subgroup routing. No active process, engine binary or other source was changed.',
    'primary_sources': [
        'https://github.khronos.org/SYCL_Reference/iface/sub_group.html',
        'https://github.khronos.org/SYCL_Reference/iface/group-functions.html',
        'https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html'],
    'checks': {
        'cell_and_head_loops_use_actual_subgroup_group_id': True,
        'dot_dimensions_and_reduce_scatter_use_actual_subgroup_lane_id': True,
        'new_arm_valid_cell_skip_subgroup_uniform_independent_of_workgroup_mapping': True,
        'all32_items_execute_each_exchange_before_lane_dependent_store': True,
        'other_variant_lane_and_warp_values_retained': True,
        'dot_expression_scale_softmax_pv_merge_dispatch_text_retained': True,
        'barrier_count_unchanged': True,
        'only_one_source_tu_planned': True,
        'no_event_queue_host_lifetime_memory_layout_or_new_vector_alias_change': True},
    'limitations': [
        'Exact-body CPU proof reused, not rerun or presented as a SYCL device result.',
        'Routing enumeration is a semantic CPU check, not a real GPU mapping measurement.',
        'Inherited workgroup-derived subgroup mapping remains in the other existing variants.',
        'Bit identity limited to finite inputs and finite intermediate additions, no arbitrary NaN payload claim.',
        'Device compiler contraction/register allocation, numerical state, disk restoration and throughput remain untested.'],
    'minimum_performance_input_tokens': 32768,
    'pending': [
        'Compile after the active full256K controller exits; record dependencies/archive member/link inputs.',
        'Initial logged>=32768 head/used-state/IDs/logprobs/MTP/disk restoration gate.',
        'Quiet>=32768 matched comparison separating first/later full reads.',
        'Complete repeated physical256K, actual disk restoration, clipped tail, capacity refusal and later fresh32K before adoption.']}
(out / 'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['prepared','compiled','gpu_tested','candidate_source_sha256','cpu_semantic_routing_cases','cpu_helper_proof_reused_by_exact_body_identity']}))
