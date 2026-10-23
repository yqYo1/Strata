"""Main-authored production-source discriminator; repository LICENSE applies.

Production stays byte-for-byte unchanged. Both probes retain prefix pack and
admission. L retains row publication but no KV payload SLM; S has the original
K/V store layouts and five explicit tile barriers. Checksums replace consumers;
their cost and changed occupancy forbid additive time attribution.
"""
from pathlib import Path
import subprocess

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
SOURCE = 'src/kernels/sycl/qsa_bf16_prefix.cpp'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('calibration anchor not unique: ' + before[:64])
    return text.replace(before, after, 1)


PROBE = r'''
  auto event = queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<int64_t, 1> rows(Cells, handler);
    // L keeps only the row-publication arena. S has the production KV arena.
    sycl::local_accessor<MatrixTile<Calibration == 2 ? Cells * Dim : 2>, 1>
        storage(1, handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(queries) * 2 * Workgroup, Workgroup),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int tid = item.get_local_linear_id();
      const int64_t batch = int64_t(item.get_group_linear_id()) / 2;
      const int kv_head = int(item.get_group_linear_id() % 2);
      const int width = steps[batch * kStepCount + kStepWidth];
      const int count = width >= 0 && int64_t(width) <= cap ? width : 0;
      uint32_t checksum = 0;
      auto kv = sycl::address_space_cast<sycl::access::address_space::local_space,
          sycl::access::decorated::no>(storage[0].values);
      for (int64_t first = 0; first < count; first += Cells) {
        if (tid < Cells) {
          int64_t row = -1;
          if (first + tid < count) {
            const int64_t cell = ids[batch * cap + first + tid];
            const int64_t logical = cell >= 0 ? cell / shape.page_size : -1;
            const int page = logical >= 0 && logical < bounds.logical_pages
                ? pools.page_table[logical] : -1;
            if (cell < live_cells && page >= 0 && page < bounds.physical_pages)
              row = (logical * 2 + kv_head) * shape.page_size + cell % shape.page_size;
          }
          rows[tid] = row;
          if (row >= 0) checksum += uint32_t(packed.bad_rows[row]);
        }
        item.barrier(sycl::access::fence_space::local_space); // publication
        for (int i = tid; i < Cells * (Dim / 4); i += Workgroup) {
          const int cell = i % Cells, d = (i / Cells) * 4;
          const int64_t row = rows[cell];
          sycl::vec<uint16_t, 4> words{uint16_t(0)};
          if (row >= 0)
            words.load(0, sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(packed.k + row * Dim + d));
          for (int j = 0; j < 4; ++j) {
            if constexpr (Calibration == 2)
              kv[(d + j) * Cells + cell] = sycl::bit_cast<BFloat>(uint16_t(words[j]));
            else checksum += uint32_t(words[j]);
          }
        }
        if constexpr (Calibration == 2) {
          item.barrier(sycl::access::fence_space::local_space); // K ready
          // Cross-thread consumer prevents forwarding away the SLM stores.
          for (int i = (tid + 1) % Workgroup; i < Cells * Dim; i += Workgroup)
            checksum += sycl::bit_cast<uint16_t>(kv[i]);
          item.barrier(sycl::access::fence_space::local_space); // K readers done
        }
        for (int i = tid; i < Cells * (Dim / 4); i += Workgroup) {
          const int cell = i / (Dim / 4), d = (i % (Dim / 4)) * 4;
          const int64_t row = rows[cell];
          sycl::vec<uint16_t, 4> words{uint16_t(0)};
          if (row >= 0)
            words.load(0, sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(packed.v + row * Dim + d));
          for (int j = 0; j < 4; ++j) {
            if constexpr (Calibration == 2)
              kv[cell * Dim + d + j] = sycl::bit_cast<BFloat>(uint16_t(words[j]));
            else checksum += uint32_t(words[j]);
          }
        }
        if constexpr (Calibration == 2) {
          item.barrier(sycl::access::fence_space::local_space); // V ready
          for (int i = (tid + 1) % Workgroup; i < Cells * Dim; i += Workgroup)
            checksum += sycl::bit_cast<uint16_t>(kv[i]);
        }
        item.barrier(sycl::access::fence_space::local_space); // readers before next rows
      }
      const uint32_t sum = sycl::reduce_over_group(item.get_group(), checksum,
                                                  sycl::plus<uint32_t>());
      if (!tid) sink[item.get_group_linear_id()] = sum;
    });
  });
  sycl_backend::finish(stream, event);
  return true;
'''


def generated(project):
    project = Path(project)
    raw = subprocess.check_output(['git', '-C', str(project), 'show', BASE + ':' + SOURCE])
    if raw != (project / SOURCE).read_bytes():
        raise ValueError('selected production prefix source changed')
    text = raw.decode()
    # Only the private template and the selected specialization are needed.
    text = text[:text.index('\nbool qsa_bf16_prefix_batch(')].rstrip() + '\n'
    text = replace_once(text, 'namespace strata::kernels {',
        '#include "qsa_resource_calibration.hpp"\nnamespace strata::kernels::resource_calibration {')
    text = replace_once(text, 'bool OrderedExp = false, bool CaptureState = false>',
        'bool OrderedExp = false, bool CaptureState = false, int Calibration = 0>')
    text = replace_once(text, 'QsaPrefixStateProbe probe = {}) {',
        'QsaPrefixStateProbe probe = {}, uint32_t *sink = nullptr) {')
    anchor = '  const auto packed = scratch;'
    validation = '''  if constexpr (Calibration != 0) {
    const size_t sink_bytes = checked_count(checked_count(queries, 2), 4);
    validate_spans({{sink, sink_bytes}},
        {pk, pv, bad, {output, query_bytes}, {query, query_bytes},
         {ids, checked_count(checked_count(queries, cap), 4)},
         {steps, checked_count(checked_count(queries, kStepCount), 4)},
         {pools.page_table, checked_count(bounds.logical_pages, 4)},
         {pools.k_q, checked_count(rows_count, Dim)}, {pools.v_q, checked_count(rows_count, Dim)},
         {pools.k_scale, checked_count(checked_count(rows_count, 4), 2)},
         {pools.v_scale, checked_count(checked_count(rows_count, 4), 2)}}, 4);
    validate_device_span(sink, sink_bytes, queue);
  }
'''
    text = replace_once(text, anchor, validation + anchor)
    start = text.index('  auto event = queue.submit(')
    if not text.endswith('  return true;\n}\n'):
        raise ValueError('unexpected production template ending')
    attention = text[start:-2]  # Includes original return; excludes closing brace.
    text = text[:start] + '  if constexpr (Calibration == 0) {\n' + attention + '\n  } else {\n' + PROBE + '\n  }\n}\n'
    text += '''
bool run(Arm arm, const uint16_t *query, const QsaAttnPools &pools,
         const int32_t *ids, const int32_t *steps, int64_t cap,
         const QsaShapes &shape, uint16_t *output, int64_t queries,
         const QsaPromptBf16Bounds &bounds, int64_t live_cells,
         const QsaBf16PackedScratch &scratch, void *stream, uint32_t *sink) {
  switch (arm) {
    case Arm::control:
      return prefix_batch_impl<Cells+1,false,false,true,true,true,true,false,0>(
          query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,stream,{},sink);
    case Arm::load:
      return prefix_batch_impl<Cells+1,false,false,true,true,true,true,false,1>(
          query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,stream,{},sink);
    case Arm::stage:
      return prefix_batch_impl<Cells+1,false,false,true,true,true,true,false,2>(
          query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,stream,{},sink);
  }
  throw std::invalid_argument("unknown calibration arm");
}
} // namespace strata::kernels::resource_calibration
'''
    return text.encode()
