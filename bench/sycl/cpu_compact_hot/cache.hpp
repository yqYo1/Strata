// Component compatibility names; production owns the implementation.
#pragma once
#include "strata/kernels/cpu/compact_gu.hpp"
#include "strata/kernels/cpu/detail/compact_iq3xxs.hpp"
#include "ggml.h"
#include <atomic>
#include <limits>
#include <stdexcept>
#include <vector>
namespace compact_gu_cache {
using strata::kernels::cpu::compact_gu::Cache;
using strata::kernels::cpu::compact_gu::Lease;
using strata::kernels::cpu::compact_gu::Key;
using strata::kernels::cpu::compact_gu::available;
using strata::kernels::cpu::compact_gu::supported;
using strata::kernels::cpu::compact_gu::projection_blocks;
using strata::kernels::cpu::compact_gu::raw_projection_bytes;
using Block = compact_iq3xxs::Block;
}
