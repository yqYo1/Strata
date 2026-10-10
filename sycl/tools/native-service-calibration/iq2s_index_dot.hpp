#pragma once
#include "quants.h"
namespace isolated_iq2s {
float direct_control(int,const block_iq2_s*,const block_q8_K*);
float index_candidate(int,const block_iq2_s*,const block_q8_K*);
}
