#pragma once
#include <cstdint>
struct signindexed_iq3s { uint16_t d; uint16_t indices[64]; uint8_t scales[4]; };
static_assert(sizeof(signindexed_iq3s)==134);
