#pragma once
#include <cstdint>
struct preindexed_iq3s { uint16_t d; uint16_t indices[64]; uint8_t signs[32]; uint8_t scales[4]; };
struct preindexed_iq2s { uint16_t d; uint16_t indices[32]; uint8_t signs[32]; uint8_t scales[8]; };
static_assert(sizeof(preindexed_iq3s)==166);
static_assert(sizeof(preindexed_iq2s)==106);
