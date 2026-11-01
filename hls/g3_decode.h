#pragma once
#include <cstdint>
#include "params.h"
void g3_decode(const int16_t in[C * (S0 + T)], int64_t score[K], int32_t pred[1]);
