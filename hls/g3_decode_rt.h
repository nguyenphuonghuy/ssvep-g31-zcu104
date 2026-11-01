#pragma once
#include <cstdint>
#include "dims.h"
// mode 0: nap tham so tu prm (status = 1 neu MAGIC/T dung, -1 neu sai) va ghi 5 checksum doc nguoc vao score[0..4];
// mode 1: giai ma 1 luot thu (status = 2).
void g3_decode_rt(const int16_t in[C * (S0 + T)], const int32_t prm[PRM_LEN], int64_t score[K], int32_t pred[1],
                  int32_t status[1], int32_t mode);
