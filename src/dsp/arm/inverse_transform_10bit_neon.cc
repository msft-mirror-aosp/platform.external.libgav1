// Copyright 2021 The libgav1 Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/dsp/inverse_transform.h"
#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/dsp/arm/common_neon.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/array_2d.h"
#include "src/utils/common.h"
#include "src/utils/compiler_attributes.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

// Include the constants and utility functions inside the anonymous namespace.
#include "src/dsp/inverse_transform.inc"

//------------------------------------------------------------------------------

LIBGAV1_ALWAYS_INLINE void Transpose4x4(const int32x4_t in[4],
                                        int32x4_t out[4]) {
  // in:
  // 00 01 02 03
  // 10 11 12 13
  // 20 21 22 23
  // 30 31 32 33

  // 00 10 02 12   a.val[0]
  // 01 11 03 13   a.val[1]
  // 20 30 22 32   b.val[0]
  // 21 31 23 33   b.val[1]
  const int32x4x2_t a = vtrnq_s32(in[0], in[1]);
  const int32x4x2_t b = vtrnq_s32(in[2], in[3]);
  out[0] = vextq_s32(vextq_s32(a.val[0], a.val[0], 2), b.val[0], 2);
  out[1] = vextq_s32(vextq_s32(a.val[1], a.val[1], 2), b.val[1], 2);
  out[2] = vextq_s32(a.val[0], vextq_s32(b.val[0], b.val[0], 2), 2);
  out[3] = vextq_s32(a.val[1], vextq_s32(b.val[1], b.val[1], 2), 2);
  // out:
  // 00 10 20 30
  // 01 11 21 31
  // 02 12 22 32
  // 03 13 23 33
}

//------------------------------------------------------------------------------
template <int store_count>
LIBGAV1_ALWAYS_INLINE void StoreDst(int32_t* dst, int32_t stride, int32_t idx,
                                    const int32x4_t* const s) {
  assert(store_count % 4 == 0);
  for (int i = 0; i < store_count; i += 4) {
    vst1q_s32(&dst[i * stride + idx], s[i]);
    vst1q_s32(&dst[(i + 1) * stride + idx], s[i + 1]);
    vst1q_s32(&dst[(i + 2) * stride + idx], s[i + 2]);
    vst1q_s32(&dst[(i + 3) * stride + idx], s[i + 3]);
  }
}

template <int load_count>
LIBGAV1_ALWAYS_INLINE void LoadSrc(const int32_t* src, int32_t stride,
                                   int32_t idx, int32x4_t* x) {
  assert(load_count % 4 == 0);
  for (int i = 0; i < load_count; i += 4) {
    x[i] = vld1q_s32(&src[i * stride + idx]);
    x[i + 1] = vld1q_s32(&src[(i + 1) * stride + idx]);
    x[i + 2] = vld1q_s32(&src[(i + 2) * stride + idx]);
    x[i + 3] = vld1q_s32(&src[(i + 3) * stride + idx]);
  }
}

// Butterfly rotate 4 values.
LIBGAV1_ALWAYS_INLINE void ButterflyRotation_4(int32x4_t* a, int32x4_t* b,
                                               const int angle,
                                               const bool flip) {
  const int32_t cos128 = Cos128(angle);
  const int32_t sin128 = Sin128(angle);
  const int32x4_t acc_x = vmulq_n_s32(*a, cos128);
  const int32x4_t acc_y = vmulq_n_s32(*a, sin128);
  // The max range for the input is 18 bits. The cos128/sin128 is 13 bits,
  // which leaves 1 bit for the add/subtract. For 10bpp, x/y will fit in a 32
  // bit lane.
  const int32x4_t x0 = vmlsq_n_s32(acc_x, *b, sin128);
  const int32x4_t y0 = vmlaq_n_s32(acc_y, *b, cos128);
  const int32x4_t x = vrshrq_n_s32(x0, 12);
  const int32x4_t y = vrshrq_n_s32(y0, 12);
  if (flip) {
    *a = y;
    *b = x;
  } else {
    *a = x;
    *b = y;
  }
}

LIBGAV1_ALWAYS_INLINE void HadamardRotation(int32x4_t* a, int32x4_t* b,
                                            bool flip) {
  int32x4_t x, y;
  if (flip) {
    y = vqaddq_s32(*b, *a);
    x = vqsubq_s32(*b, *a);
  } else {
    x = vqaddq_s32(*a, *b);
    y = vqsubq_s32(*a, *b);
  }
  *a = x;
  *b = y;
}

LIBGAV1_ALWAYS_INLINE void HadamardRotation(int32x4_t* a, int32x4_t* b,
                                            bool flip, const int32x4_t* min,
                                            const int32x4_t* max) {
  int32x4_t x, y;
  if (flip) {
    y = vqaddq_s32(*b, *a);
    x = vqsubq_s32(*b, *a);
  } else {
    x = vqaddq_s32(*a, *b);
    y = vqsubq_s32(*a, *b);
  }
  *a = vmaxq_s32(vminq_s32(x, *max), *min);
  *b = vmaxq_s32(vminq_s32(y, *max), *min);
}

using ButterflyRotationFunc = void (*)(int32x4_t* a, int32x4_t* b, int angle,
                                       bool flip);

//------------------------------------------------------------------------------
// Discrete Cosine Transforms (DCT).

template <int width>
LIBGAV1_ALWAYS_INLINE bool DctDcOnly(void* dest, int adjusted_tx_height,
                                     bool should_round, int row_shift) {
  if (adjusted_tx_height > 1) return false;

  auto* dst = static_cast<int32_t*>(dest);
  const int32x4_t v_src = vdupq_n_s32(dst[0]);
  const uint32x4_t v_mask = vdupq_n_u32(should_round ? 0xffffffff : 0);
  const int32x4_t v_src_round =
      vqrdmulhq_n_s32(v_src, kTransformRowMultiplier << (31 - 12));
  const int32x4_t s0 = vbslq_s32(v_mask, v_src_round, v_src);
  const int32_t cos128 = Cos128(32);
  const int32x4_t xy = vqrdmulhq_n_s32(s0, cos128 << (31 - 12));
  // vqrshlq_s32 will shift right if shift value is negative.
  const int32x4_t xy_shifted = vqrshlq_s32(xy, vdupq_n_s32(-row_shift));
  // Clamp result to signed 16 bits.
  const int32x4_t result = vmovl_s16(vqmovn_s32(xy_shifted));
  if (width == 4) {
    vst1q_s32(dst, result);
  } else {
    for (int i = 0; i < width; i += 4) {
      vst1q_s32(dst, result);
      dst += 4;
    }
  }
  return true;
}

template <int height>
LIBGAV1_ALWAYS_INLINE bool DctDcOnlyColumn(void* dest, int adjusted_tx_height,
                                           int width) {
  if (adjusted_tx_height > 1) return false;

  auto* dst = static_cast<int32_t*>(dest);
  const int32_t cos128 = Cos128(32);

  // Calculate dc values for first row.
  if (width == 4) {
    const int32x4_t v_src = vld1q_s32(dst);
    const int32x4_t xy = vqrdmulhq_n_s32(v_src, cos128 << (31 - 12));
    vst1q_s32(dst, xy);
  } else {
    int i = 0;
    do {
      const int32x4_t v_src = vld1q_s32(&dst[i]);
      const int32x4_t xy = vqrdmulhq_n_s32(v_src, cos128 << (31 - 12));
      vst1q_s32(&dst[i], xy);
      i += 4;
    } while (i < width);
  }

  // Copy first row to the rest of the block.
  for (int y = 1; y < height; ++y) {
    memcpy(&dst[y * width], dst, width * sizeof(dst[0]));
  }
  return true;
}

template <ButterflyRotationFunc butterfly_rotation>
LIBGAV1_ALWAYS_INLINE void Dct4Stages(int32x4_t* s, const int32x4_t* min,
                                      const int32x4_t* max,
                                      const bool is_last_stage) {
  // stage 12.
  butterfly_rotation(&s[0], &s[1], 32, true);
  butterfly_rotation(&s[2], &s[3], 48, false);

  // stage 17.
  if (is_last_stage) {
    HadamardRotation(&s[0], &s[3], false);
    HadamardRotation(&s[1], &s[2], false);
  } else {
    HadamardRotation(&s[0], &s[3], false, min, max);
    HadamardRotation(&s[1], &s[2], false, min, max);
  }
}

template <ButterflyRotationFunc butterfly_rotation>
LIBGAV1_ALWAYS_INLINE void Dct4_NEON(void* dest, int32_t step, bool is_row,
                                     int row_shift) {
  auto* const dst = static_cast<int32_t*>(dest);
  // When |is_row| is true, set range to the row range, otherwise, set to the
  // column range.
  const int32_t range = is_row ? kBitdepth10 + 7 : 15;
  const int32x4_t min = vdupq_n_s32(-(1 << range));
  const int32x4_t max = vdupq_n_s32((1 << range) - 1);
  int32x4_t s[4], x[4];

  LoadSrc<4>(dst, step, 0, x);
  if (is_row) {
    Transpose4x4(x, x);
  }

  // stage 1.
  // kBitReverseLookup 0, 2, 1, 3
  s[0] = x[0];
  s[1] = x[2];
  s[2] = x[1];
  s[3] = x[3];

  Dct4Stages<butterfly_rotation>(s, &min, &max, /*is_last_stage=*/true);

  if (is_row) {
    const int32x4_t v_row_shift = vdupq_n_s32(-row_shift);
    for (int i = 0; i < 4; ++i) {
      s[i] = vmovl_s16(vqmovn_s32(vqrshlq_s32(s[i], v_row_shift)));
    }
    Transpose4x4(s, s);
  }
  StoreDst<4>(dst, step, 0, s);
}

template <ButterflyRotationFunc butterfly_rotation>
LIBGAV1_ALWAYS_INLINE void Dct8Stages(int32x4_t* s, const int32x4_t* min,
                                      const int32x4_t* max,
                                      const bool is_last_stage) {
  // stage 8.
  butterfly_rotation(&s[4], &s[7], 56, false);
  butterfly_rotation(&s[5], &s[6], 24, false);

  // stage 13.
  HadamardRotation(&s[4], &s[5], false, min, max);
  HadamardRotation(&s[6], &s[7], true, min, max);

  // stage 18.
  butterfly_rotation(&s[6], &s[5], 32, true);

  // stage 22.
  if (is_last_stage) {
    HadamardRotation(&s[0], &s[7], false);
    HadamardRotation(&s[1], &s[6], false);
    HadamardRotation(&s[2], &s[5], false);
    HadamardRotation(&s[3], &s[4], false);
  } else {
    HadamardRotation(&s[0], &s[7], false, min, max);
    HadamardRotation(&s[1], &s[6], false, min, max);
    HadamardRotation(&s[2], &s[5], false, min, max);
    HadamardRotation(&s[3], &s[4], false, min, max);
  }
}

// Process dct8 rows or columns, depending on the |is_row| flag.
template <ButterflyRotationFunc butterfly_rotation>
LIBGAV1_ALWAYS_INLINE void Dct8_NEON(void* dest, int32_t step, bool is_row,
                                     int row_shift) {
  auto* const dst = static_cast<int32_t*>(dest);
  const int32_t range = is_row ? kBitdepth10 + 7 : 15;
  const int32x4_t min = vdupq_n_s32(-(1 << range));
  const int32x4_t max = vdupq_n_s32((1 << range) - 1);
  int32x4_t s[8], x[8];

  if (is_row) {
    LoadSrc<4>(dst, step, 0, &x[0]);
    LoadSrc<4>(dst, step, 4, &x[4]);
    Transpose4x4(&x[0], &x[0]);
    Transpose4x4(&x[4], &x[4]);
  } else {
    LoadSrc<8>(dst, step, 0, &x[0]);
  }

  // stage 1.
  // kBitReverseLookup 0, 4, 2, 6, 1, 5, 3, 7,
  s[0] = x[0];
  s[1] = x[4];
  s[2] = x[2];
  s[3] = x[6];
  s[4] = x[1];
  s[5] = x[5];
  s[6] = x[3];
  s[7] = x[7];

  Dct4Stages<butterfly_rotation>(s, &min, &max, /*is_last_stage=*/false);
  Dct8Stages<butterfly_rotation>(s, &min, &max, /*is_last_stage=*/true);

  if (is_row) {
    const int32x4_t v_row_shift = vdupq_n_s32(-row_shift);
    for (int i = 0; i < 8; ++i) {
      s[i] = vmovl_s16(vqmovn_s32(vqrshlq_s32(s[i], v_row_shift)));
    }
    Transpose4x4(&s[0], &s[0]);
    Transpose4x4(&s[4], &s[4]);
    StoreDst<4>(dst, step, 0, &s[0]);
    StoreDst<4>(dst, step, 4, &s[4]);
  } else {
    StoreDst<8>(dst, step, 0, &s[0]);
  }
}

template <ButterflyRotationFunc butterfly_rotation>
LIBGAV1_ALWAYS_INLINE void Dct16Stages(int32x4_t* s, const int32x4_t* min,
                                       const int32x4_t* max,
                                       const bool is_last_stage) {
  // stage 5.
  butterfly_rotation(&s[8], &s[15], 60, false);
  butterfly_rotation(&s[9], &s[14], 28, false);
  butterfly_rotation(&s[10], &s[13], 44, false);
  butterfly_rotation(&s[11], &s[12], 12, false);

  // stage 9.
  HadamardRotation(&s[8], &s[9], false, min, max);
  HadamardRotation(&s[10], &s[11], true, min, max);
  HadamardRotation(&s[12], &s[13], false, min, max);
  HadamardRotation(&s[14], &s[15], true, min, max);

  // stage 14.
  butterfly_rotation(&s[14], &s[9], 48, true);
  butterfly_rotation(&s[13], &s[10], 112, true);

  // stage 19.
  HadamardRotation(&s[8], &s[11], false, min, max);
  HadamardRotation(&s[9], &s[10], false, min, max);
  HadamardRotation(&s[12], &s[15], true, min, max);
  HadamardRotation(&s[13], &s[14], true, min, max);

  // stage 23.
  butterfly_rotation(&s[13], &s[10], 32, true);
  butterfly_rotation(&s[12], &s[11], 32, true);

  // stage 26.
  if (is_last_stage) {
    HadamardRotation(&s[0], &s[15], false);
    HadamardRotation(&s[1], &s[14], false);
    HadamardRotation(&s[2], &s[13], false);
    HadamardRotation(&s[3], &s[12], false);
    HadamardRotation(&s[4], &s[11], false);
    HadamardRotation(&s[5], &s[10], false);
    HadamardRotation(&s[6], &s[9], false);
    HadamardRotation(&s[7], &s[8], false);
  } else {
    HadamardRotation(&s[0], &s[15], false, min, max);
    HadamardRotation(&s[1], &s[14], false, min, max);
    HadamardRotation(&s[2], &s[13], false, min, max);
    HadamardRotation(&s[3], &s[12], false, min, max);
    HadamardRotation(&s[4], &s[11], false, min, max);
    HadamardRotation(&s[5], &s[10], false, min, max);
    HadamardRotation(&s[6], &s[9], false, min, max);
    HadamardRotation(&s[7], &s[8], false, min, max);
  }
}

// Process dct16 rows or columns, depending on the |is_row| flag.
template <ButterflyRotationFunc butterfly_rotation>
LIBGAV1_ALWAYS_INLINE void Dct16_NEON(void* dest, int32_t step, bool is_row,
                                      int row_shift) {
  auto* const dst = static_cast<int32_t*>(dest);
  const int32_t range = (is_row) ? (kBitdepth10 + 7) : 15;
  const int32x4_t min = vdupq_n_s32(-(1 << range));
  const int32x4_t max = vdupq_n_s32((1 << range) - 1);
  int32x4_t s[16], x[16];

  if (is_row) {
    for (int idx = 0; idx < 16; idx += 8) {
      LoadSrc<4>(dst, step, idx, &x[idx]);
      LoadSrc<4>(dst, step, idx + 4, &x[idx + 4]);
      Transpose4x4(&x[idx], &x[idx]);
      Transpose4x4(&x[idx + 4], &x[idx + 4]);
    }
  } else {
    LoadSrc<16>(dst, step, 0, &x[0]);
  }

  // stage 1
  // kBitReverseLookup 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15,
  s[0] = x[0];
  s[1] = x[8];
  s[2] = x[4];
  s[3] = x[12];
  s[4] = x[2];
  s[5] = x[10];
  s[6] = x[6];
  s[7] = x[14];
  s[8] = x[1];
  s[9] = x[9];
  s[10] = x[5];
  s[11] = x[13];
  s[12] = x[3];
  s[13] = x[11];
  s[14] = x[7];
  s[15] = x[15];

  Dct4Stages<butterfly_rotation>(s, &min, &max, /*is_last_stage=*/false);
  Dct8Stages<butterfly_rotation>(s, &min, &max, /*is_last_stage=*/false);
  Dct16Stages<butterfly_rotation>(s, &min, &max, /*is_last_stage=*/true);

  if (is_row) {
    const int32x4_t v_row_shift = vdupq_n_s32(-row_shift);
    for (int i = 0; i < 16; ++i) {
      s[i] = vmovl_s16(vqmovn_s32(vqrshlq_s32(s[i], v_row_shift)));
    }
    for (int idx = 0; idx < 16; idx += 8) {
      Transpose4x4(&s[idx], &s[idx]);
      Transpose4x4(&s[idx + 4], &s[idx + 4]);
      StoreDst<4>(dst, step, idx, &s[idx]);
      StoreDst<4>(dst, step, idx + 4, &s[idx + 4]);
    }
  } else {
    StoreDst<16>(dst, step, 0, &s[0]);
  }
}

//------------------------------------------------------------------------------
// row/column transform loops

template <int tx_height>
LIBGAV1_ALWAYS_INLINE void FlipColumns(int32_t* source, int tx_width) {
  if (tx_width >= 16) {
    int i = 0;
    do {
      // 00 01 02 03
      const int32x4_t a = vld1q_s32(&source[i]);
      const int32x4_t b = vld1q_s32(&source[i + 4]);
      const int32x4_t c = vld1q_s32(&source[i + 8]);
      const int32x4_t d = vld1q_s32(&source[i + 12]);
      // 01 00 03 02
      const int32x4_t a_rev = vrev64q_s32(a);
      const int32x4_t b_rev = vrev64q_s32(b);
      const int32x4_t c_rev = vrev64q_s32(c);
      const int32x4_t d_rev = vrev64q_s32(d);
      // 03 02 01 00
      vst1q_s32(&source[i], vextq_s32(d_rev, d_rev, 2));
      vst1q_s32(&source[i + 4], vextq_s32(c_rev, c_rev, 2));
      vst1q_s32(&source[i + 8], vextq_s32(b_rev, b_rev, 2));
      vst1q_s32(&source[i + 12], vextq_s32(a_rev, a_rev, 2));
      i += 16;
    } while (i < tx_width * tx_height);
  } else if (tx_width == 8) {
    for (int i = 0; i < 8 * tx_height; i += 8) {
      // 00 01 02 03
      const int32x4_t a = vld1q_s32(&source[i]);
      const int32x4_t b = vld1q_s32(&source[i + 4]);
      // 01 00 03 02
      const int32x4_t a_rev = vrev64q_s32(a);
      const int32x4_t b_rev = vrev64q_s32(b);
      // 03 02 01 00
      vst1q_s32(&source[i], vextq_s32(b_rev, b_rev, 2));
      vst1q_s32(&source[i + 4], vextq_s32(a_rev, a_rev, 2));
    }
  } else {
    // Process two rows per iteration.
    for (int i = 0; i < 4 * tx_height; i += 8) {
      // 00 01 02 03
      const int32x4_t a = vld1q_s32(&source[i]);
      const int32x4_t b = vld1q_s32(&source[i + 4]);
      // 01 00 03 02
      const int32x4_t a_rev = vrev64q_s32(a);
      const int32x4_t b_rev = vrev64q_s32(b);
      // 03 02 01 00
      vst1q_s32(&source[i], vextq_s32(a_rev, a_rev, 2));
      vst1q_s32(&source[i + 4], vextq_s32(b_rev, b_rev, 2));
    }
  }
}

template <int tx_width>
LIBGAV1_ALWAYS_INLINE void ApplyRounding(int32_t* source, int num_rows) {
  // Process two rows per iteration.
  int i = 0;
  do {
    const int32x4_t a_lo = vld1q_s32(&source[i]);
    const int32x4_t a_hi = vld1q_s32(&source[i + 4]);
    const int32x4_t b_lo =
        vqrdmulhq_n_s32(a_lo, kTransformRowMultiplier << (31 - 12));
    const int32x4_t b_hi =
        vqrdmulhq_n_s32(a_hi, kTransformRowMultiplier << (31 - 12));
    vst1q_s32(&source[i], b_lo);
    vst1q_s32(&source[i + 4], b_hi);
    i += 8;
  } while (i < tx_width * num_rows);
}

template <int tx_width>
LIBGAV1_ALWAYS_INLINE void RowShift(int32_t* source, int num_rows,
                                    int row_shift) {
  // vqrshlq_s32 will shift right if shift value is negative.
  row_shift = -row_shift;

  // Process two rows per iteration.
  int i = 0;
  do {
    const int32x4_t residual0 = vld1q_s32(&source[i]);
    const int32x4_t residual1 = vld1q_s32(&source[i + 4]);
    vst1q_s32(&source[i], vqrshlq_s32(residual0, vdupq_n_s32(row_shift)));
    vst1q_s32(&source[i + 4], vqrshlq_s32(residual1, vdupq_n_s32(row_shift)));
    i += 8;
  } while (i < tx_width * num_rows);
}

template <int tx_height, bool enable_flip_rows = false>
LIBGAV1_ALWAYS_INLINE void StoreToFrameWithRound(
    Array2DView<uint16_t> frame, const int start_x, const int start_y,
    const int tx_width, const int32_t* source, TransformType tx_type) {
  const bool flip_rows =
      enable_flip_rows ? kTransformFlipRowsMask.Contains(tx_type) : false;
  const int stride = frame.columns();
  uint16_t* dst = frame[start_y] + start_x;

  if (tx_width == 4) {
    for (int i = 0; i < tx_height; ++i) {
      const int row = flip_rows ? (tx_height - i - 1) * 4 : i * 4;
      const int32x4_t residual = vld1q_s32(&source[row]);
      const uint16x4_t frame_data = vld1_u16(dst);
      const int32x4_t a = vrshrq_n_s32(residual, 4);
      const uint32x4_t b = vaddw_u16(vreinterpretq_u32_s32(a), frame_data);
      const uint16x4_t d = vqmovun_s32(vreinterpretq_s32_u32(b));
      vst1_u16(dst, vmin_u16(d, vdup_n_u16((1 << kBitdepth10) - 1)));
      dst += stride;
    }
  } else {
    for (int i = 0; i < tx_height; ++i) {
      const int y = start_y + i;
      const int row = flip_rows ? (tx_height - i - 1) * tx_width : i * tx_width;
      int j = 0;
      do {
        const int x = start_x + j;
        const int32x4_t residual = vld1q_s32(&source[row + j]);
        const int32x4_t residual_hi = vld1q_s32(&source[row + j + 4]);
        const uint16x8_t frame_data = vld1q_u16(frame[y] + x);
        const int32x4_t a = vrshrq_n_s32(residual, 4);
        const int32x4_t a_hi = vrshrq_n_s32(residual_hi, 4);
        const uint32x4_t b =
            vaddw_u16(vreinterpretq_u32_s32(a), vget_low_u16(frame_data));
        const uint32x4_t b_hi =
            vaddw_u16(vreinterpretq_u32_s32(a_hi), vget_high_u16(frame_data));
        const uint16x4_t d = vqmovun_s32(vreinterpretq_s32_u32(b));
        const uint16x4_t d_hi = vqmovun_s32(vreinterpretq_s32_u32(b_hi));
        vst1q_u16(frame[y] + x, vminq_u16(vcombine_u16(d, d_hi),
                                          vdupq_n_u16((1 << kBitdepth10) - 1)));
        j += 8;
      } while (j < tx_width);
    }
  }
}

void Dct4TransformLoopRow_NEON(TransformType /*tx_type*/, TransformSize tx_size,
                               int adjusted_tx_height, void* src_buffer,
                               int /*start_x*/, int /*start_y*/,
                               void* /*dst_frame*/) {
  auto* src = static_cast<int32_t*>(src_buffer);
  const int tx_height = kTransformHeight[tx_size];
  const bool should_round = (tx_height == 8);
  const int row_shift = (tx_height == 16);

  if (DctDcOnly<4>(src, adjusted_tx_height, should_round, row_shift)) {
    return;
  }

  if (should_round) {
    ApplyRounding<4>(src, adjusted_tx_height);
  }

  // Process 4 1d dct4 rows in parallel per iteration.
  int i = adjusted_tx_height;
  auto* data = src;
  do {
    Dct4_NEON<ButterflyRotation_4>(data, /*step=*/4, /*is_row=*/true,
                                   row_shift);
    data += 16;
    i -= 4;
  } while (i != 0);
}

void Dct4TransformLoopColumn_NEON(TransformType tx_type, TransformSize tx_size,
                                  int adjusted_tx_height, void* src_buffer,
                                  int start_x, int start_y, void* dst_frame) {
  auto* src = static_cast<int32_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];

  if (kTransformFlipColumnsMask.Contains(tx_type)) {
    FlipColumns<4>(src, tx_width);
  }

  if (!DctDcOnlyColumn<4>(src, adjusted_tx_height, tx_width)) {
    // Process 4 1d dct4 columns in parallel per iteration.
    int i = tx_width;
    auto* data = src;
    do {
      Dct4_NEON<ButterflyRotation_4>(data, tx_width, /*transpose=*/false,
                                     /*row_shift=*/0);
      data += 4;
      i -= 4;
    } while (i != 0);
  }

  auto& frame = *static_cast<Array2DView<uint16_t>*>(dst_frame);
  StoreToFrameWithRound<4>(frame, start_x, start_y, tx_width, src, tx_type);
}

void Dct8TransformLoopRow_NEON(TransformType /*tx_type*/, TransformSize tx_size,
                               int adjusted_tx_height, void* src_buffer,
                               int /*start_x*/, int /*start_y*/,
                               void* /*dst_frame*/) {
  auto* src = static_cast<int32_t*>(src_buffer);
  const bool should_round = kShouldRound[tx_size];
  const uint8_t row_shift = kTransformRowShift[tx_size];

  if (DctDcOnly<8>(src, adjusted_tx_height, should_round, row_shift)) {
    return;
  }

  if (should_round) {
    ApplyRounding<8>(src, adjusted_tx_height);
  }

  // Process 4 1d dct8 rows in parallel per iteration.
  int i = adjusted_tx_height;
  auto* data = src;
  do {
    Dct8_NEON<ButterflyRotation_4>(data, /*step=*/8, /*is_row=*/true,
                                   row_shift);
    data += 32;
    i -= 4;
  } while (i != 0);
}

void Dct8TransformLoopColumn_NEON(TransformType tx_type, TransformSize tx_size,
                                  int adjusted_tx_height, void* src_buffer,
                                  int start_x, int start_y, void* dst_frame) {
  auto* src = static_cast<int32_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];

  if (kTransformFlipColumnsMask.Contains(tx_type)) {
    FlipColumns<8>(src, tx_width);
  }

  if (!DctDcOnlyColumn<8>(src, adjusted_tx_height, tx_width)) {
    // Process 4 1d dct8 columns in parallel per iteration.
    int i = tx_width;
    auto* data = src;
    do {
      Dct8_NEON<ButterflyRotation_4>(data, tx_width, /*is_row=*/false,
                                     /*row_shift=*/0);
      data += 4;
      i -= 4;
    } while (i != 0);
  }
  auto& frame = *static_cast<Array2DView<uint16_t>*>(dst_frame);
  StoreToFrameWithRound<8>(frame, start_x, start_y, tx_width, src, tx_type);
}

void Dct16TransformLoopRow_NEON(TransformType /*tx_type*/,
                                TransformSize tx_size, int adjusted_tx_height,
                                void* src_buffer, int /*start_x*/,
                                int /*start_y*/, void* /*dst_frame*/) {
  auto* src = static_cast<int32_t*>(src_buffer);
  const bool should_round = kShouldRound[tx_size];
  const uint8_t row_shift = kTransformRowShift[tx_size];

  if (DctDcOnly<16>(src, adjusted_tx_height, should_round, row_shift)) {
    return;
  }

  if (should_round) {
    ApplyRounding<16>(src, adjusted_tx_height);
  }

  assert(adjusted_tx_height % 4 == 0);
  int i = adjusted_tx_height;
  auto* data = src;
  do {
    // Process 4 1d dct16 rows in parallel per iteration.
    Dct16_NEON<ButterflyRotation_4>(data, 16, /*is_row=*/true, row_shift);
    data += 64;
    i -= 4;
  } while (i != 0);
}

void Dct16TransformLoopColumn_NEON(TransformType tx_type, TransformSize tx_size,
                                   int adjusted_tx_height, void* src_buffer,
                                   int start_x, int start_y, void* dst_frame) {
  auto* src = static_cast<int32_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];

  if (kTransformFlipColumnsMask.Contains(tx_type)) {
    FlipColumns<16>(src, tx_width);
  }

  if (!DctDcOnlyColumn<16>(src, adjusted_tx_height, tx_width)) {
    // Process 4 1d dct8 columns in parallel per iteration.
    int i = tx_width;
    auto* data = src;
    do {
      Dct16_NEON<ButterflyRotation_4>(data, tx_width, /*is_row=*/false,
                                      /*row_shift=*/0);
      data += 4;
      i -= 4;
    } while (i != 0);
  }
  auto& frame = *static_cast<Array2DView<uint16_t>*>(dst_frame);
  StoreToFrameWithRound<16>(frame, start_x, start_y, tx_width, src, tx_type);
}

//------------------------------------------------------------------------------

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth10);
  assert(dsp != nullptr);
  // Maximum transform size for Dct is 64.
  dsp->inverse_transforms[k1DTransformDct][k1DTransformSize4][kRow] =
      Dct4TransformLoopRow_NEON;
  dsp->inverse_transforms[k1DTransformDct][k1DTransformSize4][kColumn] =
      Dct4TransformLoopColumn_NEON;
  dsp->inverse_transforms[k1DTransformDct][k1DTransformSize8][kRow] =
      Dct8TransformLoopRow_NEON;
  dsp->inverse_transforms[k1DTransformDct][k1DTransformSize8][kColumn] =
      Dct8TransformLoopColumn_NEON;
  dsp->inverse_transforms[k1DTransformDct][k1DTransformSize16][kRow] =
      Dct16TransformLoopRow_NEON;
  dsp->inverse_transforms[k1DTransformDct][k1DTransformSize16][kColumn] =
      Dct16TransformLoopColumn_NEON;
}

}  // namespace

void InverseTransformInit10bpp_NEON() { Init10bpp(); }

}  // namespace dsp
}  // namespace libgav1
#else   // !LIBGAV1_ENABLE_NEON || LIBGAV1_MAX_BITDEPTH < 10
namespace libgav1 {
namespace dsp {

void InverseTransformInit10bpp_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10
