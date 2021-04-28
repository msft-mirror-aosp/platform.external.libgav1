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

#include "src/dsp/loop_restoration.h"
#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10
#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/dsp/arm/common_neon.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/compiler_attributes.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

//------------------------------------------------------------------------------
// Wiener

// Must make a local copy of coefficients to help compiler know that they have
// no overlap with other buffers. Using 'const' keyword is not enough. Actually
// compiler doesn't make a copy, since there is enough registers in this case.
inline void PopulateWienerCoefficients(
    const RestorationUnitInfo& restoration_info, const int direction,
    int16_t filter[4]) {
  for (int i = 0; i < 4; ++i) {
    filter[i] = restoration_info.wiener_info.filter[direction][i];
  }
}

inline int32x4x2_t WienerHorizontal2(const uint16x8_t s0, const uint16x8_t s1,
                                     const int16_t filter,
                                     const int32x4x2_t sum) {
  const int16x8_t ss = vreinterpretq_s16_u16(vaddq_u16(s0, s1));
  int32x4x2_t res;
  res.val[0] = vmlal_n_s16(sum.val[0], vget_low_s16(ss), filter);
  res.val[1] = vmlal_n_s16(sum.val[1], vget_high_s16(ss), filter);
  return res;
}

inline void WienerHorizontalSum(const uint16x8_t s[3], const int16_t filter[4],
                                int32x4x2_t sum, int16_t* const wiener_buffer) {
  constexpr int offset =
      1 << (kBitdepth10 + kWienerFilterBits - kInterRoundBitsHorizontal - 1);
  constexpr int limit = (offset << 2) - 1;
  const int16x8_t s_0_2 = vreinterpretq_s16_u16(vaddq_u16(s[0], s[2]));
  const int16x8_t s_1 = vreinterpretq_s16_u16(s[1]);
  int16x4x2_t sum16;
  sum.val[0] = vmlal_n_s16(sum.val[0], vget_low_s16(s_0_2), filter[2]);
  sum.val[0] = vmlal_n_s16(sum.val[0], vget_low_s16(s_1), filter[3]);
  sum16.val[0] = vqshrn_n_s32(sum.val[0], kInterRoundBitsHorizontal);
  sum16.val[0] = vmax_s16(sum16.val[0], vdup_n_s16(-offset));
  sum16.val[0] = vmin_s16(sum16.val[0], vdup_n_s16(limit - offset));
  vst1_s16(wiener_buffer, sum16.val[0]);
  sum.val[1] = vmlal_n_s16(sum.val[1], vget_high_s16(s_0_2), filter[2]);
  sum.val[1] = vmlal_n_s16(sum.val[1], vget_high_s16(s_1), filter[3]);
  sum16.val[1] = vqshrn_n_s32(sum.val[1], kInterRoundBitsHorizontal);
  sum16.val[1] = vmax_s16(sum16.val[1], vdup_n_s16(-offset));
  sum16.val[1] = vmin_s16(sum16.val[1], vdup_n_s16(limit - offset));
  vst1_s16(wiener_buffer + 4, sum16.val[1]);
}

inline void WienerHorizontalTap7(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const int16_t filter[4],
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    const uint16_t* src_ptr = src;
    uint16x8_t s[8];
    s[0] = vld1q_u16(src_ptr);
    ptrdiff_t x = width;
    do {
      src_ptr += 8;
      s[7] = vld1q_u16(src_ptr);
      s[1] = vextq_u16(s[0], s[7], 1);
      s[2] = vextq_u16(s[0], s[7], 2);
      s[3] = vextq_u16(s[0], s[7], 3);
      s[4] = vextq_u16(s[0], s[7], 4);
      s[5] = vextq_u16(s[0], s[7], 5);
      s[6] = vextq_u16(s[0], s[7], 6);

      int32x4x2_t sum;
      sum.val[0] = sum.val[1] =
          vdupq_n_s32(1 << (kInterRoundBitsHorizontal - 1));
      sum = WienerHorizontal2(s[0], s[6], filter[0], sum);
      sum = WienerHorizontal2(s[1], s[5], filter[1], sum);
      WienerHorizontalSum(s + 2, filter, sum, *wiener_buffer);
      s[0] = s[7];
      *wiener_buffer += 8;
      x -= 8;
    } while (x != 0);
    src += src_stride;
  }
}

inline void WienerHorizontalTap5(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const int16_t filter[4],
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    const uint16_t* src_ptr = src;
    uint16x8_t s[6];
    s[0] = vld1q_u16(src_ptr);
    ptrdiff_t x = width;
    do {
      src_ptr += 8;
      s[5] = vld1q_u16(src_ptr);
      s[1] = vextq_u16(s[0], s[5], 1);
      s[2] = vextq_u16(s[0], s[5], 2);
      s[3] = vextq_u16(s[0], s[5], 3);
      s[4] = vextq_u16(s[0], s[5], 4);

      int32x4x2_t sum;
      sum.val[0] = sum.val[1] =
          vdupq_n_s32(1 << (kInterRoundBitsHorizontal - 1));
      sum = WienerHorizontal2(s[0], s[4], filter[1], sum);
      WienerHorizontalSum(s + 1, filter, sum, *wiener_buffer);
      s[0] = s[5];
      *wiener_buffer += 8;
      x -= 8;
    } while (x != 0);
    src += src_stride;
  }
}

inline void WienerHorizontalTap3(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const int16_t filter[4],
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    const uint16_t* src_ptr = src;
    uint16x8_t s[3];
    ptrdiff_t x = width;
    do {
      s[0] = vld1q_u16(src_ptr);
      s[1] = vld1q_u16(src_ptr + 1);
      s[2] = vld1q_u16(src_ptr + 2);

      int32x4x2_t sum;
      sum.val[0] = sum.val[1] =
          vdupq_n_s32(1 << (kInterRoundBitsHorizontal - 1));
      WienerHorizontalSum(s, filter, sum, *wiener_buffer);
      src_ptr += 8;
      *wiener_buffer += 8;
      x -= 8;
    } while (x != 0);
    src += src_stride;
  }
}

inline void WienerHorizontalTap1(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      const uint16x8_t s = vld1q_u16(src + x);
      const int16x8_t d = vreinterpretq_s16_u16(vshlq_n_u16(s, 4));
      vst1q_s16(*wiener_buffer + x, d);
      x += 8;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline int32x4x2_t WienerVertical2(const int16x8_t a0, const int16x8_t a1,
                                   const int16_t filter,
                                   const int32x4x2_t sum) {
  int32x4x2_t d;
  d.val[0] = vmlal_n_s16(sum.val[0], vget_low_s16(a0), filter);
  d.val[1] = vmlal_n_s16(sum.val[1], vget_high_s16(a0), filter);
  d.val[0] = vmlal_n_s16(d.val[0], vget_low_s16(a1), filter);
  d.val[1] = vmlal_n_s16(d.val[1], vget_high_s16(a1), filter);
  return d;
}

inline uint16x8_t WienerVertical(const int16x8_t a[3], const int16_t filter[4],
                                 const int32x4x2_t sum) {
  int32x4x2_t d = WienerVertical2(a[0], a[2], filter[2], sum);
  d.val[0] = vmlal_n_s16(d.val[0], vget_low_s16(a[1]), filter[3]);
  d.val[1] = vmlal_n_s16(d.val[1], vget_high_s16(a[1]), filter[3]);
  const uint16x4_t sum_lo_16 = vqrshrun_n_s32(d.val[0], 11);
  const uint16x4_t sum_hi_16 = vqrshrun_n_s32(d.val[1], 11);
  return vcombine_u16(sum_lo_16, sum_hi_16);
}

inline uint16x8_t WienerVerticalTap7Kernel(const int16_t* const wiener_buffer,
                                           const ptrdiff_t wiener_stride,
                                           const int16_t filter[4],
                                           int16x8_t a[7]) {
  int32x4x2_t sum;
  a[0] = vld1q_s16(wiener_buffer + 0 * wiener_stride);
  a[1] = vld1q_s16(wiener_buffer + 1 * wiener_stride);
  a[5] = vld1q_s16(wiener_buffer + 5 * wiener_stride);
  a[6] = vld1q_s16(wiener_buffer + 6 * wiener_stride);
  sum.val[0] = sum.val[1] = vdupq_n_s32(0);
  sum = WienerVertical2(a[0], a[6], filter[0], sum);
  sum = WienerVertical2(a[1], a[5], filter[1], sum);
  a[2] = vld1q_s16(wiener_buffer + 2 * wiener_stride);
  a[3] = vld1q_s16(wiener_buffer + 3 * wiener_stride);
  a[4] = vld1q_s16(wiener_buffer + 4 * wiener_stride);
  return WienerVertical(a + 2, filter, sum);
}

inline uint16x8x2_t WienerVerticalTap7Kernel2(
    const int16_t* const wiener_buffer, const ptrdiff_t wiener_stride,
    const int16_t filter[4]) {
  int16x8_t a[8];
  int32x4x2_t sum;
  uint16x8x2_t d;
  d.val[0] = WienerVerticalTap7Kernel(wiener_buffer, wiener_stride, filter, a);
  a[7] = vld1q_s16(wiener_buffer + 7 * wiener_stride);
  sum.val[0] = sum.val[1] = vdupq_n_s32(0);
  sum = WienerVertical2(a[1], a[7], filter[0], sum);
  sum = WienerVertical2(a[2], a[6], filter[1], sum);
  d.val[1] = WienerVertical(a + 3, filter, sum);
  return d;
}

inline void WienerVerticalTap7(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t filter[4], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const uint16x8_t v_max_bitdepth = vdupq_n_u16((1 << kBitdepth10) - 1);
  for (int y = height >> 1; y != 0; --y) {
    uint16_t* dst_ptr = dst;
    ptrdiff_t x = width;
    do {
      uint16x8x2_t d[2];
      d[0] = WienerVerticalTap7Kernel2(wiener_buffer + 0, width, filter);
      d[1] = WienerVerticalTap7Kernel2(wiener_buffer + 8, width, filter);
      vst1q_u16(dst_ptr, vminq_u16(d[0].val[0], v_max_bitdepth));
      vst1q_u16(dst_ptr + 8, vminq_u16(d[1].val[0], v_max_bitdepth));
      vst1q_u16(dst_ptr + dst_stride, vminq_u16(d[0].val[1], v_max_bitdepth));
      vst1q_u16(dst_ptr + 8 + dst_stride,
                vminq_u16(d[1].val[1], v_max_bitdepth));
      wiener_buffer += 16;
      dst_ptr += 16;
      x -= 16;
    } while (x != 0);
    wiener_buffer += width;
    dst += 2 * dst_stride;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = width;
    do {
      int16x8_t a[7];
      const uint16x8_t d0 =
          WienerVerticalTap7Kernel(wiener_buffer + 0, width, filter, a);
      const uint16x8_t d1 =
          WienerVerticalTap7Kernel(wiener_buffer + 8, width, filter, a);
      vst1q_u16(dst, vminq_u16(d0, v_max_bitdepth));
      vst1q_u16(dst + 8, vminq_u16(d1, v_max_bitdepth));
      wiener_buffer += 16;
      dst += 16;
      x -= 16;
    } while (x != 0);
  }
}

inline uint16x8_t WienerVerticalTap5Kernel(const int16_t* const wiener_buffer,
                                           const ptrdiff_t wiener_stride,
                                           const int16_t filter[4],
                                           int16x8_t a[5]) {
  a[0] = vld1q_s16(wiener_buffer + 0 * wiener_stride);
  a[1] = vld1q_s16(wiener_buffer + 1 * wiener_stride);
  a[2] = vld1q_s16(wiener_buffer + 2 * wiener_stride);
  a[3] = vld1q_s16(wiener_buffer + 3 * wiener_stride);
  a[4] = vld1q_s16(wiener_buffer + 4 * wiener_stride);
  int32x4x2_t sum;
  sum.val[0] = sum.val[1] = vdupq_n_s32(0);
  sum = WienerVertical2(a[0], a[4], filter[1], sum);
  return WienerVertical(a + 1, filter, sum);
}

inline uint16x8x2_t WienerVerticalTap5Kernel2(
    const int16_t* const wiener_buffer, const ptrdiff_t wiener_stride,
    const int16_t filter[4]) {
  int16x8_t a[6];
  int32x4x2_t sum;
  uint16x8x2_t d;
  d.val[0] = WienerVerticalTap5Kernel(wiener_buffer, wiener_stride, filter, a);
  a[5] = vld1q_s16(wiener_buffer + 5 * wiener_stride);
  sum.val[0] = sum.val[1] = vdupq_n_s32(0);
  sum = WienerVertical2(a[1], a[5], filter[1], sum);
  d.val[1] = WienerVertical(a + 2, filter, sum);
  return d;
}

inline void WienerVerticalTap5(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t filter[4], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const uint16x8_t v_max_bitdepth = vdupq_n_u16((1 << kBitdepth10) - 1);
  for (int y = height >> 1; y != 0; --y) {
    uint16_t* dst_ptr = dst;
    ptrdiff_t x = width;
    do {
      uint16x8x2_t d[2];
      d[0] = WienerVerticalTap5Kernel2(wiener_buffer + 0, width, filter);
      d[1] = WienerVerticalTap5Kernel2(wiener_buffer + 8, width, filter);
      vst1q_u16(dst_ptr, vminq_u16(d[0].val[0], v_max_bitdepth));
      vst1q_u16(dst_ptr + 8, vminq_u16(d[1].val[0], v_max_bitdepth));
      vst1q_u16(dst_ptr + dst_stride, vminq_u16(d[0].val[1], v_max_bitdepth));
      vst1q_u16(dst_ptr + 8 + dst_stride,
                vminq_u16(d[1].val[1], v_max_bitdepth));
      wiener_buffer += 16;
      dst_ptr += 16;
      x -= 16;
    } while (x != 0);
    wiener_buffer += width;
    dst += 2 * dst_stride;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = width;
    do {
      int16x8_t a[5];
      const uint16x8_t d0 =
          WienerVerticalTap5Kernel(wiener_buffer + 0, width, filter, a);
      const uint16x8_t d1 =
          WienerVerticalTap5Kernel(wiener_buffer + 8, width, filter, a);
      vst1q_u16(dst, vminq_u16(d0, v_max_bitdepth));
      vst1q_u16(dst + 8, vminq_u16(d1, v_max_bitdepth));
      wiener_buffer += 16;
      dst += 16;
      x -= 16;
    } while (x != 0);
  }
}

inline uint16x8_t WienerVerticalTap3Kernel(const int16_t* const wiener_buffer,
                                           const ptrdiff_t wiener_stride,
                                           const int16_t filter[4],
                                           int16x8_t a[3]) {
  a[0] = vld1q_s16(wiener_buffer + 0 * wiener_stride);
  a[1] = vld1q_s16(wiener_buffer + 1 * wiener_stride);
  a[2] = vld1q_s16(wiener_buffer + 2 * wiener_stride);
  int32x4x2_t sum;
  sum.val[0] = sum.val[1] = vdupq_n_s32(0);
  return WienerVertical(a, filter, sum);
}

inline uint16x8x2_t WienerVerticalTap3Kernel2(
    const int16_t* const wiener_buffer, const ptrdiff_t wiener_stride,
    const int16_t filter[4]) {
  int16x8_t a[4];
  int32x4x2_t sum;
  uint16x8x2_t d;
  d.val[0] = WienerVerticalTap3Kernel(wiener_buffer, wiener_stride, filter, a);
  a[3] = vld1q_s16(wiener_buffer + 3 * wiener_stride);
  sum.val[0] = sum.val[1] = vdupq_n_s32(0);
  d.val[1] = WienerVertical(a + 1, filter, sum);
  return d;
}

inline void WienerVerticalTap3(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t filter[4], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const uint16x8_t v_max_bitdepth = vdupq_n_u16((1 << kBitdepth10) - 1);

  for (int y = height >> 1; y != 0; --y) {
    uint16_t* dst_ptr = dst;
    ptrdiff_t x = width;
    do {
      uint16x8x2_t d[2];
      d[0] = WienerVerticalTap3Kernel2(wiener_buffer + 0, width, filter);
      d[1] = WienerVerticalTap3Kernel2(wiener_buffer + 8, width, filter);

      vst1q_u16(dst_ptr, vminq_u16(d[0].val[0], v_max_bitdepth));
      vst1q_u16(dst_ptr + 8, vminq_u16(d[1].val[0], v_max_bitdepth));
      vst1q_u16(dst_ptr + dst_stride, vminq_u16(d[0].val[1], v_max_bitdepth));
      vst1q_u16(dst_ptr + 8 + dst_stride,
                vminq_u16(d[1].val[1], v_max_bitdepth));

      wiener_buffer += 16;
      dst_ptr += 16;
      x -= 16;
    } while (x != 0);
    wiener_buffer += width;
    dst += 2 * dst_stride;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = width;
    do {
      int16x8_t a[3];
      const uint16x8_t d0 =
          WienerVerticalTap3Kernel(wiener_buffer + 0, width, filter, a);
      const uint16x8_t d1 =
          WienerVerticalTap3Kernel(wiener_buffer + 8, width, filter, a);
      vst1q_u16(dst, vminq_u16(d0, v_max_bitdepth));
      vst1q_u16(dst + 8, vminq_u16(d1, v_max_bitdepth));
      wiener_buffer += 16;
      dst += 16;
      x -= 16;
    } while (x != 0);
  }
}

inline void WienerVerticalTap1Kernel(const int16_t* const wiener_buffer,
                                     uint16_t* const dst) {
  const uint16x8_t v_max_bitdepth = vdupq_n_u16((1 << kBitdepth10) - 1);
  const int16x8_t a0 = vld1q_s16(wiener_buffer + 0);
  const int16x8_t a1 = vld1q_s16(wiener_buffer + 8);
  const int16x8_t d0 = vrshrq_n_s16(a0, 4);
  const int16x8_t d1 = vrshrq_n_s16(a1, 4);
  vst1q_u16(dst, vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(d0, vdupq_n_s16(0))),
                           v_max_bitdepth));
  vst1q_u16(dst + 8,
            vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(d1, vdupq_n_s16(0))),
                      v_max_bitdepth));
}

inline void WienerVerticalTap1(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               uint16_t* dst, const ptrdiff_t dst_stride) {
  for (int y = height >> 1; y != 0; --y) {
    uint16_t* dst_ptr = dst;
    ptrdiff_t x = width;
    do {
      WienerVerticalTap1Kernel(wiener_buffer, dst_ptr);
      WienerVerticalTap1Kernel(wiener_buffer + width, dst_ptr + dst_stride);
      wiener_buffer += 16;
      dst_ptr += 16;
      x -= 16;
    } while (x != 0);
    wiener_buffer += width;
    dst += 2 * dst_stride;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = width;
    do {
      WienerVerticalTap1Kernel(wiener_buffer, dst);
      wiener_buffer += 16;
      dst += 16;
      x -= 16;
    } while (x != 0);
  }
}

// For width 16 and up, store the horizontal results, and then do the vertical
// filter row by row. This is faster than doing it column by column when
// considering cache issues.
void WienerFilter_NEON(
    const RestorationUnitInfo& restoration_info, const void* const source,
    const ptrdiff_t stride, const void* const top_border,
    const ptrdiff_t top_border_stride, const void* const bottom_border,
    const ptrdiff_t bottom_border_stride, const int width, const int height,
    RestorationBuffer* const restoration_buffer, void* const dest) {
  const int16_t* const number_leading_zero_coefficients =
      restoration_info.wiener_info.number_leading_zero_coefficients;
  const int number_rows_to_skip = std::max(
      static_cast<int>(number_leading_zero_coefficients[WienerInfo::kVertical]),
      1);
  const ptrdiff_t wiener_stride = Align(width, 16);
  int16_t* const wiener_buffer_vertical = restoration_buffer->wiener_buffer;
  // The values are saturated to 13 bits before storing.
  int16_t* wiener_buffer_horizontal =
      wiener_buffer_vertical + number_rows_to_skip * wiener_stride;
  int16_t filter_horizontal[(kWienerFilterTaps + 1) / 2];
  int16_t filter_vertical[(kWienerFilterTaps + 1) / 2];
  PopulateWienerCoefficients(restoration_info, WienerInfo::kHorizontal,
                             filter_horizontal);
  PopulateWienerCoefficients(restoration_info, WienerInfo::kVertical,
                             filter_vertical);

  // horizontal filtering.
  const int height_horizontal =
      height + kWienerFilterTaps - 1 - 2 * number_rows_to_skip;
  const int height_extra = (height_horizontal - height) >> 1;
  assert(height_extra <= 2);
  const auto* const src = static_cast<const uint16_t*>(source);
  const auto* const top = static_cast<const uint16_t*>(top_border);
  const auto* const bottom = static_cast<const uint16_t*>(bottom_border);
  if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 0) {
    WienerHorizontalTap7(top + (2 - height_extra) * top_border_stride - 3,
                         top_border_stride, wiener_stride, height_extra,
                         filter_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap7(src - 3, stride, wiener_stride, height,
                         filter_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap7(bottom - 3, bottom_border_stride, wiener_stride,
                         height_extra, filter_horizontal,
                         &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 1) {
    WienerHorizontalTap5(top + (2 - height_extra) * top_border_stride - 2,
                         top_border_stride, wiener_stride, height_extra,
                         filter_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap5(src - 2, stride, wiener_stride, height,
                         filter_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap5(bottom - 2, bottom_border_stride, wiener_stride,
                         height_extra, filter_horizontal,
                         &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 2) {
    WienerHorizontalTap3(top + (2 - height_extra) * top_border_stride - 1,
                         top_border_stride, wiener_stride, height_extra,
                         filter_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap3(src - 1, stride, wiener_stride, height,
                         filter_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap3(bottom - 1, bottom_border_stride, wiener_stride,
                         height_extra, filter_horizontal,
                         &wiener_buffer_horizontal);
  } else {
    assert(number_leading_zero_coefficients[WienerInfo::kHorizontal] == 3);
    WienerHorizontalTap1(top + (2 - height_extra) * top_border_stride,
                         top_border_stride, wiener_stride, height_extra,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap1(src, stride, wiener_stride, height,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap1(bottom, bottom_border_stride, wiener_stride,
                         height_extra, &wiener_buffer_horizontal);
  }

  // vertical filtering.
  auto* dst = static_cast<uint16_t*>(dest);
  if (number_leading_zero_coefficients[WienerInfo::kVertical] == 0) {
    // Because the top row of |source| is a duplicate of the second row, and the
    // bottom row of |source| is a duplicate of its above row, we can duplicate
    // the top and bottom row of |wiener_buffer| accordingly.
    memcpy(wiener_buffer_horizontal, wiener_buffer_horizontal - wiener_stride,
           sizeof(*wiener_buffer_horizontal) * wiener_stride);
    memcpy(restoration_buffer->wiener_buffer,
           restoration_buffer->wiener_buffer + wiener_stride,
           sizeof(*restoration_buffer->wiener_buffer) * wiener_stride);
    WienerVerticalTap7(wiener_buffer_vertical, wiener_stride, height,
                       filter_vertical, dst, stride);
  } else if (number_leading_zero_coefficients[WienerInfo::kVertical] == 1) {
    WienerVerticalTap5(wiener_buffer_vertical + wiener_stride, wiener_stride,
                       height, filter_vertical, dst, stride);
  } else if (number_leading_zero_coefficients[WienerInfo::kVertical] == 2) {
    WienerVerticalTap3(wiener_buffer_vertical + 2 * wiener_stride,
                       wiener_stride, height, filter_vertical, dst, stride);
  } else {
    assert(number_leading_zero_coefficients[WienerInfo::kVertical] == 3);
    WienerVerticalTap1(wiener_buffer_vertical + 3 * wiener_stride,
                       wiener_stride, height, dst, stride);
  }
}

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth10);
  assert(dsp != nullptr);
  dsp->loop_restorations[0] = WienerFilter_NEON;
}

}  // namespace

void LoopRestorationInit10bpp_NEON() { Init10bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !(LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10)
namespace libgav1 {
namespace dsp {

void LoopRestorationInit10bpp_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10
