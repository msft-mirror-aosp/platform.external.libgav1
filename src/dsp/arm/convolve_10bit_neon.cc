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

#include "src/dsp/convolve.h"
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

// Include the constants and utility functions inside the anonymous namespace.
#include "src/dsp/convolve.inc"

template <int filter_index>
int32x4x2_t SumOnePassTaps(const uint16x8_t* const src,
                           const int16x4_t* const taps) {
  const auto* ssrc = reinterpret_cast<const int16x8_t*>(src);
  int32x4x2_t sum;
  if (filter_index < 2) {
    // 6 taps.
    sum.val[0] = vmull_s16(vget_low_s16(ssrc[0]), taps[0]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[1]), taps[1]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[2]), taps[2]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[3]), taps[3]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[4]), taps[4]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[5]), taps[5]);

    sum.val[1] = vmull_s16(vget_high_s16(ssrc[0]), taps[0]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[1]), taps[1]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[2]), taps[2]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[3]), taps[3]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[4]), taps[4]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[5]), taps[5]);
  } else if (filter_index == 2) {
    // 8 taps.
    sum.val[0] = vmull_s16(vget_low_s16(ssrc[0]), taps[0]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[1]), taps[1]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[2]), taps[2]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[3]), taps[3]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[4]), taps[4]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[5]), taps[5]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[6]), taps[6]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[7]), taps[7]);

    sum.val[1] = vmull_s16(vget_high_s16(ssrc[0]), taps[0]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[1]), taps[1]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[2]), taps[2]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[3]), taps[3]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[4]), taps[4]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[5]), taps[5]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[6]), taps[6]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[7]), taps[7]);
  } else if (filter_index == 3) {
    // 2 taps.
    sum.val[0] = vmull_s16(vget_low_s16(ssrc[0]), taps[0]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[1]), taps[1]);

    sum.val[1] = vmull_s16(vget_high_s16(ssrc[0]), taps[0]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[1]), taps[1]);
  } else {
    // 4 taps.
    sum.val[0] = vmull_s16(vget_low_s16(ssrc[0]), taps[0]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[1]), taps[1]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[2]), taps[2]);
    sum.val[0] = vmlal_s16(sum.val[0], vget_low_s16(ssrc[3]), taps[3]);

    sum.val[1] = vmull_s16(vget_high_s16(ssrc[0]), taps[0]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[1]), taps[1]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[2]), taps[2]);
    sum.val[1] = vmlal_s16(sum.val[1], vget_high_s16(ssrc[3]), taps[3]);
  }
  return sum;
}

template <int filter_index>
int32x4_t SumOnePassTaps(const uint16x4_t* const src,
                         const int16x4_t* const taps) {
  const auto* ssrc = reinterpret_cast<const int16x4_t*>(src);
  int32x4_t sum;
  if (filter_index < 2) {
    // 6 taps.
    sum = vmull_s16(ssrc[0], taps[0]);
    sum = vmlal_s16(sum, ssrc[1], taps[1]);
    sum = vmlal_s16(sum, ssrc[2], taps[2]);
    sum = vmlal_s16(sum, ssrc[3], taps[3]);
    sum = vmlal_s16(sum, ssrc[4], taps[4]);
    sum = vmlal_s16(sum, ssrc[5], taps[5]);
  } else if (filter_index == 2) {
    // 8 taps.
    sum = vmull_s16(ssrc[0], taps[0]);
    sum = vmlal_s16(sum, ssrc[1], taps[1]);
    sum = vmlal_s16(sum, ssrc[2], taps[2]);
    sum = vmlal_s16(sum, ssrc[3], taps[3]);
    sum = vmlal_s16(sum, ssrc[4], taps[4]);
    sum = vmlal_s16(sum, ssrc[5], taps[5]);
    sum = vmlal_s16(sum, ssrc[6], taps[6]);
    sum = vmlal_s16(sum, ssrc[7], taps[7]);
  } else if (filter_index == 3) {
    // 2 taps.
    sum = vmull_s16(ssrc[0], taps[0]);
    sum = vmlal_s16(sum, ssrc[1], taps[1]);
  } else {
    // 4 taps.
    sum = vmull_s16(ssrc[0], taps[0]);
    sum = vmlal_s16(sum, ssrc[1], taps[1]);
    sum = vmlal_s16(sum, ssrc[2], taps[2]);
    sum = vmlal_s16(sum, ssrc[3], taps[3]);
  }
  return sum;
}

template <int filter_index, bool is_compound>
void FilterHorizontalWidth8AndUp(const uint16_t* LIBGAV1_RESTRICT src,
                                 const ptrdiff_t src_stride,
                                 void* LIBGAV1_RESTRICT const dest,
                                 const ptrdiff_t pred_stride, const int width,
                                 const int height,
                                 const int16x4_t* const v_tap) {
  auto* dest16 = static_cast<uint16_t*>(dest);
  const uint16x4_t v_max_bitdepth = vdup_n_u16((1 << kBitdepth10) - 1);
  int y = height;
  do {
    int x = 0;
    do {
      const uint16x8_t src_long = vld1q_u16(src + x);
      const uint16x8_t src_long_hi = vld1q_u16(src + x + 8);
      uint16x8_t v_src[8];
      int32x4x2_t v_sum;
      if (filter_index < 2) {
        v_src[0] = src_long;
        v_src[1] = vextq_u16(src_long, src_long_hi, 1);
        v_src[2] = vextq_u16(src_long, src_long_hi, 2);
        v_src[3] = vextq_u16(src_long, src_long_hi, 3);
        v_src[4] = vextq_u16(src_long, src_long_hi, 4);
        v_src[5] = vextq_u16(src_long, src_long_hi, 5);
        v_sum = SumOnePassTaps<filter_index>(v_src, v_tap + 1);
      } else if (filter_index == 2) {
        v_src[0] = src_long;
        v_src[1] = vextq_u16(src_long, src_long_hi, 1);
        v_src[2] = vextq_u16(src_long, src_long_hi, 2);
        v_src[3] = vextq_u16(src_long, src_long_hi, 3);
        v_src[4] = vextq_u16(src_long, src_long_hi, 4);
        v_src[5] = vextq_u16(src_long, src_long_hi, 5);
        v_src[6] = vextq_u16(src_long, src_long_hi, 6);
        v_src[7] = vextq_u16(src_long, src_long_hi, 7);
        v_sum = SumOnePassTaps<filter_index>(v_src, v_tap);
      } else if (filter_index == 3) {
        v_src[0] = src_long;
        v_src[1] = vextq_u16(src_long, src_long_hi, 1);
        v_sum = SumOnePassTaps<filter_index>(v_src, v_tap + 3);
      } else if (filter_index > 3) {
        v_src[0] = src_long;
        v_src[1] = vextq_u16(src_long, src_long_hi, 1);
        v_src[2] = vextq_u16(src_long, src_long_hi, 2);
        v_src[3] = vextq_u16(src_long, src_long_hi, 3);
        v_sum = SumOnePassTaps<filter_index>(v_src, v_tap + 2);
      }
      if (is_compound) {
        const int16x4_t v_compound_offset = vdup_n_s16(kCompoundOffset);
        const int16x4_t d0 =
            vqrshrn_n_s32(v_sum.val[0], kInterRoundBitsHorizontal - 1);
        const int16x4_t d1 =
            vqrshrn_n_s32(v_sum.val[1], kInterRoundBitsHorizontal - 1);
        vst1_u16(&dest16[x],
                 vreinterpret_u16_s16(vadd_s16(d0, v_compound_offset)));
        vst1_u16(&dest16[x + 4],
                 vreinterpret_u16_s16(vadd_s16(d1, v_compound_offset)));
      } else {
        // Normally the Horizontal pass does the downshift in two passes:
        // kInterRoundBitsHorizontal - 1 and then (kFilterBits -
        // kInterRoundBitsHorizontal). Each one uses a rounding shift.
        // Combining them requires adding the rounding offset from the skipped
        // shift.
        const int32x4_t v_first_shift_rounding_bit =
            vdupq_n_s32(1 << (kInterRoundBitsHorizontal - 2));
        v_sum.val[0] = vaddq_s32(v_sum.val[0], v_first_shift_rounding_bit);
        v_sum.val[1] = vaddq_s32(v_sum.val[1], v_first_shift_rounding_bit);
        const uint16x4_t d0 = vmin_u16(
            vqrshrun_n_s32(v_sum.val[0], kFilterBits - 1), v_max_bitdepth);
        const uint16x4_t d1 = vmin_u16(
            vqrshrun_n_s32(v_sum.val[1], kFilterBits - 1), v_max_bitdepth);
        vst1_u16(&dest16[x], d0);
        vst1_u16(&dest16[x + 4], d1);
      }
      x += 8;
    } while (x < width);
    src += src_stride >> 1;
    dest16 += is_compound ? pred_stride : pred_stride >> 1;
  } while (--y != 0);
}

template <int filter_index, bool is_compound>
void FilterHorizontalWidth4(const uint16_t* LIBGAV1_RESTRICT src,
                            const ptrdiff_t src_stride,
                            void* LIBGAV1_RESTRICT const dest,
                            const ptrdiff_t pred_stride, const int height,
                            const int16x4_t* const v_tap) {
  auto* dest16 = static_cast<uint16_t*>(dest);
  const uint16x4_t v_max_bitdepth = vdup_n_u16((1 << kBitdepth10) - 1);
  int y = height;
  do {
    const uint16x8_t v_zero = vdupq_n_u16(0);
    uint16x4_t v_src[4];
    int32x4_t v_sum;
    const uint16x8_t src_long = vld1q_u16(src);
    v_src[0] = vget_low_u16(src_long);
    if (filter_index == 3) {
      v_src[1] = vget_low_u16(vextq_u16(src_long, v_zero, 1));
      v_sum = SumOnePassTaps<filter_index>(v_src, v_tap + 3);
    } else {
      v_src[1] = vget_low_u16(vextq_u16(src_long, v_zero, 1));
      v_src[2] = vget_low_u16(vextq_u16(src_long, v_zero, 2));
      v_src[3] = vget_low_u16(vextq_u16(src_long, v_zero, 3));
      v_sum = SumOnePassTaps<filter_index>(v_src, v_tap + 2);
    }
    if (is_compound) {
      const int16x4_t d0 = vqrshrn_n_s32(v_sum, kInterRoundBitsHorizontal - 1);
      vst1_u16(&dest16[0],
               vreinterpret_u16_s16(vadd_s16(d0, vdup_n_s16(kCompoundOffset))));
    } else {
      const int32x4_t v_first_shift_rounding_bit =
          vdupq_n_s32(1 << (kInterRoundBitsHorizontal - 2));
      v_sum = vaddq_s32(v_sum, v_first_shift_rounding_bit);
      const uint16x4_t d0 =
          vmin_u16(vqrshrun_n_s32(v_sum, kFilterBits - 1), v_max_bitdepth);
      vst1_u16(&dest16[0], d0);
    }
    src += src_stride >> 1;
    dest16 += is_compound ? pred_stride : pred_stride >> 1;
  } while (--y != 0);
}

template <int filter_index>
void FilterHorizontalWidth2(const uint16_t* LIBGAV1_RESTRICT src,
                            const ptrdiff_t src_stride,
                            void* LIBGAV1_RESTRICT const dest,
                            const ptrdiff_t pred_stride, const int height,
                            const int16x4_t* const v_tap) {
  auto* dest16 = static_cast<uint16_t*>(dest);
  const uint16x4_t v_max_bitdepth = vdup_n_u16((1 << kBitdepth10) - 1);
  int y = height >> 1;
  do {
    const int16x8_t v_zero = vdupq_n_s16(0);
    const int16x8_t input0 = vreinterpretq_s16_u16(vld1q_u16(src));
    const int16x8_t input1 =
        vreinterpretq_s16_u16(vld1q_u16(src + (src_stride >> 1)));
    const int16x8x2_t input = vzipq_s16(input0, input1);
    int32x4_t v_sum;
    if (filter_index == 3) {
      v_sum = vmull_s16(vget_low_s16(input.val[0]), v_tap[3]);
      v_sum = vmlal_s16(v_sum,
                        vget_low_s16(vextq_s16(input.val[0], input.val[1], 2)),
                        v_tap[4]);
    } else {
      v_sum = vmull_s16(vget_low_s16(input.val[0]), v_tap[2]);
      v_sum = vmlal_s16(v_sum, vget_low_s16(vextq_s16(input.val[0], v_zero, 2)),
                        v_tap[3]);
      v_sum = vmlal_s16(v_sum, vget_low_s16(vextq_s16(input.val[0], v_zero, 4)),
                        v_tap[4]);
      v_sum = vmlal_s16(v_sum,
                        vget_low_s16(vextq_s16(input.val[0], input.val[1], 6)),
                        v_tap[5]);
    }
    // Normally the Horizontal pass does the downshift in two passes:
    // kInterRoundBitsHorizontal - 1 and then (kFilterBits -
    // kInterRoundBitsHorizontal). Each one uses a rounding shift.
    // Combining them requires adding the rounding offset from the skipped
    // shift.
    const int32x4_t v_first_shift_rounding_bit =
        vdupq_n_s32(1 << (kInterRoundBitsHorizontal - 2));
    v_sum = vaddq_s32(v_sum, v_first_shift_rounding_bit);
    const uint16x4_t d0 =
        vmin_u16(vqrshrun_n_s32(v_sum, kFilterBits - 1), v_max_bitdepth);
    dest16[0] = vget_lane_u16(d0, 0);
    dest16[1] = vget_lane_u16(d0, 2);
    dest16 += pred_stride >> 1;
    dest16[0] = vget_lane_u16(d0, 1);
    dest16[1] = vget_lane_u16(d0, 3);
    dest16 += pred_stride >> 1;
    src += src_stride;
  } while (--y != 0);
}

template <int filter_index, bool is_compound>
void FilterHorizontal(const uint16_t* LIBGAV1_RESTRICT const src,
                      const ptrdiff_t src_stride,
                      void* LIBGAV1_RESTRICT const dest,
                      const ptrdiff_t pred_stride, const int width,
                      const int height, const int16x4_t* const v_tap) {
  assert(width < 8 || filter_index <= 3);
  // Don't simplify the redundant if conditions with the template parameters,
  // which helps the compiler generate compact code.
  if (width >= 8 && filter_index <= 3) {
    FilterHorizontalWidth8AndUp<filter_index, is_compound>(
        src, src_stride, dest, pred_stride, width, height, v_tap);
    return;
  }

  // Horizontal passes only needs to account for number of taps 2 and 4 when
  // |width| <= 4.
  assert(width <= 4);
  assert(filter_index >= 3 && filter_index <= 5);
  if (filter_index >= 3 && filter_index <= 5) {
    if (width == 4) {
      FilterHorizontalWidth4<filter_index, is_compound>(
          src, src_stride, dest, pred_stride, height, v_tap);
      return;
    }
    assert(width == 2);
    if (!is_compound) {
      FilterHorizontalWidth2<filter_index>(src, src_stride, dest, pred_stride,
                                           height, v_tap);
    }
  }
}

template <bool is_compound = false>
LIBGAV1_ALWAYS_INLINE void DoHorizontalPass(
    const uint16_t* LIBGAV1_RESTRICT const src, const ptrdiff_t src_stride,
    void* LIBGAV1_RESTRICT const dst, const ptrdiff_t dst_stride,
    const int width, const int height, const int filter_id,
    const int filter_index) {
  // Duplicate the absolute value for each tap.  Negative taps are corrected
  // by using the vmlsl_u8 instruction.  Positive taps use vmlal_u8.
  int16x4_t v_tap[kSubPixelTaps];
  assert(filter_id != 0);

  for (int k = 0; k < kSubPixelTaps; ++k) {
    v_tap[k] = vdup_n_s16(kHalfSubPixelFilters[filter_index][filter_id][k]);
  }

  if (filter_index == 2) {  // 8 tap.
    FilterHorizontal<2, is_compound>(src, src_stride, dst, dst_stride, width,
                                     height, v_tap);
  } else if (filter_index == 1) {  // 6 tap.
    FilterHorizontal<1, is_compound>(src + 1, src_stride, dst, dst_stride,
                                     width, height, v_tap);
  } else if (filter_index == 0) {  // 6 tap.
    FilterHorizontal<0, is_compound>(src + 1, src_stride, dst, dst_stride,
                                     width, height, v_tap);
  } else if (filter_index == 4) {  // 4 tap.
    FilterHorizontal<4, is_compound>(src + 2, src_stride, dst, dst_stride,
                                     width, height, v_tap);
  } else if (filter_index == 5) {  // 4 tap.
    FilterHorizontal<5, is_compound>(src + 2, src_stride, dst, dst_stride,
                                     width, height, v_tap);
  } else {  // 2 tap.
    FilterHorizontal<3, is_compound>(src + 3, src_stride, dst, dst_stride,
                                     width, height, v_tap);
  }
}

void ConvolveHorizontal_NEON(
    const void* LIBGAV1_RESTRICT const reference,
    const ptrdiff_t reference_stride, const int horizontal_filter_index,
    const int /*vertical_filter_index*/, const int horizontal_filter_id,
    const int /*vertical_filter_id*/, const int width, const int height,
    void* LIBGAV1_RESTRICT const prediction, const ptrdiff_t pred_stride) {
  const int filter_index = GetFilterIndex(horizontal_filter_index, width);
  // Set |src| to the outermost tap.
  const auto* const src =
      static_cast<const uint16_t*>(reference) - kHorizontalOffset;
  auto* const dest = static_cast<uint16_t*>(prediction);

  DoHorizontalPass(src, reference_stride, dest, pred_stride, width, height,
                   horizontal_filter_id, filter_index);
}

void ConvolveCompoundHorizontal_NEON(
    const void* LIBGAV1_RESTRICT const reference,
    const ptrdiff_t reference_stride, const int horizontal_filter_index,
    const int /*vertical_filter_index*/, const int horizontal_filter_id,
    const int /*vertical_filter_id*/, const int width, const int height,
    void* LIBGAV1_RESTRICT const prediction, const ptrdiff_t /*pred_stride*/) {
  const int filter_index = GetFilterIndex(horizontal_filter_index, width);
  const auto* const src =
      static_cast<const uint16_t*>(reference) - kHorizontalOffset;
  auto* const dest = static_cast<uint16_t*>(prediction);

  DoHorizontalPass</*is_compound=*/true>(src, reference_stride, dest, width,
                                         width, height, horizontal_filter_id,
                                         filter_index);
}

template <int filter_index, bool is_compound = false>
void FilterVertical(const uint16_t* LIBGAV1_RESTRICT const src,
                    const ptrdiff_t src_stride,
                    void* LIBGAV1_RESTRICT const dst,
                    const ptrdiff_t dst_stride, const int width,
                    const int height, const int16x4_t* const taps) {
  const int num_taps = GetNumTapsInFilter(filter_index);
  const int next_row = num_taps - 1;
  const uint16x4_t v_max_bitdepth = vdup_n_u16((1 << kBitdepth10) - 1);
  auto* const dst16 = static_cast<uint16_t*>(dst);
  assert(width >= 8);

  int x = 0;
  do {
    const uint16_t* src_x = src + x;
    uint16x8_t srcs[8];
    srcs[0] = vld1q_u16(src_x);
    src_x += src_stride;
    if (num_taps >= 4) {
      srcs[1] = vld1q_u16(src_x);
      src_x += src_stride;
      srcs[2] = vld1q_u16(src_x);
      src_x += src_stride;
      if (num_taps >= 6) {
        srcs[3] = vld1q_u16(src_x);
        src_x += src_stride;
        srcs[4] = vld1q_u16(src_x);
        src_x += src_stride;
        if (num_taps == 8) {
          srcs[5] = vld1q_u16(src_x);
          src_x += src_stride;
          srcs[6] = vld1q_u16(src_x);
          src_x += src_stride;
        }
      }
    }

    // Decreasing the y loop counter produces worse code with clang.
    // Don't unroll this loop since it generates too much code and the decoder
    // is even slower.
    int y = 0;
    do {
      srcs[next_row] = vld1q_u16(src_x);
      src_x += src_stride;

      const int32x4x2_t v_sum = SumOnePassTaps<filter_index>(srcs, taps);
      if (is_compound) {
        const int16x4_t v_compound_offset = vdup_n_s16(kCompoundOffset);
        const int16x4_t d0 =
            vqrshrn_n_s32(v_sum.val[0], kInterRoundBitsHorizontal - 1);
        const int16x4_t d1 =
            vqrshrn_n_s32(v_sum.val[1], kInterRoundBitsHorizontal - 1);
        vst1_u16(dst16 + x + y * dst_stride,
                 vreinterpret_u16_s16(vadd_s16(d0, v_compound_offset)));
        vst1_u16(dst16 + x + 4 + y * dst_stride,
                 vreinterpret_u16_s16(vadd_s16(d1, v_compound_offset)));
      } else {
        const uint16x4_t d0 = vmin_u16(
            vqrshrun_n_s32(v_sum.val[0], kFilterBits - 1), v_max_bitdepth);
        const uint16x4_t d1 = vmin_u16(
            vqrshrun_n_s32(v_sum.val[1], kFilterBits - 1), v_max_bitdepth);
        vst1_u16(dst16 + x + y * dst_stride, d0);
        vst1_u16(dst16 + x + 4 + y * dst_stride, d1);
      }

      srcs[0] = srcs[1];
      if (num_taps >= 4) {
        srcs[1] = srcs[2];
        srcs[2] = srcs[3];
        if (num_taps >= 6) {
          srcs[3] = srcs[4];
          srcs[4] = srcs[5];
          if (num_taps == 8) {
            srcs[5] = srcs[6];
            srcs[6] = srcs[7];
          }
        }
      }
    } while (++y < height);
    x += 8;
  } while (x < width);
}

template <int filter_index, bool is_compound = false>
void FilterVertical4xH(const uint16_t* LIBGAV1_RESTRICT src,
                       const ptrdiff_t src_stride,
                       void* LIBGAV1_RESTRICT const dst,
                       const ptrdiff_t dst_stride, const int height,
                       const int16x4_t* const taps) {
  const int num_taps = GetNumTapsInFilter(filter_index);
  const int next_row = num_taps - 1;
  const uint16x4_t v_max_bitdepth = vdup_n_u16((1 << kBitdepth10) - 1);
  auto* dst16 = static_cast<uint16_t*>(dst);

  uint16x4_t srcs[9];
  srcs[0] = vld1_u16(src);
  src += src_stride;
  if (num_taps >= 4) {
    srcs[1] = vld1_u16(src);
    src += src_stride;
    srcs[2] = vld1_u16(src);
    src += src_stride;
    if (num_taps >= 6) {
      srcs[3] = vld1_u16(src);
      src += src_stride;
      srcs[4] = vld1_u16(src);
      src += src_stride;
      if (num_taps == 8) {
        srcs[5] = vld1_u16(src);
        src += src_stride;
        srcs[6] = vld1_u16(src);
        src += src_stride;
      }
    }
  }

  int y = height;
  do {
    srcs[next_row] = vld1_u16(src);
    src += src_stride;
    srcs[num_taps] = vld1_u16(src);
    src += src_stride;

    const int32x4_t v_sum = SumOnePassTaps<filter_index>(srcs, taps);
    const int32x4_t v_sum_1 = SumOnePassTaps<filter_index>(srcs + 1, taps);
    if (is_compound) {
      const int16x4_t d0 = vqrshrn_n_s32(v_sum, kInterRoundBitsHorizontal - 1);
      const int16x4_t d1 =
          vqrshrn_n_s32(v_sum_1, kInterRoundBitsHorizontal - 1);
      vst1_u16(dst16,
               vreinterpret_u16_s16(vadd_s16(d0, vdup_n_s16(kCompoundOffset))));
      dst16 += dst_stride;
      vst1_u16(dst16,
               vreinterpret_u16_s16(vadd_s16(d1, vdup_n_s16(kCompoundOffset))));
      dst16 += dst_stride;
    } else {
      const uint16x4_t d0 =
          vmin_u16(vqrshrun_n_s32(v_sum, kFilterBits - 1), v_max_bitdepth);
      const uint16x4_t d1 =
          vmin_u16(vqrshrun_n_s32(v_sum_1, kFilterBits - 1), v_max_bitdepth);
      vst1_u16(dst16, d0);
      dst16 += dst_stride;
      vst1_u16(dst16, d1);
      dst16 += dst_stride;
    }

    srcs[0] = srcs[2];
    if (num_taps >= 4) {
      srcs[1] = srcs[3];
      srcs[2] = srcs[4];
      if (num_taps >= 6) {
        srcs[3] = srcs[5];
        srcs[4] = srcs[6];
        if (num_taps == 8) {
          srcs[5] = srcs[7];
          srcs[6] = srcs[8];
        }
      }
    }
    y -= 2;
  } while (y != 0);
}

template <int filter_index>
void FilterVertical2xH(const uint16_t* LIBGAV1_RESTRICT src,
                       const ptrdiff_t src_stride,
                       void* LIBGAV1_RESTRICT const dst,
                       const ptrdiff_t dst_stride, const int height,
                       const int16x4_t* const taps) {
  const int num_taps = GetNumTapsInFilter(filter_index);
  const int next_row = num_taps - 1;
  const uint16x4_t v_max_bitdepth = vdup_n_u16((1 << kBitdepth10) - 1);
  auto* dst16 = static_cast<uint16_t*>(dst);
  const uint16x4_t v_zero = vdup_n_u16(0);

  uint16x4_t srcs[9];
  srcs[0] = Load2<0>(src, v_zero);
  src += src_stride;
  if (num_taps >= 4) {
    srcs[0] = Load2<1>(src, srcs[0]);
    src += src_stride;
    srcs[2] = Load2<0>(src, v_zero);
    src += src_stride;
    srcs[1] = vext_u16(srcs[0], srcs[2], 2);
    if (num_taps >= 6) {
      srcs[2] = Load2<1>(src, srcs[2]);
      src += src_stride;
      srcs[4] = Load2<0>(src, v_zero);
      src += src_stride;
      srcs[3] = vext_u16(srcs[2], srcs[4], 2);
      if (num_taps == 8) {
        srcs[4] = Load2<1>(src, srcs[4]);
        src += src_stride;
        srcs[6] = Load2<0>(src, v_zero);
        src += src_stride;
        srcs[5] = vext_u16(srcs[4], srcs[6], 2);
      }
    }
  }

  int y = height;
  do {
    srcs[next_row - 1] = Load2<1>(src, srcs[next_row - 1]);
    src += src_stride;
    srcs[num_taps] = Load2<0>(src, v_zero);
    src += src_stride;
    srcs[next_row] = vext_u16(srcs[next_row - 1], srcs[num_taps], 2);

    const int32x4_t v_sum = SumOnePassTaps<filter_index>(srcs, taps);
    const uint16x4_t d0 =
        vmin_u16(vqrshrun_n_s32(v_sum, kFilterBits - 1), v_max_bitdepth);
    Store2<0>(dst16, d0);
    dst16 += dst_stride;
    Store2<1>(dst16, d0);
    dst16 += dst_stride;

    srcs[0] = srcs[2];
    if (num_taps >= 4) {
      srcs[1] = srcs[3];
      srcs[2] = srcs[4];
      if (num_taps >= 6) {
        srcs[3] = srcs[5];
        srcs[4] = srcs[6];
        if (num_taps == 8) {
          srcs[5] = srcs[7];
          srcs[6] = srcs[8];
        }
      }
    }
    y -= 2;
  } while (y != 0);
}

void ConvolveVertical_NEON(
    const void* LIBGAV1_RESTRICT const reference,
    const ptrdiff_t reference_stride, const int /*horizontal_filter_index*/,
    const int vertical_filter_index, const int /*horizontal_filter_id*/,
    const int vertical_filter_id, const int width, const int height,
    void* LIBGAV1_RESTRICT const prediction, const ptrdiff_t pred_stride) {
  const int filter_index = GetFilterIndex(vertical_filter_index, height);
  const int vertical_taps = GetNumTapsInFilter(filter_index);
  const ptrdiff_t src_stride = reference_stride >> 1;
  const auto* src = static_cast<const uint16_t*>(reference) -
                    (vertical_taps / 2 - 1) * src_stride;
  auto* const dest = static_cast<uint16_t*>(prediction);
  const ptrdiff_t dest_stride = pred_stride >> 1;
  assert(vertical_filter_id != 0);

  int16x4_t taps[8];
  for (int k = 0; k < kSubPixelTaps; ++k) {
    taps[k] =
        vdup_n_s16(kHalfSubPixelFilters[filter_index][vertical_filter_id][k]);
  }

  if (filter_index == 0) {  // 6 tap.
    if (width == 2) {
      FilterVertical2xH<0>(src, src_stride, dest, dest_stride, height,
                           taps + 1);
    } else if (width == 4) {
      FilterVertical4xH<0>(src, src_stride, dest, dest_stride, height,
                           taps + 1);
    } else {
      FilterVertical<0>(src, src_stride, dest, dest_stride, width, height,
                        taps + 1);
    }
  } else if ((static_cast<int>(filter_index == 1) &
              (static_cast<int>(vertical_filter_id == 1) |
               static_cast<int>(vertical_filter_id == 7) |
               static_cast<int>(vertical_filter_id == 8) |
               static_cast<int>(vertical_filter_id == 9) |
               static_cast<int>(vertical_filter_id == 15))) != 0) {  // 6 tap.
    if (width == 2) {
      FilterVertical2xH<1>(src, src_stride, dest, dest_stride, height,
                           taps + 1);
    } else if (width == 4) {
      FilterVertical4xH<1>(src, src_stride, dest, dest_stride, height,
                           taps + 1);
    } else {
      FilterVertical<1>(src, src_stride, dest, dest_stride, width, height,
                        taps + 1);
    }
  } else if (filter_index == 2) {  // 8 tap.
    if (width == 2) {
      FilterVertical2xH<2>(src, src_stride, dest, dest_stride, height, taps);
    } else if (width == 4) {
      FilterVertical4xH<2>(src, src_stride, dest, dest_stride, height, taps);
    } else {
      FilterVertical<2>(src, src_stride, dest, dest_stride, width, height,
                        taps);
    }
  } else if (filter_index == 3) {  // 2 tap.
    if (width == 2) {
      FilterVertical2xH<3>(src, src_stride, dest, dest_stride, height,
                           taps + 3);
    } else if (width == 4) {
      FilterVertical4xH<3>(src, src_stride, dest, dest_stride, height,
                           taps + 3);
    } else {
      FilterVertical<3>(src, src_stride, dest, dest_stride, width, height,
                        taps + 3);
    }
  } else {
    // 4 tap. When |filter_index| == 1 the |vertical_filter_id| values listed
    // below map to 4 tap filters.
    assert(filter_index == 5 || filter_index == 4 ||
           (filter_index == 1 &&
            (vertical_filter_id == 0 || vertical_filter_id == 2 ||
             vertical_filter_id == 3 || vertical_filter_id == 4 ||
             vertical_filter_id == 5 || vertical_filter_id == 6 ||
             vertical_filter_id == 10 || vertical_filter_id == 11 ||
             vertical_filter_id == 12 || vertical_filter_id == 13 ||
             vertical_filter_id == 14)));
    // According to GetNumTapsInFilter() this has 6 taps but here we are
    // treating it as though it has 4.
    if (filter_index == 1) src += src_stride;
    if (width == 2) {
      FilterVertical2xH<5>(src, src_stride, dest, dest_stride, height,
                           taps + 2);
    } else if (width == 4) {
      FilterVertical4xH<5>(src, src_stride, dest, dest_stride, height,
                           taps + 2);
    } else {
      FilterVertical<5>(src, src_stride, dest, dest_stride, width, height,
                        taps + 2);
    }
  }
}

void ConvolveCompoundVertical_NEON(
    const void* LIBGAV1_RESTRICT const reference,
    const ptrdiff_t reference_stride, const int /*horizontal_filter_index*/,
    const int vertical_filter_index, const int /*horizontal_filter_id*/,
    const int vertical_filter_id, const int width, const int height,
    void* LIBGAV1_RESTRICT const prediction, const ptrdiff_t /*pred_stride*/) {
  const int filter_index = GetFilterIndex(vertical_filter_index, height);
  const int vertical_taps = GetNumTapsInFilter(filter_index);
  const ptrdiff_t src_stride = reference_stride >> 1;
  const auto* src = static_cast<const uint16_t*>(reference) -
                    (vertical_taps / 2 - 1) * src_stride;
  auto* const dest = static_cast<uint16_t*>(prediction);
  assert(vertical_filter_id != 0);

  int16x4_t taps[8];
  for (int k = 0; k < kSubPixelTaps; ++k) {
    taps[k] =
        vdup_n_s16(kHalfSubPixelFilters[filter_index][vertical_filter_id][k]);
  }

  if (filter_index == 0) {  // 6 tap.
    if (width == 4) {
      FilterVertical4xH<0, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps + 1);
    } else {
      FilterVertical<0, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps + 1);
    }
  } else if ((static_cast<int>(filter_index == 1) &
              (static_cast<int>(vertical_filter_id == 1) |
               static_cast<int>(vertical_filter_id == 7) |
               static_cast<int>(vertical_filter_id == 8) |
               static_cast<int>(vertical_filter_id == 9) |
               static_cast<int>(vertical_filter_id == 15))) != 0) {  // 6 tap.
    if (width == 4) {
      FilterVertical4xH<1, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps + 1);
    } else {
      FilterVertical<1, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps + 1);
    }
  } else if (filter_index == 2) {  // 8 tap.
    if (width == 4) {
      FilterVertical4xH<2, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps);
    } else {
      FilterVertical<2, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps);
    }
  } else if (filter_index == 3) {  // 2 tap.
    if (width == 4) {
      FilterVertical4xH<3, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps + 3);
    } else {
      FilterVertical<3, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps + 3);
    }
  } else {
    // 4 tap. When |filter_index| == 1 the |filter_id| values listed below map
    // to 4 tap filters.
    assert(filter_index == 5 || filter_index == 4 ||
           (filter_index == 1 &&
            (vertical_filter_id == 2 || vertical_filter_id == 3 ||
             vertical_filter_id == 4 || vertical_filter_id == 5 ||
             vertical_filter_id == 6 || vertical_filter_id == 10 ||
             vertical_filter_id == 11 || vertical_filter_id == 12 ||
             vertical_filter_id == 13 || vertical_filter_id == 14)));
    // According to GetNumTapsInFilter() this has 6 taps but here we are
    // treating it as though it has 4.
    if (filter_index == 1) src += src_stride;
    if (width == 4) {
      FilterVertical4xH<5, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps + 2);
    } else {
      FilterVertical<5, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps + 2);
    }
  }
}

void ConvolveCompoundCopy_NEON(
    const void* const reference, const ptrdiff_t reference_stride,
    const int /*horizontal_filter_index*/, const int /*vertical_filter_index*/,
    const int /*horizontal_filter_id*/, const int /*vertical_filter_id*/,
    const int width, const int height, void* const prediction,
    const ptrdiff_t /*pred_stride*/) {
  const auto* src = static_cast<const uint16_t*>(reference);
  const ptrdiff_t src_stride = reference_stride >> 1;
  auto* dest = static_cast<uint16_t*>(prediction);
  constexpr int final_shift =
      kInterRoundBitsVertical - kInterRoundBitsCompoundVertical;
  const uint16x8_t offset =
      vdupq_n_u16((1 << kBitdepth10) + (1 << (kBitdepth10 - 1)));

  if (width >= 16) {
    int y = height;
    do {
      int x = 0;
      int w = width;
      do {
        const uint16x8_t v_src_lo = vld1q_u16(&src[x]);
        const uint16x8_t v_src_hi = vld1q_u16(&src[x + 8]);
        const uint16x8_t v_sum_lo = vaddq_u16(v_src_lo, offset);
        const uint16x8_t v_sum_hi = vaddq_u16(v_src_hi, offset);
        const uint16x8_t v_dest_lo = vshlq_n_u16(v_sum_lo, final_shift);
        const uint16x8_t v_dest_hi = vshlq_n_u16(v_sum_hi, final_shift);
        vst1q_u16(&dest[x], v_dest_lo);
        vst1q_u16(&dest[x + 8], v_dest_hi);
        x += 16;
        w -= 16;
      } while (w != 0);
      src += src_stride;
      dest += width;
    } while (--y != 0);
  } else if (width == 8) {
    int y = height;
    do {
      const uint16x8_t v_src_lo = vld1q_u16(&src[0]);
      const uint16x8_t v_src_hi = vld1q_u16(&src[src_stride]);
      const uint16x8_t v_sum_lo = vaddq_u16(v_src_lo, offset);
      const uint16x8_t v_sum_hi = vaddq_u16(v_src_hi, offset);
      const uint16x8_t v_dest_lo = vshlq_n_u16(v_sum_lo, final_shift);
      const uint16x8_t v_dest_hi = vshlq_n_u16(v_sum_hi, final_shift);
      vst1q_u16(&dest[0], v_dest_lo);
      vst1q_u16(&dest[8], v_dest_hi);
      src += src_stride << 1;
      dest += 16;
      y -= 2;
    } while (y != 0);
  } else {  // width == 4
    int y = height;
    do {
      const uint16x4_t v_src_lo = vld1_u16(&src[0]);
      const uint16x4_t v_src_hi = vld1_u16(&src[src_stride]);
      const uint16x4_t v_sum_lo = vadd_u16(v_src_lo, vget_low_u16(offset));
      const uint16x4_t v_sum_hi = vadd_u16(v_src_hi, vget_low_u16(offset));
      const uint16x4_t v_dest_lo = vshl_n_u16(v_sum_lo, final_shift);
      const uint16x4_t v_dest_hi = vshl_n_u16(v_sum_hi, final_shift);
      vst1_u16(&dest[0], v_dest_lo);
      vst1_u16(&dest[4], v_dest_hi);
      src += src_stride << 1;
      dest += 8;
      y -= 2;
    } while (y != 0);
  }
}

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth10);
  assert(dsp != nullptr);
  dsp->convolve[0][0][0][1] = ConvolveHorizontal_NEON;
  dsp->convolve[0][0][1][0] = ConvolveVertical_NEON;

  dsp->convolve[0][1][0][0] = ConvolveCompoundCopy_NEON;
  dsp->convolve[0][1][0][1] = ConvolveCompoundHorizontal_NEON;
  dsp->convolve[0][1][1][0] = ConvolveCompoundVertical_NEON;
}

}  // namespace

void ConvolveInit10bpp_NEON() { Init10bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !(LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10)

namespace libgav1 {
namespace dsp {

void ConvolveInit10bpp_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10
