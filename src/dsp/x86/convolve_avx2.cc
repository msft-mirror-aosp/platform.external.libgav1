// Copyright 2020 The libgav1 Authors
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

#if LIBGAV1_TARGETING_AVX2
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/x86/common_avx2.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

constexpr int kHorizontalOffset = 3;

// Multiply every entry in |src[]| by the corresponding entry in |taps[]| and
// sum. The filters in |taps[]| are pre-shifted by 1. This prevents the final
// sum from outranging int16_t.
template <int filter_index>
__m256i SumOnePassTaps(const __m256i* const src, const __m256i* const taps) {
  __m256i sum;
  if (filter_index < 2) {
    // 6 taps.
    const __m256i v_madd_21 = _mm256_maddubs_epi16(src[0], taps[0]);  // k2k1
    const __m256i v_madd_43 = _mm256_maddubs_epi16(src[1], taps[1]);  // k4k3
    const __m256i v_madd_65 = _mm256_maddubs_epi16(src[2], taps[2]);  // k6k5
    sum = _mm256_add_epi16(v_madd_21, v_madd_43);
    sum = _mm256_add_epi16(sum, v_madd_65);
  } else if (filter_index == 2) {
    // 8 taps.
    const __m256i v_madd_10 = _mm256_maddubs_epi16(src[0], taps[0]);  // k1k0
    const __m256i v_madd_32 = _mm256_maddubs_epi16(src[1], taps[1]);  // k3k2
    const __m256i v_madd_54 = _mm256_maddubs_epi16(src[2], taps[2]);  // k5k4
    const __m256i v_madd_76 = _mm256_maddubs_epi16(src[3], taps[3]);  // k7k6
    const __m256i v_sum_3210 = _mm256_add_epi16(v_madd_10, v_madd_32);
    const __m256i v_sum_7654 = _mm256_add_epi16(v_madd_54, v_madd_76);
    sum = _mm256_add_epi16(v_sum_7654, v_sum_3210);
  } else if (filter_index == 3) {
    // 2 taps.
    sum = _mm256_maddubs_epi16(src[0], taps[0]);  // k4k3
  } else {
    // 4 taps.
    const __m256i v_madd_32 = _mm256_maddubs_epi16(src[0], taps[0]);  // k3k2
    const __m256i v_madd_54 = _mm256_maddubs_epi16(src[1], taps[1]);  // k5k4
    sum = _mm256_add_epi16(v_madd_32, v_madd_54);
  }
  return sum;
}

template <int filter_index>
__m256i SumHorizontalTaps(const __m256i* const src,
                          const __m256i* const v_tap) {
  __m256i v_src[4];
  const __m256i src_long = *src;
  const __m256i src_long_dup_lo = _mm256_unpacklo_epi8(src_long, src_long);
  const __m256i src_long_dup_hi = _mm256_unpackhi_epi8(src_long, src_long);

  if (filter_index < 2) {
    // 6 taps.
    v_src[0] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 3);   // _21
    v_src[1] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 7);   // _43
    v_src[2] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 11);  // _65
  } else if (filter_index == 2) {
    // 8 taps.
    v_src[0] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 1);   // _10
    v_src[1] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 5);   // _32
    v_src[2] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 9);   // _54
    v_src[3] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 13);  // _76
  } else if (filter_index == 3) {
    // 2 taps.
    v_src[0] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 7);  // _43
  } else if (filter_index > 3) {
    // 4 taps.
    v_src[0] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 5);  // _32
    v_src[1] = _mm256_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 9);  // _54
  }
  return SumOnePassTaps<filter_index>(v_src, v_tap);
}

template <int filter_index>
__m256i SimpleHorizontalTaps(const __m256i* const src,
                             const __m256i* const v_tap) {
  __m256i sum = SumHorizontalTaps<filter_index>(src, v_tap);

  // Normally the Horizontal pass does the downshift in two passes:
  // kInterRoundBitsHorizontal - 1 and then (kFilterBits -
  // kInterRoundBitsHorizontal). Each one uses a rounding shift. Combining them
  // requires adding the rounding offset from the skipped shift.
  constexpr int first_shift_rounding_bit = 1 << (kInterRoundBitsHorizontal - 2);

  sum = _mm256_add_epi16(sum, _mm256_set1_epi16(first_shift_rounding_bit));
  sum = RightShiftWithRounding_S16(sum, kFilterBits - 1);
  return _mm256_packus_epi16(sum, sum);
}

template <int filter_index>
__m128i SumHorizontalTaps2x2(const uint8_t* src, const ptrdiff_t src_stride,
                             const __m128i* const v_tap) {
  // 00 01 02 03 04 05 06 07 10 11 12 13 14 15 16 17
  const __m128i v_src = LoadHi8(LoadLo8(&src[0]), &src[src_stride]);

  if (filter_index == 3) {
    // 03 04 04 05 05 06 06 07 13 14 14 15 15 16 16 17
    const __m128i v_src_43 = _mm_shuffle_epi8(
        v_src, _mm_set_epi32(0x0f0e0e0d, 0x0d0c0c0b, 0x07060605, 0x05040403));
    const __m128i v_sum_43 = _mm_maddubs_epi16(v_src_43, v_tap[0]);  // k4k3
    return v_sum_43;
  }

  // 02 03 03 04 04 05 05 06 12 13 13 14 14 15 15 16
  const __m128i v_src_32 = _mm_shuffle_epi8(
      v_src, _mm_set_epi32(0x0e0d0d0c, 0x0c0b0b0a, 0x06050504, 0x04030302));
  // 04 05 05 06 06 07 07 xx 14 15 15 16 16 17 17 xx
  const __m128i v_src_54 = _mm_shuffle_epi8(
      v_src, _mm_set_epi32(0x800f0f0e, 0x0e0d0d0c, 0x80070706, 0x06050504));
  const __m128i v_madd_32 = _mm_maddubs_epi16(v_src_32, v_tap[0]);  // k3k2
  const __m128i v_madd_54 = _mm_maddubs_epi16(v_src_54, v_tap[1]);  // k5k4
  const __m128i v_sum_5432 = _mm_add_epi16(v_madd_54, v_madd_32);
  return v_sum_5432;
}

template <int filter_index>
__m128i SimpleHorizontalTaps2x2(const uint8_t* src, const ptrdiff_t src_stride,
                                const __m128i* const v_tap) {
  __m128i sum = SumHorizontalTaps2x2<filter_index>(src, src_stride, v_tap);

  // Normally the Horizontal pass does the downshift in two passes:
  // kInterRoundBitsHorizontal - 1 and then (kFilterBits -
  // kInterRoundBitsHorizontal). Each one uses a rounding shift. Combining them
  // requires adding the rounding offset from the skipped shift.
  constexpr int first_shift_rounding_bit = 1 << (kInterRoundBitsHorizontal - 2);

  sum = _mm_add_epi16(sum, _mm_set1_epi16(first_shift_rounding_bit));
  sum = RightShiftWithRounding_S16(sum, kFilterBits - 1);
  return _mm_packus_epi16(sum, sum);
}

template <int filter_index>
__m128i HorizontalTaps8To16_2x2(const uint8_t* src, const ptrdiff_t src_stride,
                                const __m128i* const v_tap) {
  const __m128i sum =
      SumHorizontalTaps2x2<filter_index>(src, src_stride, v_tap);

  return RightShiftWithRounding_S16(sum, kInterRoundBitsHorizontal - 1);
}

// Filter 2xh sizes.
template <int num_taps, int step, int filter_index, bool is_2d = false,
          bool is_compound = false>
void FilterHorizontal(const uint8_t* src, const ptrdiff_t src_stride,
                      void* const dest, const ptrdiff_t pred_stride,
                      const int /*width*/, const int height,
                      const __m128i* const v_tap) {
  auto* dest8 = static_cast<uint8_t*>(dest);
  auto* dest16 = static_cast<uint16_t*>(dest);

  // Horizontal passes only need to account for |num_taps| 2 and 4 when
  // |width| <= 4.
  assert(num_taps <= 4);
  if (num_taps <= 4) {
    if (!is_compound) {
      int y = 0;
      do {
        if (is_2d) {
          const __m128i sum =
              HorizontalTaps8To16_2x2<filter_index>(src, src_stride, v_tap);
          Store4(&dest16[0], sum);
          dest16 += pred_stride;
          Store4(&dest16[0], _mm_srli_si128(sum, 8));
          dest16 += pred_stride;
        } else {
          const __m128i sum =
              SimpleHorizontalTaps2x2<filter_index>(src, src_stride, v_tap);
          Store2(dest8, sum);
          dest8 += pred_stride;
          Store2(dest8, _mm_srli_si128(sum, 4));
          dest8 += pred_stride;
        }

        src += src_stride << 1;
        y += 2;
      } while (y < height - 1);

      // The 2d filters have an odd |height| because the horizontal pass
      // generates context for the vertical pass.
      if (is_2d) {
        assert(height % 2 == 1);
        __m128i sum;
        const __m128i input = LoadLo8(&src[2]);
        if (filter_index == 3) {
          // 03 04 04 05 05 06 06 07 ....
          const __m128i v_src_43 =
              _mm_srli_si128(_mm_unpacklo_epi8(input, input), 3);
          sum = _mm_maddubs_epi16(v_src_43, v_tap[0]);  // k4k3
        } else {
          // 02 03 03 04 04 05 05 06 06 07 ....
          const __m128i v_src_32 =
              _mm_srli_si128(_mm_unpacklo_epi8(input, input), 1);
          // 04 05 05 06 06 07 07 08 ...
          const __m128i v_src_54 = _mm_srli_si128(v_src_32, 4);
          const __m128i v_madd_32 =
              _mm_maddubs_epi16(v_src_32, v_tap[0]);  // k3k2
          const __m128i v_madd_54 =
              _mm_maddubs_epi16(v_src_54, v_tap[1]);  // k5k4
          sum = _mm_add_epi16(v_madd_54, v_madd_32);
        }
        sum = RightShiftWithRounding_S16(sum, kInterRoundBitsHorizontal - 1);
        Store4(dest16, sum);
      }
    }
  }
}

// Filter widths >= 4.
template <int num_taps, int step, int filter_index, bool is_2d = false,
          bool is_compound = false>
void FilterHorizontal(const uint8_t* src, const ptrdiff_t src_stride,
                      void* const dest, const ptrdiff_t pred_stride,
                      const int width, const int height,
                      const __m256i* const v_tap) {
  auto* dest8 = static_cast<uint8_t*>(dest);
  auto* dest16 = static_cast<uint16_t*>(dest);

  if (width >= 32) {
    int y = height;
    do {
      int x = 0;
      do {
        if (is_2d || is_compound) {
          // placeholder
        } else {
          // Load src used to calculate dest8[7:0] and dest8[23:16].
          const __m256i src_long = LoadUnaligned32(&src[x]);
          const __m256i result =
              SimpleHorizontalTaps<filter_index>(&src_long, v_tap);
          // Load src used to calculate dest8[15:8] and dest8[31:24].
          const __m256i src_long2 = LoadUnaligned32(&src[x + 8]);
          const __m256i result2 =
              SimpleHorizontalTaps<filter_index>(&src_long2, v_tap);
          // Combine results and store.
          StoreUnaligned32(&dest8[x], _mm256_unpacklo_epi64(result, result2));
        }
        x += step * 4;
      } while (x < width);
      src += src_stride;
      dest8 += pred_stride;
      dest16 += pred_stride;
    } while (--y != 0);
  } else if (width == 16) {
    int y = height;
    do {
      if (is_2d || is_compound) {
        // placeholder
      } else {
        // Load into 2 128 bit lanes.
        const __m256i src_long = _mm256_setr_m128i(
            LoadUnaligned16(&src[0]), LoadUnaligned16(&src[src_stride]));
        const __m256i result =
            SimpleHorizontalTaps<filter_index>(&src_long, v_tap);
        const __m256i src_long2 = _mm256_setr_m128i(
            LoadUnaligned16(&src[8]), LoadUnaligned16(&src[8 + src_stride]));
        const __m256i result2 =
            SimpleHorizontalTaps<filter_index>(&src_long2, v_tap);
        const __m256i packed_result = _mm256_unpacklo_epi64(result, result2);
        StoreUnaligned16(&dest8[0], _mm256_castsi256_si128(packed_result));
        StoreUnaligned16(&dest8[pred_stride],
                         _mm256_extracti128_si256(packed_result, 1));
      }
      src += src_stride * 2;
      dest8 += pred_stride * 2;
      dest16 += pred_stride * 2;
      y -= 2;
    } while (y != 0);
  } else if (width == 8) {
    int y = height;
    do {
      if (is_2d || is_compound) {
        // placeholder
      } else {
        const __m128i this_row = LoadUnaligned16(&src[0]);
        const __m128i next_row = LoadUnaligned16(&src[src_stride]);
        // Load into 2 128 bit lanes.
        const __m256i src_long = _mm256_setr_m128i(this_row, next_row);
        const __m256i result =
            SimpleHorizontalTaps<filter_index>(&src_long, v_tap);
        StoreLo8(&dest8[0], _mm256_castsi256_si128(result));
        StoreLo8(&dest8[pred_stride], _mm256_extracti128_si256(result, 1));
      }
      src += src_stride * 2;
      dest8 += pred_stride * 2;
      dest16 += pred_stride * 2;
      y -= 2;
    } while (y != 0);
  } else {  // width == 4
    int y = height;
    do {
      if (is_2d || is_compound) {
        // placeholder
      } else {
        const __m128i this_row = LoadUnaligned16(&src[0]);
        const __m128i next_row = LoadUnaligned16(&src[src_stride]);
        // Load into 2 128 bit lanes.
        const __m256i src_long = _mm256_setr_m128i(this_row, next_row);
        const __m256i result =
            SimpleHorizontalTaps<filter_index>(&src_long, v_tap);
        Store4(&dest8[0], _mm256_castsi256_si128(result));
        Store4(&dest8[pred_stride], _mm256_extracti128_si256(result, 1));
      }
      src += src_stride * 2;
      dest8 += pred_stride * 2;
      dest16 += pred_stride * 2;
      y -= 2;
    } while (y != 0);
  }
}

template <int num_taps, bool is_2d_vertical = false>
LIBGAV1_ALWAYS_INLINE void SetupTaps(const __m128i* const filter,
                                     __m128i* v_tap) {
  if (num_taps == 8) {
    v_tap[0] = _mm_shufflelo_epi16(*filter, 0x0);   // k1k0
    v_tap[1] = _mm_shufflelo_epi16(*filter, 0x55);  // k3k2
    v_tap[2] = _mm_shufflelo_epi16(*filter, 0xaa);  // k5k4
    v_tap[3] = _mm_shufflelo_epi16(*filter, 0xff);  // k7k6
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
      v_tap[1] = _mm_cvtepi8_epi16(v_tap[1]);
      v_tap[2] = _mm_cvtepi8_epi16(v_tap[2]);
      v_tap[3] = _mm_cvtepi8_epi16(v_tap[3]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
      v_tap[1] = _mm_unpacklo_epi64(v_tap[1], v_tap[1]);
      v_tap[2] = _mm_unpacklo_epi64(v_tap[2], v_tap[2]);
      v_tap[3] = _mm_unpacklo_epi64(v_tap[3], v_tap[3]);
    }
  } else if (num_taps == 6) {
    const __m128i adjusted_filter = _mm_srli_si128(*filter, 1);
    v_tap[0] = _mm_shufflelo_epi16(adjusted_filter, 0x0);   // k2k1
    v_tap[1] = _mm_shufflelo_epi16(adjusted_filter, 0x55);  // k4k3
    v_tap[2] = _mm_shufflelo_epi16(adjusted_filter, 0xaa);  // k6k5
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
      v_tap[1] = _mm_cvtepi8_epi16(v_tap[1]);
      v_tap[2] = _mm_cvtepi8_epi16(v_tap[2]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
      v_tap[1] = _mm_unpacklo_epi64(v_tap[1], v_tap[1]);
      v_tap[2] = _mm_unpacklo_epi64(v_tap[2], v_tap[2]);
    }
  } else if (num_taps == 4) {
    v_tap[0] = _mm_shufflelo_epi16(*filter, 0x55);  // k3k2
    v_tap[1] = _mm_shufflelo_epi16(*filter, 0xaa);  // k5k4
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
      v_tap[1] = _mm_cvtepi8_epi16(v_tap[1]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
      v_tap[1] = _mm_unpacklo_epi64(v_tap[1], v_tap[1]);
    }
  } else {  // num_taps == 2
    const __m128i adjusted_filter = _mm_srli_si128(*filter, 1);
    v_tap[0] = _mm_shufflelo_epi16(adjusted_filter, 0x55);  // k4k3
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
    }
  }
}

template <int num_taps, bool is_2d_vertical = false>
LIBGAV1_ALWAYS_INLINE void SetupTaps(const __m128i* const filter,
                                     __m256i* v_tap) {
  if (num_taps == 8) {
    v_tap[0] = _mm256_broadcastw_epi16(*filter);                     // k1k0
    v_tap[1] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 2));  // k3k2
    v_tap[2] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 4));  // k5k4
    v_tap[3] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 6));  // k7k6
    if (is_2d_vertical) {
      // placeholder
    }
  } else if (num_taps == 6) {
    v_tap[0] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 1));  // k2k1
    v_tap[1] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 3));  // k4k3
    v_tap[2] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 5));  // k6k5
    if (is_2d_vertical) {
      // placeholder
    }
  } else if (num_taps == 4) {
    v_tap[0] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 2));  // k3k2
    v_tap[1] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 4));  // k5k4
    if (is_2d_vertical) {
      // placeholder
    }
  } else {  // num_taps == 2
    v_tap[0] = _mm256_broadcastw_epi16(_mm_srli_si128(*filter, 3));  // k4k3
    if (is_2d_vertical) {
      // placeholder
    }
  }
}

template <bool is_2d = false, bool is_compound = false>
LIBGAV1_ALWAYS_INLINE void DoHorizontalPass2xH(
    const uint8_t* const src, const ptrdiff_t src_stride, void* const dst,
    const ptrdiff_t dst_stride, const int width, const int height,
    const int filter_id, const int filter_index) {
  assert(filter_id != 0);
  __m128i v_tap[4];
  const __m128i v_horizontal_filter =
      LoadLo8(kHalfSubPixelFilters[filter_index][filter_id]);

  if (filter_index == 4) {  // 4 tap.
    SetupTaps<4>(&v_horizontal_filter, v_tap);
    FilterHorizontal<4, 8, 4, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 5) {  // 4 tap.
    SetupTaps<4>(&v_horizontal_filter, v_tap);
    FilterHorizontal<4, 8, 5, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else {  // 2 tap.
    SetupTaps<2>(&v_horizontal_filter, v_tap);
    FilterHorizontal<2, 8, 3, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  }
}

template <bool is_2d = false, bool is_compound = false>
LIBGAV1_ALWAYS_INLINE void DoHorizontalPass(
    const uint8_t* const src, const ptrdiff_t src_stride, void* const dst,
    const ptrdiff_t dst_stride, const int width, const int height,
    const int filter_id, const int filter_index) {
  assert(filter_id != 0);
  __m256i v_tap[4];
  const __m128i v_horizontal_filter =
      LoadLo8(kHalfSubPixelFilters[filter_index][filter_id]);

  if (filter_index == 2) {  // 8 tap.
    SetupTaps<8>(&v_horizontal_filter, v_tap);
    FilterHorizontal<8, 8, 2, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 1) {  // 6 tap.
    SetupTaps<6>(&v_horizontal_filter, v_tap);
    FilterHorizontal<6, 8, 1, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 0) {  // 6 tap.
    SetupTaps<6>(&v_horizontal_filter, v_tap);
    FilterHorizontal<6, 8, 0, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 4) {  // 4 tap.
    SetupTaps<4>(&v_horizontal_filter, v_tap);
    FilterHorizontal<4, 8, 4, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 5) {  // 4 tap.
    SetupTaps<4>(&v_horizontal_filter, v_tap);
    FilterHorizontal<4, 8, 5, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else {  // 2 tap.
    SetupTaps<2>(&v_horizontal_filter, v_tap);
    FilterHorizontal<2, 8, 3, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  }
}

void ConvolveHorizontal_AVX2(const void* const reference,
                             const ptrdiff_t reference_stride,
                             const int horizontal_filter_index,
                             const int /*vertical_filter_index*/,
                             const int horizontal_filter_id,
                             const int /*vertical_filter_id*/, const int width,
                             const int height, void* prediction,
                             const ptrdiff_t pred_stride) {
  const int filter_index = GetFilterIndex(horizontal_filter_index, width);
  // Set |src| to the outermost tap.
  const auto* src = static_cast<const uint8_t*>(reference) - kHorizontalOffset;
  auto* dest = static_cast<uint8_t*>(prediction);

  if (width > 2) {
    DoHorizontalPass(src, reference_stride, dest, pred_stride, width, height,
                     horizontal_filter_id, filter_index);
  } else {
    // Use non avx2 version for smaller widths.
    DoHorizontalPass2xH(src, reference_stride, dest, pred_stride, width, height,
                        horizontal_filter_id, filter_index);
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
  dsp->convolve[0][0][0][1] = ConvolveHorizontal_AVX2;
}

}  // namespace
}  // namespace low_bitdepth

void ConvolveInit_AVX2() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else  // !LIBGAV1_TARGETING_AVX2
namespace libgav1 {
namespace dsp {

void ConvolveInit_AVX2() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_TARGETING_AVX2
