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

#include "src/dsp/x86/weight_mask_sse4.h"

#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_SSE4_1

#include <smmintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

constexpr int kRoundingBits8bpp = 4;

template <bool mask_is_inverse>
inline void WeightMask4x2_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                               const uint16_t* prediction_1, ptrdiff_t stride_1,
                               uint8_t* mask, ptrdiff_t mask_stride) {
  const __m128i pred_0 =
      LoadHi8(LoadLo8(prediction_0), prediction_0 + stride_0);
  const __m128i pred_1 =
      LoadHi8(LoadLo8(prediction_1), prediction_1 + stride_1);
  const __m128i difference = RightShiftWithRounding_U16(
      _mm_abs_epi16(_mm_sub_epi16(pred_0, pred_1)), kRoundingBits8bpp);
  const __m128i scaled_difference = _mm_srli_epi16(difference, 4);
  const __m128i difference_offset = _mm_set1_epi8(38);
  const __m128i adjusted_difference =
      _mm_adds_epu8(_mm_packus_epi16(scaled_difference, scaled_difference),
                    difference_offset);
  const __m128i mask_ceiling = _mm_set1_epi8(64);
  const __m128i mask_value = _mm_min_epi8(adjusted_difference, mask_ceiling);
  if (mask_is_inverse) {
    const __m128i inverted_mask_value = _mm_sub_epi8(mask_ceiling, mask_value);
    Store4(mask, inverted_mask_value);
    Store4(mask + mask_stride, _mm_srli_si128(inverted_mask_value, 4));
  } else {
    Store4(mask, mask_value);
    Store4(mask + mask_stride, _mm_srli_si128(mask_value, 4));
  }
}

template <bool mask_is_inverse>
void WeightMask4x4_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                        const uint16_t* prediction_1, ptrdiff_t stride_1,
                        uint8_t* mask, ptrdiff_t mask_stride) {
  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += stride_0 << 1;
  prediction_1 += stride_1 << 1;
  mask += mask_stride << 1;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
}

template <bool mask_is_inverse>
void WeightMask4x8_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                        const uint16_t* prediction_1, ptrdiff_t stride_1,
                        uint8_t* mask, ptrdiff_t mask_stride) {
  const ptrdiff_t double_stride_0 = stride_0 << 1;
  const ptrdiff_t double_stride_1 = stride_1 << 1;
  const ptrdiff_t double_mask_stride = mask_stride << 1;
  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
}

template <bool mask_is_inverse>
void WeightMask4x16_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                         const uint16_t* prediction_1, ptrdiff_t stride_1,
                         uint8_t* mask, ptrdiff_t mask_stride) {
  const ptrdiff_t double_stride_0 = stride_0 << 1;
  const ptrdiff_t double_stride_1 = stride_1 << 1;
  const ptrdiff_t double_mask_stride = mask_stride << 1;
  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
  prediction_0 += double_stride_0;
  prediction_1 += double_stride_1;
  mask += double_mask_stride;

  WeightMask4x2_SSE4<mask_is_inverse>(prediction_0, stride_0, prediction_1,
                                      stride_1, mask, mask_stride);
}

template <bool mask_is_inverse>
inline void WeightMask8_SSE4(const uint16_t* prediction_0,
                             const uint16_t* prediction_1, uint8_t* mask) {
  const __m128i pred_0 = LoadAligned16(prediction_0);
  const __m128i pred_1 = LoadAligned16(prediction_1);
  const __m128i difference = RightShiftWithRounding_U16(
      _mm_abs_epi16(_mm_sub_epi16(pred_0, pred_1)), kRoundingBits8bpp);
  const __m128i scaled_difference = _mm_srli_epi16(difference, 4);
  const __m128i difference_offset = _mm_set1_epi8(38);
  const __m128i adjusted_difference =
      _mm_adds_epu8(_mm_packus_epi16(scaled_difference, scaled_difference),
                    difference_offset);
  const __m128i mask_ceiling = _mm_set1_epi8(64);
  const __m128i mask_value = _mm_min_epi8(adjusted_difference, mask_ceiling);
  if (mask_is_inverse) {
    const __m128i inverted_mask_value = _mm_sub_epi8(mask_ceiling, mask_value);
    StoreLo8(mask, inverted_mask_value);
  } else {
    StoreLo8(mask, mask_value);
  }
}

#define WEIGHT8_WITHOUT_STRIDE \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0, prediction_1, mask)

#define WEIGHT8_AND_STRIDE  \
  WEIGHT8_WITHOUT_STRIDE;   \
  prediction_0 += stride_0; \
  prediction_1 += stride_1; \
  mask += mask_stride

template <bool mask_is_inverse>
void WeightMask8x4_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                        const uint16_t* prediction_1, ptrdiff_t stride_1,
                        uint8_t* mask, ptrdiff_t mask_stride) {
  WEIGHT8_AND_STRIDE;
  WEIGHT8_AND_STRIDE;
  WEIGHT8_AND_STRIDE;
  WEIGHT8_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask8x8_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                        const uint16_t* prediction_1, ptrdiff_t stride_1,
                        uint8_t* mask, ptrdiff_t mask_stride) {
  int y = 0;
  do {
    WEIGHT8_AND_STRIDE;
  } while (++y < 7);
  WEIGHT8_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask8x16_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                         const uint16_t* prediction_1, ptrdiff_t stride_1,
                         uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT8_AND_STRIDE;
    WEIGHT8_AND_STRIDE;
    WEIGHT8_AND_STRIDE;
  } while (++y3 < 5);
  WEIGHT8_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask8x32_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                         const uint16_t* prediction_1, ptrdiff_t stride_1,
                         uint8_t* mask, ptrdiff_t mask_stride) {
  int y5 = 0;
  do {
    WEIGHT8_AND_STRIDE;
    WEIGHT8_AND_STRIDE;
    WEIGHT8_AND_STRIDE;
    WEIGHT8_AND_STRIDE;
    WEIGHT8_AND_STRIDE;
  } while (++y5 < 6);
  WEIGHT8_AND_STRIDE;
  WEIGHT8_WITHOUT_STRIDE;
}

#define WEIGHT16_WITHOUT_STRIDE                                         \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0, prediction_1, mask);  \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 8, prediction_1 + 8, \
                                    mask + 8)

#define WEIGHT16_AND_STRIDE \
  WEIGHT16_WITHOUT_STRIDE;  \
  prediction_0 += stride_0; \
  prediction_1 += stride_1; \
  mask += mask_stride

template <bool mask_is_inverse>
void WeightMask16x4_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                         const uint16_t* prediction_1, ptrdiff_t stride_1,
                         uint8_t* mask, ptrdiff_t mask_stride) {
  int y = 0;
  do {
    WEIGHT16_AND_STRIDE;
  } while (++y < 3);
  WEIGHT16_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask16x8_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                         const uint16_t* prediction_1, ptrdiff_t stride_1,
                         uint8_t* mask, ptrdiff_t mask_stride) {
  int y = 0;
  do {
    WEIGHT16_AND_STRIDE;
  } while (++y < 7);
  WEIGHT16_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask16x16_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
  } while (++y3 < 5);
  WEIGHT16_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask16x32_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y5 = 0;
  do {
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
  } while (++y5 < 6);
  WEIGHT16_AND_STRIDE;
  WEIGHT16_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask16x64_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
    WEIGHT16_AND_STRIDE;
  } while (++y3 < 21);
  WEIGHT16_WITHOUT_STRIDE;
}

#define WEIGHT32_WITHOUT_STRIDE                                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0, prediction_1, mask);    \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 8, prediction_1 + 8,   \
                                    mask + 8);                            \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 16, prediction_1 + 16, \
                                    mask + 16);                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 24, prediction_1 + 24, \
                                    mask + 24)

#define WEIGHT32_AND_STRIDE \
  WEIGHT32_WITHOUT_STRIDE;  \
  prediction_0 += stride_0; \
  prediction_1 += stride_1; \
  mask += mask_stride

template <bool mask_is_inverse>
void WeightMask32x8_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                         const uint16_t* prediction_1, ptrdiff_t stride_1,
                         uint8_t* mask, ptrdiff_t mask_stride) {
  WEIGHT32_AND_STRIDE;
  WEIGHT32_AND_STRIDE;
  WEIGHT32_AND_STRIDE;
  WEIGHT32_AND_STRIDE;
  WEIGHT32_AND_STRIDE;
  WEIGHT32_AND_STRIDE;
  WEIGHT32_AND_STRIDE;
  WEIGHT32_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask32x16_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
  } while (++y3 < 5);
  WEIGHT32_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask32x32_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y5 = 0;
  do {
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
  } while (++y5 < 6);
  WEIGHT32_AND_STRIDE;
  WEIGHT32_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask32x64_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
    WEIGHT32_AND_STRIDE;
  } while (++y3 < 21);
  WEIGHT32_WITHOUT_STRIDE;
}

#define WEIGHT64_WITHOUT_STRIDE                                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0, prediction_1, mask);    \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 8, prediction_1 + 8,   \
                                    mask + 8);                            \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 16, prediction_1 + 16, \
                                    mask + 16);                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 24, prediction_1 + 24, \
                                    mask + 24);                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 32, prediction_1 + 32, \
                                    mask + 32);                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 40, prediction_1 + 40, \
                                    mask + 40);                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 48, prediction_1 + 48, \
                                    mask + 48);                           \
  WeightMask8_SSE4<mask_is_inverse>(prediction_0 + 56, prediction_1 + 56, \
                                    mask + 56)

#define WEIGHT64_AND_STRIDE \
  WEIGHT64_WITHOUT_STRIDE;  \
  prediction_0 += stride_0; \
  prediction_1 += stride_1; \
  mask += mask_stride

template <bool mask_is_inverse>
void WeightMask64x16_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
  } while (++y3 < 5);
  WEIGHT64_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask64x32_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y5 = 0;
  do {
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
  } while (++y5 < 6);
  WEIGHT64_AND_STRIDE;
  WEIGHT64_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask64x64_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                          const uint16_t* prediction_1, ptrdiff_t stride_1,
                          uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
  } while (++y3 < 21);
  WEIGHT64_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask64x128_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                           const uint16_t* prediction_1, ptrdiff_t stride_1,
                           uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  do {
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
    WEIGHT64_AND_STRIDE;
  } while (++y3 < 42);
  WEIGHT64_AND_STRIDE;
  WEIGHT64_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask128x64_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                           const uint16_t* prediction_1, ptrdiff_t stride_1,
                           uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  const ptrdiff_t adjusted_stride_0 = stride_0 - 64;
  const ptrdiff_t adjusted_stride_1 = stride_1 - 64;
  const ptrdiff_t adjusted_mask_stride = mask_stride - 64;
  do {
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += 64;
    prediction_1 += 64;
    mask += 64;
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += adjusted_stride_0;
    prediction_1 += adjusted_stride_1;
    mask += adjusted_mask_stride;

    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += 64;
    prediction_1 += 64;
    mask += 64;
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += adjusted_stride_0;
    prediction_1 += adjusted_stride_1;
    mask += adjusted_mask_stride;

    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += 64;
    prediction_1 += 64;
    mask += 64;
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += adjusted_stride_0;
    prediction_1 += adjusted_stride_1;
    mask += adjusted_mask_stride;
  } while (++y3 < 21);
  WEIGHT64_WITHOUT_STRIDE;
  prediction_0 += 64;
  prediction_1 += 64;
  mask += 64;
  WEIGHT64_WITHOUT_STRIDE;
}

template <bool mask_is_inverse>
void WeightMask128x128_SSE4(const uint16_t* prediction_0, ptrdiff_t stride_0,
                            const uint16_t* prediction_1, ptrdiff_t stride_1,
                            uint8_t* mask, ptrdiff_t mask_stride) {
  int y3 = 0;
  const ptrdiff_t adjusted_stride_0 = stride_0 - 64;
  const ptrdiff_t adjusted_stride_1 = stride_1 - 64;
  const ptrdiff_t adjusted_mask_stride = mask_stride - 64;
  do {
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += 64;
    prediction_1 += 64;
    mask += 64;
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += adjusted_stride_0;
    prediction_1 += adjusted_stride_1;
    mask += adjusted_mask_stride;

    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += 64;
    prediction_1 += 64;
    mask += 64;
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += adjusted_stride_0;
    prediction_1 += adjusted_stride_1;
    mask += adjusted_mask_stride;

    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += 64;
    prediction_1 += 64;
    mask += 64;
    WEIGHT64_WITHOUT_STRIDE;
    prediction_0 += adjusted_stride_0;
    prediction_1 += adjusted_stride_1;
    mask += adjusted_mask_stride;
  } while (++y3 < 42);
  WEIGHT64_WITHOUT_STRIDE;
  prediction_0 += 64;
  prediction_1 += 64;
  mask += 64;
  WEIGHT64_WITHOUT_STRIDE;
  prediction_0 += adjusted_stride_0;
  prediction_1 += adjusted_stride_1;
  mask += adjusted_mask_stride;

  WEIGHT64_WITHOUT_STRIDE;
  prediction_0 += 64;
  prediction_1 += 64;
  mask += 64;
  WEIGHT64_WITHOUT_STRIDE;
}

#define INIT_WEIGHT_MASK_8BPP(width, height, w_index, h_index) \
  dsp->weight_mask[w_index][h_index][0] =                      \
      WeightMask##width##x##height##_SSE4<0>;                  \
  dsp->weight_mask[w_index][h_index][1] = WeightMask##width##x##height##_SSE4<1>
void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
  INIT_WEIGHT_MASK_8BPP(4, 4, 0, 0);
  INIT_WEIGHT_MASK_8BPP(4, 8, 0, 1);
  INIT_WEIGHT_MASK_8BPP(4, 16, 0, 2);
  INIT_WEIGHT_MASK_8BPP(8, 4, 1, 0);
  INIT_WEIGHT_MASK_8BPP(8, 8, 1, 1);
  INIT_WEIGHT_MASK_8BPP(8, 16, 1, 2);
  INIT_WEIGHT_MASK_8BPP(8, 32, 1, 3);
  INIT_WEIGHT_MASK_8BPP(16, 4, 2, 0);
  INIT_WEIGHT_MASK_8BPP(16, 8, 2, 1);
  INIT_WEIGHT_MASK_8BPP(16, 16, 2, 2);
  INIT_WEIGHT_MASK_8BPP(16, 32, 2, 3);
  INIT_WEIGHT_MASK_8BPP(16, 64, 2, 4);
  INIT_WEIGHT_MASK_8BPP(32, 8, 3, 1);
  INIT_WEIGHT_MASK_8BPP(32, 16, 3, 2);
  INIT_WEIGHT_MASK_8BPP(32, 32, 3, 3);
  INIT_WEIGHT_MASK_8BPP(32, 64, 3, 4);
  INIT_WEIGHT_MASK_8BPP(64, 16, 4, 2);
  INIT_WEIGHT_MASK_8BPP(64, 32, 4, 3);
  INIT_WEIGHT_MASK_8BPP(64, 64, 4, 4);
  INIT_WEIGHT_MASK_8BPP(64, 128, 4, 5);
  INIT_WEIGHT_MASK_8BPP(128, 64, 5, 4);
  INIT_WEIGHT_MASK_8BPP(128, 128, 5, 5);
}

}  // namespace
}  // namespace low_bitdepth

void WeightMaskInit_SSE4_1() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !LIBGAV1_ENABLE_SSE4_1

namespace libgav1 {
namespace dsp {

void WeightMaskInit_SSE4_1() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_SSE4_1