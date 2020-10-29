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

#include "src/dsp/loop_restoration.h"
#include "src/utils/cpu.h"

#if LIBGAV1_TARGETING_AVX2 && LIBGAV1_MAX_BITDEPTH >= 10
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/dsp/common.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/x86/common_avx2.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

inline void WienerHorizontalClip(const __m256i s[2],
                                 int16_t* const wiener_buffer) {
  constexpr int offset =
      1 << (10 + kWienerFilterBits - kInterRoundBitsHorizontal - 1);
  constexpr int limit = (offset << 2) - 1;
  const __m256i offsets = _mm256_set1_epi16(-offset);
  const __m256i limits = _mm256_set1_epi16(limit - offset);
  const __m256i round = _mm256_set1_epi32(1 << (kInterRoundBitsHorizontal - 1));
  const __m256i sum0 = _mm256_add_epi32(s[0], round);
  const __m256i sum1 = _mm256_add_epi32(s[1], round);
  const __m256i rounded_sum0 =
      _mm256_srai_epi32(sum0, kInterRoundBitsHorizontal);
  const __m256i rounded_sum1 =
      _mm256_srai_epi32(sum1, kInterRoundBitsHorizontal);
  const __m256i rounded_sum = _mm256_packs_epi32(rounded_sum0, rounded_sum1);
  const __m256i d0 = _mm256_max_epi16(rounded_sum, offsets);
  const __m256i d1 = _mm256_min_epi16(d0, limits);
  StoreAligned32(wiener_buffer, d1);
}

inline void WienerHorizontalTap7Kernel(const __m256i s[7],
                                       const __m256i filter[2],
                                       int16_t* const wiener_buffer) {
  const __m256i s06 = _mm256_add_epi16(s[0], s[6]);
  const __m256i s15 = _mm256_add_epi16(s[1], s[5]);
  const __m256i s24 = _mm256_add_epi16(s[2], s[4]);
  const __m256i ss0 = _mm256_unpacklo_epi16(s06, s15);
  const __m256i ss1 = _mm256_unpackhi_epi16(s06, s15);
  const __m256i ss2 = _mm256_unpacklo_epi16(s24, s[3]);
  const __m256i ss3 = _mm256_unpackhi_epi16(s24, s[3]);
  __m256i madds[4];
  madds[0] = _mm256_madd_epi16(ss0, filter[0]);
  madds[1] = _mm256_madd_epi16(ss1, filter[0]);
  madds[2] = _mm256_madd_epi16(ss2, filter[1]);
  madds[3] = _mm256_madd_epi16(ss3, filter[1]);
  madds[0] = _mm256_add_epi32(madds[0], madds[2]);
  madds[1] = _mm256_add_epi32(madds[1], madds[3]);
  WienerHorizontalClip(madds, wiener_buffer);
}

inline void WienerHorizontalTap5Kernel(const __m256i s[5], const __m256i filter,
                                       int16_t* const wiener_buffer) {
  const __m256i s04 = _mm256_add_epi16(s[0], s[4]);
  const __m256i s13 = _mm256_add_epi16(s[1], s[3]);
  const __m256i s2d = _mm256_add_epi16(s[2], s[2]);
  const __m256i s0m = _mm256_sub_epi16(s04, s2d);
  const __m256i s1m = _mm256_sub_epi16(s13, s2d);
  const __m256i ss0 = _mm256_unpacklo_epi16(s0m, s1m);
  const __m256i ss1 = _mm256_unpackhi_epi16(s0m, s1m);
  __m256i madds[2];
  madds[0] = _mm256_madd_epi16(ss0, filter);
  madds[1] = _mm256_madd_epi16(ss1, filter);
  const __m256i s2_lo = _mm256_unpacklo_epi16(s[2], _mm256_setzero_si256());
  const __m256i s2_hi = _mm256_unpackhi_epi16(s[2], _mm256_setzero_si256());
  const __m256i s2x128_lo = _mm256_slli_epi32(s2_lo, 7);
  const __m256i s2x128_hi = _mm256_slli_epi32(s2_hi, 7);
  madds[0] = _mm256_add_epi32(madds[0], s2x128_lo);
  madds[1] = _mm256_add_epi32(madds[1], s2x128_hi);
  WienerHorizontalClip(madds, wiener_buffer);
}

inline void WienerHorizontalTap3Kernel(const __m256i s[3], const __m256i filter,
                                       int16_t* const wiener_buffer) {
  const __m256i s02 = _mm256_add_epi16(s[0], s[2]);
  const __m256i ss0 = _mm256_unpacklo_epi16(s02, s[1]);
  const __m256i ss1 = _mm256_unpackhi_epi16(s02, s[1]);
  __m256i madds[2];
  madds[0] = _mm256_madd_epi16(ss0, filter);
  madds[1] = _mm256_madd_epi16(ss1, filter);
  WienerHorizontalClip(madds, wiener_buffer);
}

inline void WienerHorizontalTap7(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m256i* const coefficients,
                                 int16_t** const wiener_buffer) {
  __m256i filter[2];
  filter[0] = _mm256_shuffle_epi32(*coefficients, 0x0);
  filter[1] = _mm256_shuffle_epi32(*coefficients, 0x55);
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i s[7];
      s[0] = LoadUnaligned32(src + x + 0);
      s[1] = LoadUnaligned32(src + x + 1);
      s[2] = LoadUnaligned32(src + x + 2);
      s[3] = LoadUnaligned32(src + x + 3);
      s[4] = LoadUnaligned32(src + x + 4);
      s[5] = LoadUnaligned32(src + x + 5);
      s[6] = LoadUnaligned32(src + x + 6);
      WienerHorizontalTap7Kernel(s, filter, *wiener_buffer + x);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap5(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m256i* const coefficients,
                                 int16_t** const wiener_buffer) {
  const __m256i filter =
      _mm256_shuffle_epi8(*coefficients, _mm256_set1_epi32(0x05040302));
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i s[5];
      s[0] = LoadUnaligned32(src + x + 0);
      s[1] = LoadUnaligned32(src + x + 1);
      s[2] = LoadUnaligned32(src + x + 2);
      s[3] = LoadUnaligned32(src + x + 3);
      s[4] = LoadUnaligned32(src + x + 4);
      WienerHorizontalTap5Kernel(s, filter, *wiener_buffer + x);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap3(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m256i* const coefficients,
                                 int16_t** const wiener_buffer) {
  const auto filter = _mm256_shuffle_epi32(*coefficients, 0x55);
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i s[3];
      s[0] = LoadUnaligned32(src + x + 0);
      s[1] = LoadUnaligned32(src + x + 1);
      s[2] = LoadUnaligned32(src + x + 2);
      WienerHorizontalTap3Kernel(s, filter, *wiener_buffer + x);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap1(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      const __m256i s0 = LoadUnaligned32(src + x);
      const __m256i d0 = _mm256_slli_epi16(s0, 4);
      StoreAligned32(*wiener_buffer + x, d0);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline __m256i WienerVertical7(const __m256i a[4], const __m256i filter[4]) {
  const __m256i madd0 = _mm256_madd_epi16(a[0], filter[0]);
  const __m256i madd1 = _mm256_madd_epi16(a[1], filter[1]);
  const __m256i madd2 = _mm256_madd_epi16(a[2], filter[2]);
  const __m256i madd3 = _mm256_madd_epi16(a[3], filter[3]);
  const __m256i madd01 = _mm256_add_epi32(madd0, madd1);
  const __m256i madd23 = _mm256_add_epi32(madd2, madd3);
  const __m256i sum = _mm256_add_epi32(madd01, madd23);
  return _mm256_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m256i WienerVertical5(const __m256i a[3], const __m256i filter[3]) {
  const __m256i madd0 = _mm256_madd_epi16(a[0], filter[0]);
  const __m256i madd1 = _mm256_madd_epi16(a[1], filter[1]);
  const __m256i madd2 = _mm256_madd_epi16(a[2], filter[2]);
  const __m256i madd01 = _mm256_add_epi32(madd0, madd1);
  const __m256i sum = _mm256_add_epi32(madd01, madd2);
  return _mm256_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m256i WienerVertical3(const __m256i a[2], const __m256i filter[2]) {
  const __m256i madd0 = _mm256_madd_epi16(a[0], filter[0]);
  const __m256i madd1 = _mm256_madd_epi16(a[1], filter[1]);
  const __m256i sum = _mm256_add_epi32(madd0, madd1);
  return _mm256_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m256i WienerVerticalClip(const __m256i s[2]) {
  const __m256i d = _mm256_packus_epi32(s[0], s[1]);
  return _mm256_min_epu16(d, _mm256_set1_epi16(1023));
}

inline __m256i WienerVerticalFilter7(const __m256i a[7],
                                     const __m256i filter[2]) {
  const __m256i round = _mm256_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m256i b[4], c[2];
  b[0] = _mm256_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm256_unpacklo_epi16(a[2], a[3]);
  b[2] = _mm256_unpacklo_epi16(a[4], a[5]);
  b[3] = _mm256_unpacklo_epi16(a[6], round);
  c[0] = WienerVertical7(b, filter);
  b[0] = _mm256_unpackhi_epi16(a[0], a[1]);
  b[1] = _mm256_unpackhi_epi16(a[2], a[3]);
  b[2] = _mm256_unpackhi_epi16(a[4], a[5]);
  b[3] = _mm256_unpackhi_epi16(a[6], round);
  c[1] = WienerVertical7(b, filter);
  return WienerVerticalClip(c);
}

inline __m256i WienerVerticalFilter5(const __m256i a[5],
                                     const __m256i filter[3]) {
  const __m256i round = _mm256_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m256i b[3], c[2];
  b[0] = _mm256_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm256_unpacklo_epi16(a[2], a[3]);
  b[2] = _mm256_unpacklo_epi16(a[4], round);
  c[0] = WienerVertical5(b, filter);
  b[0] = _mm256_unpackhi_epi16(a[0], a[1]);
  b[1] = _mm256_unpackhi_epi16(a[2], a[3]);
  b[2] = _mm256_unpackhi_epi16(a[4], round);
  c[1] = WienerVertical5(b, filter);
  return WienerVerticalClip(c);
}

inline __m256i WienerVerticalFilter3(const __m256i a[3],
                                     const __m256i filter[2]) {
  const __m256i round = _mm256_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m256i b[2], c[2];
  b[0] = _mm256_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm256_unpacklo_epi16(a[2], round);
  c[0] = WienerVertical3(b, filter);
  b[0] = _mm256_unpackhi_epi16(a[0], a[1]);
  b[1] = _mm256_unpackhi_epi16(a[2], round);
  c[1] = WienerVertical3(b, filter);
  return WienerVerticalClip(c);
}

inline __m256i WienerVerticalTap7Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m256i filter[2], __m256i a[7]) {
  a[0] = LoadAligned32(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned32(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned32(wiener_buffer + 2 * wiener_stride);
  a[3] = LoadAligned32(wiener_buffer + 3 * wiener_stride);
  a[4] = LoadAligned32(wiener_buffer + 4 * wiener_stride);
  a[5] = LoadAligned32(wiener_buffer + 5 * wiener_stride);
  a[6] = LoadAligned32(wiener_buffer + 6 * wiener_stride);
  return WienerVerticalFilter7(a, filter);
}

inline __m256i WienerVerticalTap5Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m256i filter[3], __m256i a[5]) {
  a[0] = LoadAligned32(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned32(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned32(wiener_buffer + 2 * wiener_stride);
  a[3] = LoadAligned32(wiener_buffer + 3 * wiener_stride);
  a[4] = LoadAligned32(wiener_buffer + 4 * wiener_stride);
  return WienerVerticalFilter5(a, filter);
}

inline __m256i WienerVerticalTap3Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m256i filter[2], __m256i a[3]) {
  a[0] = LoadAligned32(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned32(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned32(wiener_buffer + 2 * wiener_stride);
  return WienerVerticalFilter3(a, filter);
}

inline void WienerVerticalTap7Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m256i filter[2], __m256i d[2]) {
  __m256i a[8];
  d[0] = WienerVerticalTap7Kernel(wiener_buffer, wiener_stride, filter, a);
  a[7] = LoadAligned32(wiener_buffer + 7 * wiener_stride);
  d[1] = WienerVerticalFilter7(a + 1, filter);
}

inline void WienerVerticalTap5Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m256i filter[3], __m256i d[2]) {
  __m256i a[6];
  d[0] = WienerVerticalTap5Kernel(wiener_buffer, wiener_stride, filter, a);
  a[5] = LoadAligned32(wiener_buffer + 5 * wiener_stride);
  d[1] = WienerVerticalFilter5(a + 1, filter);
}

inline void WienerVerticalTap3Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m256i filter[2], __m256i d[2]) {
  __m256i a[4];
  d[0] = WienerVerticalTap3Kernel(wiener_buffer, wiener_stride, filter, a);
  a[3] = LoadAligned32(wiener_buffer + 3 * wiener_stride);
  d[1] = WienerVerticalFilter3(a + 1, filter);
}

inline void WienerVerticalTap7(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[4], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m256i c = _mm256_broadcastq_epi64(LoadLo8(coefficients));
  __m256i filter[4];
  filter[0] = _mm256_shuffle_epi32(c, 0x0);
  filter[1] = _mm256_shuffle_epi32(c, 0x55);
  filter[2] = _mm256_shuffle_epi8(c, _mm256_set1_epi32(0x03020504));
  filter[3] =
      _mm256_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[0]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i d[2];
      WienerVerticalTap7Kernel2(wiener_buffer + x, width, filter, d);
      StoreUnaligned32(dst + x, d[0]);
      StoreUnaligned32(dst + dst_stride + x, d[1]);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m256i a[7];
      const __m256i d =
          WienerVerticalTap7Kernel(wiener_buffer + x, width, filter, a);
      StoreUnaligned32(dst + x, d);
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap5(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[3], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m256i c = _mm256_broadcastq_epi64(LoadLo8(coefficients));
  __m256i filter[3];
  filter[0] = _mm256_shuffle_epi32(c, 0x0);
  filter[1] = _mm256_shuffle_epi8(c, _mm256_set1_epi32(0x03020504));
  filter[2] =
      _mm256_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[0]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i d[2];
      WienerVerticalTap5Kernel2(wiener_buffer + x, width, filter, d);
      StoreUnaligned32(dst + x, d[0]);
      StoreUnaligned32(dst + dst_stride + x, d[1]);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m256i a[5];
      const __m256i d =
          WienerVerticalTap5Kernel(wiener_buffer + x, width, filter, a);
      StoreUnaligned32(dst + x, d);
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap3(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[2], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  __m256i filter[2];
  filter[0] =
      _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(coefficients));
  filter[1] =
      _mm256_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[0]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i d[2][2];
      WienerVerticalTap3Kernel2(wiener_buffer + x, width, filter, d[0]);
      StoreUnaligned32(dst + x, d[0][0]);
      StoreUnaligned32(dst + dst_stride + x, d[0][1]);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m256i a[3];
      const __m256i d =
          WienerVerticalTap3Kernel(wiener_buffer + x, width, filter, a);
      StoreUnaligned32(dst + x, d);
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap1Kernel(const int16_t* const wiener_buffer,
                                     uint16_t* const dst) {
  const __m256i a = LoadAligned32(wiener_buffer);
  const __m256i b = _mm256_add_epi16(a, _mm256_set1_epi16(8));
  const __m256i c = _mm256_srai_epi16(b, 4);
  const __m256i d = _mm256_max_epi16(c, _mm256_setzero_si256());
  const __m256i e = _mm256_min_epi16(d, _mm256_set1_epi16(1023));
  StoreUnaligned32(dst, e);
}

inline void WienerVerticalTap1(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               uint16_t* dst, const ptrdiff_t dst_stride) {
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      WienerVerticalTap1Kernel(wiener_buffer + x, dst + x);
      WienerVerticalTap1Kernel(wiener_buffer + width + x, dst + dst_stride + x);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      WienerVerticalTap1Kernel(wiener_buffer + x, dst + x);
      x += 16;
    } while (x < width);
  }
}

void WienerFilter_AVX2(const RestorationUnitInfo& restoration_info,
                       const void* const source, const void* const top_border,
                       const void* const bottom_border, const ptrdiff_t stride,
                       const int width, const int height,
                       RestorationBuffer* const restoration_buffer,
                       void* const dest) {
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

  // horizontal filtering.
  // Over-reads up to 15 - |kRestorationHorizontalBorder| values.
  const int height_horizontal =
      height + kWienerFilterTaps - 1 - 2 * number_rows_to_skip;
  const int height_extra = (height_horizontal - height) >> 1;
  assert(height_extra <= 2);
  const auto* const src = static_cast<const uint16_t*>(source);
  const auto* const top = static_cast<const uint16_t*>(top_border);
  const auto* const bottom = static_cast<const uint16_t*>(bottom_border);
  const __m128i c =
      LoadLo8(restoration_info.wiener_info.filter[WienerInfo::kHorizontal]);
  const __m256i coefficients_horizontal = _mm256_broadcastq_epi64(c);
  if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 0) {
    WienerHorizontalTap7(top + (2 - height_extra) * stride - 3, stride,
                         wiener_stride, height_extra, &coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap7(src - 3, stride, wiener_stride, height,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap7(bottom - 3, stride, wiener_stride, height_extra,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 1) {
    WienerHorizontalTap5(top + (2 - height_extra) * stride - 2, stride,
                         wiener_stride, height_extra, &coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap5(src - 2, stride, wiener_stride, height,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap5(bottom - 2, stride, wiener_stride, height_extra,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 2) {
    // The maximum over-reads happen here.
    WienerHorizontalTap3(top + (2 - height_extra) * stride - 1, stride,
                         wiener_stride, height_extra, &coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap3(src - 1, stride, wiener_stride, height,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap3(bottom - 1, stride, wiener_stride, height_extra,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
  } else {
    assert(number_leading_zero_coefficients[WienerInfo::kHorizontal] == 3);
    WienerHorizontalTap1(top + (2 - height_extra) * stride, stride,
                         wiener_stride, height_extra,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap1(src, stride, wiener_stride, height,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap1(bottom, stride, wiener_stride, height_extra,
                         &wiener_buffer_horizontal);
  }

  // vertical filtering.
  // Over-writes up to 15 values.
  const int16_t* const filter_vertical =
      restoration_info.wiener_info.filter[WienerInfo::kVertical];
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
                       height, filter_vertical + 1, dst, stride);
  } else if (number_leading_zero_coefficients[WienerInfo::kVertical] == 2) {
    WienerVerticalTap3(wiener_buffer_vertical + 2 * wiener_stride,
                       wiener_stride, height, filter_vertical + 2, dst, stride);
  } else {
    assert(number_leading_zero_coefficients[WienerInfo::kVertical] == 3);
    WienerVerticalTap1(wiener_buffer_vertical + 3 * wiener_stride,
                       wiener_stride, height, dst, stride);
  }
}

//------------------------------------------------------------------------------
// SGR

// When |height| is 1, |src_stride| could be set to an arbitrary value.
template <int size>
LIBGAV1_ALWAYS_INLINE void BoxSum(const uint16_t* src,
                                  const ptrdiff_t src_stride, const int height,
                                  const int width, uint16_t* const* sums,
                                  uint32_t* const* square_sums) {
  int y = height;
  do {
    uint32_t sum = 0;
    uint32_t square_sum = 0;
    for (int dx = 0; dx < size; ++dx) {
      const uint16_t source = src[dx];
      sum += source;
      square_sum += source * source;
    }
    (*sums)[0] = sum;
    (*square_sums)[0] = square_sum;
    int x = 1;
    do {
      const uint16_t source0 = src[x - 1];
      const uint16_t source1 = src[x - 1 + size];
      sum -= source0;
      sum += source1;
      square_sum -= source0 * source0;
      square_sum += source1 * source1;
      (*sums)[x] = sum;
      (*square_sums)[x] = square_sum;
    } while (++x != width);
    src += src_stride;
    ++sums;
    ++square_sums;
  } while (--y != 0);
}

// When |height| is 1, |src_stride| could be set to an arbitrary value.
LIBGAV1_ALWAYS_INLINE void BoxSum(const uint16_t* src,
                                  const ptrdiff_t src_stride, const int height,
                                  const int width, uint16_t* const* sum3,
                                  uint16_t* const* sum5,
                                  uint32_t* const* square_sum3,
                                  uint32_t* const* square_sum5) {
  int y = height;
  do {
    uint32_t sum = 0;
    uint32_t square_sum = 0;
    for (int dx = 0; dx < 4; ++dx) {
      const uint16_t source = src[dx];
      sum += source;
      square_sum += source * source;
    }
    int x = 0;
    do {
      const uint16_t source0 = src[x];
      const uint16_t source1 = src[x + 4];
      sum -= source0;
      square_sum -= source0 * source0;
      (*sum3)[x] = sum;
      (*square_sum3)[x] = square_sum;
      sum += source1;
      square_sum += source1 * source1;
      (*sum5)[x] = sum + source0;
      (*square_sum5)[x] = square_sum + source0 * source0;
    } while (++x != width);
    src += src_stride;
    ++sum3;
    ++sum5;
    ++square_sum3;
    ++square_sum5;
  } while (--y != 0);
}

template <int n>
inline void CalculateIntermediate(const uint32_t s, uint32_t a,
                                  const uint32_t b, uint8_t* const ma_ptr,
                                  uint32_t* const b_ptr) {
  // a: before shift, max is 25 * (2^(bitdepth) - 1) * (2^(bitdepth) - 1).
  // since max bitdepth = 12, max < 2^31.
  // after shift, a < 2^16 * n < 2^22 regardless of bitdepth
  a = RightShiftWithRounding(a, 4);
  // b: max is 25 * (2^(bitdepth) - 1). If bitdepth = 12, max < 2^19.
  // d < 2^8 * n < 2^14 regardless of bitdepth
  const uint32_t d = RightShiftWithRounding(b, 2);
  // p: Each term in calculating p = a * n - b * b is < 2^16 * n^2 < 2^28,
  // and p itself satisfies p < 2^14 * n^2 < 2^26.
  // This bound on p is due to:
  // https://en.wikipedia.org/wiki/Popoviciu's_inequality_on_variances
  // Note: Sometimes, in high bitdepth, we can end up with a*n < b*b.
  // This is an artifact of rounding, and can only happen if all pixels
  // are (almost) identical, so in this case we saturate to p=0.
  const uint32_t p = (a * n < d * d) ? 0 : a * n - d * d;
  // p * s < (2^14 * n^2) * round(2^20 / (n^2 * scale)) < 2^34 / scale <
  // 2^32 as long as scale >= 4. So p * s fits into a uint32_t, and z < 2^12
  // (this holds even after accounting for the rounding in s)
  const uint32_t z = RightShiftWithRounding(p * s, kSgrProjScaleBits);
  // ma: range [0, 255].
  const uint32_t ma = kSgrMaLookup[std::min(z, 255u)];
  const uint32_t one_over_n = ((1 << kSgrProjReciprocalBits) + (n >> 1)) / n;
  // ma < 2^8, b < 2^(bitdepth) * n,
  // one_over_n = round(2^12 / n)
  // => the product here is < 2^(20 + bitdepth) <= 2^32,
  // and b is set to a value < 2^(8 + bitdepth).
  // This holds even with the rounding in one_over_n and in the overall result,
  // as long as ma is strictly less than 2^8.
  const uint32_t b2 = ma * b * one_over_n;
  *ma_ptr = ma;
  *b_ptr = RightShiftWithRounding(b2, kSgrProjReciprocalBits);
}

template <typename T>
inline uint32_t Sum343(const T* const src) {
  return 3 * (src[0] + src[2]) + 4 * src[1];
}

template <typename T>
inline uint32_t Sum444(const T* const src) {
  return 4 * (src[0] + src[1] + src[2]);
}

template <typename T>
inline uint32_t Sum565(const T* const src) {
  return 5 * (src[0] + src[2]) + 6 * src[1];
}

LIBGAV1_ALWAYS_INLINE void BoxFilterPreProcess5(
    const uint16_t* const sum5[5], const uint32_t* const square_sum5[5],
    const int width, const uint32_t s, SgrBuffer* const sgr_buffer,
    uint16_t* const ma565, uint32_t* const b565) {
  int x = 0;
  do {
    uint32_t a = 0;
    uint32_t b = 0;
    for (int dy = 0; dy < 5; ++dy) {
      a += square_sum5[dy][x];
      b += sum5[dy][x];
    }
    CalculateIntermediate<25>(s, a, b, sgr_buffer->ma + x, sgr_buffer->b + x);
  } while (++x != width + 2);
  x = 0;
  do {
    ma565[x] = Sum565(sgr_buffer->ma + x);
    b565[x] = Sum565(sgr_buffer->b + x);
  } while (++x != width);
}

LIBGAV1_ALWAYS_INLINE void BoxFilterPreProcess3(
    const uint16_t* const sum3[3], const uint32_t* const square_sum3[3],
    const int width, const uint32_t s, const bool calculate444,
    SgrBuffer* const sgr_buffer, uint16_t* const ma343, uint32_t* const b343,
    uint16_t* const ma444, uint32_t* const b444) {
  int x = 0;
  do {
    uint32_t a = 0;
    uint32_t b = 0;
    for (int dy = 0; dy < 3; ++dy) {
      a += square_sum3[dy][x];
      b += sum3[dy][x];
    }
    CalculateIntermediate<9>(s, a, b, sgr_buffer->ma + x, sgr_buffer->b + x);
  } while (++x != width + 2);
  x = 0;
  do {
    ma343[x] = Sum343(sgr_buffer->ma + x);
    b343[x] = Sum343(sgr_buffer->b + x);
  } while (++x != width);
  if (calculate444) {
    x = 0;
    do {
      ma444[x] = Sum444(sgr_buffer->ma + x);
      b444[x] = Sum444(sgr_buffer->b + x);
    } while (++x != width);
  }
}

inline int CalculateFilteredOutput(const uint16_t src, const uint32_t ma,
                                   const uint32_t b, const int shift) {
  const int32_t v = b - ma * src;
  return RightShiftWithRounding(v,
                                kSgrProjSgrBits + shift - kSgrProjRestoreBits);
}

inline void BoxFilterPass1Kernel(const uint16_t src0, const uint16_t src1,
                                 const uint16_t* const ma565[2],
                                 const uint32_t* const b565[2],
                                 const ptrdiff_t x, int p[2]) {
  p[0] = CalculateFilteredOutput(src0, ma565[0][x] + ma565[1][x],
                                 b565[0][x] + b565[1][x], 5);
  p[1] = CalculateFilteredOutput(src1, ma565[1][x], b565[1][x], 4);
}

inline int BoxFilterPass2Kernel(const uint16_t src,
                                const uint16_t* const ma343[3],
                                const uint16_t* const ma444,
                                const uint32_t* const b343[3],
                                const uint32_t* const b444, const ptrdiff_t x) {
  const uint32_t ma = ma343[0][x] + ma444[x] + ma343[2][x];
  const uint32_t b = b343[0][x] + b444[x] + b343[2][x];
  return CalculateFilteredOutput(src, ma, b, 5);
}

inline uint16_t SelfGuidedFinal(const int src, const int v) {
  // if radius_pass_0 == 0 and radius_pass_1 == 0, the range of v is:
  // bits(u) + bits(w0/w1/w2) + 2 = bitdepth + 13.
  // Then, range of s is bitdepth + 2. This is a rough estimation, taking the
  // maximum value of each element.
  const int s = src + RightShiftWithRounding(
                          v, kSgrProjRestoreBits + kSgrProjPrecisionBits);
  return static_cast<uint16_t>(Clip3(s, 0, 1023));
}

inline uint16_t SelfGuidedDoubleMultiplier(const int src, const int filter0,
                                           const int filter1, const int16_t w0,
                                           const int16_t w2) {
  const int v = w0 * filter0 + w2 * filter1;
  return SelfGuidedFinal(src, v);
}

inline uint16_t SelfGuidedSingleMultiplier(const int src, const int filter,
                                           const int16_t w0) {
  const int v = w0 * filter;
  return SelfGuidedFinal(src, v);
}

inline void BoxFilterPass1(const uint16_t* const src, const ptrdiff_t stride,
                           uint16_t* const sum5[5],
                           uint32_t* const square_sum5[5], const int width,
                           const uint32_t scale, const int16_t w0,
                           SgrBuffer* const sgr_buffer,
                           uint16_t* const ma565[2], uint32_t* const b565[2],
                           uint16_t* dst) {
  BoxFilterPreProcess5(sum5, square_sum5, width, scale, sgr_buffer, ma565[1],
                       b565[1]);
  int x = 0;
  do {
    int p[2];
    BoxFilterPass1Kernel(src[x], src[stride + x], ma565, b565, x, p);
    dst[x] = SelfGuidedSingleMultiplier(src[x], p[0], w0);
    dst[stride + x] = SelfGuidedSingleMultiplier(src[stride + x], p[1], w0);
  } while (++x != width);
}

inline void BoxFilterPass2(
    const uint16_t* const src, const uint16_t* const src0, const int width,
    const uint16_t scale, const int16_t w0, uint16_t* const sum3[4],
    uint32_t* const square_sum3[4], SgrBuffer* const sgr_buffer,
    uint16_t* const ma343[4], uint16_t* const ma444[3], uint32_t* const b343[4],
    uint32_t* const b444[3], uint16_t* dst) {
  BoxSum<3>(src0, 0, 1, width + 2, sum3 + 2, square_sum3 + 2);
  BoxFilterPreProcess3(sum3, square_sum3, width, scale, true, sgr_buffer,
                       ma343[2], b343[2], ma444[1], b444[1]);
  int x = 0;
  do {
    const int p =
        BoxFilterPass2Kernel(src[x], ma343, ma444[0], b343, b444[0], x);
    dst[x] = SelfGuidedSingleMultiplier(src[x], p, w0);
  } while (++x != width);
}

inline void BoxFilter(const uint16_t* const src, const ptrdiff_t stride,
                      uint16_t* const sum3[4], uint16_t* const sum5[5],
                      uint32_t* const square_sum3[4],
                      uint32_t* const square_sum5[5], const int width,
                      const uint16_t scales[2], const int16_t w0,
                      const int16_t w2, SgrBuffer* const sgr_buffer,
                      uint16_t* const ma343[4], uint16_t* const ma444[3],
                      uint16_t* const ma565[2], uint32_t* const b343[4],
                      uint32_t* const b444[3], uint32_t* const b565[2],
                      uint16_t* dst) {
  BoxFilterPreProcess5(sum5, square_sum5, width, scales[0], sgr_buffer,
                       ma565[1], b565[1]);
  BoxFilterPreProcess3(sum3, square_sum3, width, scales[1], true, sgr_buffer,
                       ma343[2], b343[2], ma444[1], b444[1]);
  BoxFilterPreProcess3(sum3 + 1, square_sum3 + 1, width, scales[1], true,
                       sgr_buffer, ma343[3], b343[3], ma444[2], b444[2]);
  int x = 0;
  do {
    int p[2][2];
    BoxFilterPass1Kernel(src[x], src[stride + x], ma565, b565, x, p[0]);
    p[1][0] = BoxFilterPass2Kernel(src[x], ma343, ma444[0], b343, b444[0], x);
    p[1][1] = BoxFilterPass2Kernel(src[stride + x], ma343 + 1, ma444[1],
                                   b343 + 1, b444[1], x);
    dst[x] = SelfGuidedDoubleMultiplier(src[x], p[0][0], p[1][0], w0, w2);
    dst[stride + x] =
        SelfGuidedDoubleMultiplier(src[stride + x], p[0][1], p[1][1], w0, w2);
  } while (++x != width);
}

inline void BoxFilterProcess_C(const RestorationUnitInfo& restoration_info,
                               const uint16_t* src,
                               const uint16_t* const top_border,
                               const uint16_t* bottom_border,
                               const ptrdiff_t stride, const int width,
                               const int height, SgrBuffer* const sgr_buffer,
                               uint16_t* dst) {
  const auto temp_stride = Align<ptrdiff_t>(width, 8);
  const ptrdiff_t sum_stride = temp_stride + 8;
  const int sgr_proj_index = restoration_info.sgr_proj_info.index;
  const uint16_t* const scales = kSgrScaleParameter[sgr_proj_index];  // < 2^12.
  const int16_t w0 = restoration_info.sgr_proj_info.multiplier[0];
  const int16_t w1 = restoration_info.sgr_proj_info.multiplier[1];
  const int16_t w2 = (1 << kSgrProjPrecisionBits) - w0 - w1;
  uint16_t *sum3[4], *sum5[5], *ma343[4], *ma444[3], *ma565[2];
  uint32_t *square_sum3[4], *square_sum5[5], *b343[4], *b444[3], *b565[2];
  sum3[0] = sgr_buffer->sum3;
  square_sum3[0] = sgr_buffer->square_sum3;
  ma343[0] = sgr_buffer->ma343;
  b343[0] = sgr_buffer->b343;
  for (int i = 1; i <= 3; ++i) {
    sum3[i] = sum3[i - 1] + sum_stride;
    square_sum3[i] = square_sum3[i - 1] + sum_stride;
    ma343[i] = ma343[i - 1] + temp_stride;
    b343[i] = b343[i - 1] + temp_stride;
  }
  sum5[0] = sgr_buffer->sum5;
  square_sum5[0] = sgr_buffer->square_sum5;
  for (int i = 1; i <= 4; ++i) {
    sum5[i] = sum5[i - 1] + sum_stride;
    square_sum5[i] = square_sum5[i - 1] + sum_stride;
  }
  ma444[0] = sgr_buffer->ma444;
  b444[0] = sgr_buffer->b444;
  for (int i = 1; i <= 2; ++i) {
    ma444[i] = ma444[i - 1] + temp_stride;
    b444[i] = b444[i - 1] + temp_stride;
  }
  ma565[0] = sgr_buffer->ma565;
  ma565[1] = ma565[0] + temp_stride;
  b565[0] = sgr_buffer->b565;
  b565[1] = b565[0] + temp_stride;
  assert(scales[0] != 0);
  assert(scales[1] != 0);
  BoxSum(top_border, stride, 2, width + 2, sum3, sum5 + 1, square_sum3,
         square_sum5 + 1);
  sum5[0] = sum5[1];
  square_sum5[0] = square_sum5[1];
  BoxSum(src, stride, 1, width + 2, sum3 + 2, sum5 + 3, square_sum3 + 2,
         square_sum5 + 3);
  const uint16_t* const s = (height > 1) ? src + stride : bottom_border;
  BoxSum(s, 0, 1, width + 2, sum3 + 3, sum5 + 4, square_sum3 + 3,
         square_sum5 + 4);
  BoxFilterPreProcess5(sum5, square_sum5, width, scales[0], sgr_buffer,
                       ma565[0], b565[0]);
  BoxFilterPreProcess3(sum3, square_sum3, width, scales[1], false, sgr_buffer,
                       ma343[0], b343[0], nullptr, nullptr);
  BoxFilterPreProcess3(sum3 + 1, square_sum3 + 1, width, scales[1], true,
                       sgr_buffer, ma343[1], b343[1], ma444[0], b444[0]);
  sum5[0] = sgr_buffer->sum5;
  square_sum5[0] = sgr_buffer->square_sum5;

  for (int y = (height >> 1) - 1; y > 0; --y) {
    Circulate4PointersBy2<uint16_t>(sum3);
    Circulate4PointersBy2<uint32_t>(square_sum3);
    Circulate5PointersBy2<uint16_t>(sum5);
    Circulate5PointersBy2<uint32_t>(square_sum5);
    BoxSum(src + 2 * stride, stride, 2, width + 2, sum3 + 2, sum5 + 3,
           square_sum3 + 2, square_sum5 + 3);
    BoxFilter(src + 3, stride, sum3, sum5, square_sum3, square_sum5, width,
              scales, w0, w2, sgr_buffer, ma343, ma444, ma565, b343, b444, b565,
              dst);
    src += 2 * stride;
    dst += 2 * stride;
    Circulate4PointersBy2<uint16_t>(ma343);
    Circulate4PointersBy2<uint32_t>(b343);
    std::swap(ma444[0], ma444[2]);
    std::swap(b444[0], b444[2]);
    std::swap(ma565[0], ma565[1]);
    std::swap(b565[0], b565[1]);
  }

  Circulate4PointersBy2<uint16_t>(sum3);
  Circulate4PointersBy2<uint32_t>(square_sum3);
  Circulate5PointersBy2<uint16_t>(sum5);
  Circulate5PointersBy2<uint32_t>(square_sum5);
  if ((height & 1) == 0 || height > 1) {
    const uint16_t* sr;
    ptrdiff_t s_stride;
    if ((height & 1) == 0) {
      sr = bottom_border;
      s_stride = stride;
    } else {
      sr = src + 2 * stride;
      s_stride = bottom_border - (src + 2 * stride);
    }
    BoxSum(sr, s_stride, 2, width + 2, sum3 + 2, sum5 + 3, square_sum3 + 2,
           square_sum5 + 3);
    BoxFilter(src + 3, stride, sum3, sum5, square_sum3, square_sum5, width,
              scales, w0, w2, sgr_buffer, ma343, ma444, ma565, b343, b444, b565,
              dst);
  }
  if ((height & 1) != 0) {
    src += 3;
    if (height > 1) {
      src += 2 * stride;
      dst += 2 * stride;
      Circulate4PointersBy2<uint16_t>(sum3);
      Circulate4PointersBy2<uint32_t>(square_sum3);
      Circulate5PointersBy2<uint16_t>(sum5);
      Circulate5PointersBy2<uint32_t>(square_sum5);
      Circulate4PointersBy2<uint16_t>(ma343);
      Circulate4PointersBy2<uint32_t>(b343);
      std::swap(ma444[0], ma444[2]);
      std::swap(b444[0], b444[2]);
      std::swap(ma565[0], ma565[1]);
      std::swap(b565[0], b565[1]);
    }
    BoxSum(bottom_border + stride, stride, 1, width + 2, sum3 + 2, sum5 + 3,
           square_sum3 + 2, square_sum5 + 3);
    sum5[4] = sum5[3];
    square_sum5[4] = square_sum5[3];
    BoxFilterPreProcess5(sum5, square_sum5, width, scales[0], sgr_buffer,
                         ma565[1], b565[1]);
    BoxFilterPreProcess3(sum3, square_sum3, width, scales[1], false, sgr_buffer,
                         ma343[2], b343[2], nullptr, nullptr);
    int x = 0;
    do {
      const int p0 = CalculateFilteredOutput(src[x], ma565[0][x] + ma565[1][x],
                                             b565[0][x] + b565[1][x], 5);
      const int p1 =
          BoxFilterPass2Kernel(src[x], ma343, ma444[0], b343, b444[0], x);
      dst[x] = SelfGuidedDoubleMultiplier(src[x], p0, p1, w0, w2);
    } while (++x != width);
  }
}

inline void BoxFilterProcessPass1_C(
    const RestorationUnitInfo& restoration_info, const uint16_t* src,
    const uint16_t* const top_border, const uint16_t* bottom_border,
    const ptrdiff_t stride, const int width, const int height,
    SgrBuffer* const sgr_buffer, uint16_t* dst) {
  const auto temp_stride = Align<ptrdiff_t>(width, 8);
  const ptrdiff_t sum_stride = temp_stride + 8;
  const int sgr_proj_index = restoration_info.sgr_proj_info.index;
  const uint32_t scale = kSgrScaleParameter[sgr_proj_index][0];  // < 2^12.
  const int16_t w0 = restoration_info.sgr_proj_info.multiplier[0];
  uint16_t *sum5[5], *ma565[2];
  uint32_t *square_sum5[5], *b565[2];
  sum5[0] = sgr_buffer->sum5;
  square_sum5[0] = sgr_buffer->square_sum5;
  for (int i = 1; i <= 4; ++i) {
    sum5[i] = sum5[i - 1] + sum_stride;
    square_sum5[i] = square_sum5[i - 1] + sum_stride;
  }
  ma565[0] = sgr_buffer->ma565;
  ma565[1] = ma565[0] + temp_stride;
  b565[0] = sgr_buffer->b565;
  b565[1] = b565[0] + temp_stride;
  assert(scale != 0);
  BoxSum<5>(top_border, stride, 2, width + 2, sum5 + 1, square_sum5 + 1);
  sum5[0] = sum5[1];
  square_sum5[0] = square_sum5[1];
  BoxSum<5>(src, stride, 1, width + 2, sum5 + 3, square_sum5 + 3);
  const uint16_t* const s = (height > 1) ? src + stride : bottom_border;
  BoxSum<5>(s, 0, 1, width + 2, sum5 + 4, square_sum5 + 4);
  BoxFilterPreProcess5(sum5, square_sum5, width, scale, sgr_buffer, ma565[0],
                       b565[0]);
  sum5[0] = sgr_buffer->sum5;
  square_sum5[0] = sgr_buffer->square_sum5;

  for (int y = (height >> 1) - 1; y > 0; --y) {
    Circulate5PointersBy2<uint16_t>(sum5);
    Circulate5PointersBy2<uint32_t>(square_sum5);
    BoxSum<5>(src + 2 * stride, stride, 2, width + 2, sum5 + 3,
              square_sum5 + 3);
    BoxFilterPass1(src + 3, stride, sum5, square_sum5, width, scale, w0,
                   sgr_buffer, ma565, b565, dst);
    src += 2 * stride;
    dst += 2 * stride;
    std::swap(ma565[0], ma565[1]);
    std::swap(b565[0], b565[1]);
  }

  Circulate5PointersBy2<uint16_t>(sum5);
  Circulate5PointersBy2<uint32_t>(square_sum5);
  if ((height & 1) == 0 || height > 1) {
    const uint16_t* sr;
    ptrdiff_t s_stride;
    if ((height & 1) == 0) {
      sr = bottom_border;
      s_stride = stride;
    } else {
      sr = src + 2 * stride;
      s_stride = bottom_border - (src + 2 * stride);
    }
    BoxSum<5>(sr, s_stride, 2, width + 2, sum5 + 3, square_sum5 + 3);
    BoxFilterPass1(src + 3, stride, sum5, square_sum5, width, scale, w0,
                   sgr_buffer, ma565, b565, dst);
  }
  if ((height & 1) != 0) {
    src += 3;
    if (height > 1) {
      src += 2 * stride;
      dst += 2 * stride;
      std::swap(ma565[0], ma565[1]);
      std::swap(b565[0], b565[1]);
      Circulate5PointersBy2<uint16_t>(sum5);
      Circulate5PointersBy2<uint32_t>(square_sum5);
    }
    BoxSum<5>(bottom_border + stride, stride, 1, width + 2, sum5 + 3,
              square_sum5 + 3);
    sum5[4] = sum5[3];
    square_sum5[4] = square_sum5[3];
    BoxFilterPreProcess5(sum5, square_sum5, width, scale, sgr_buffer, ma565[1],
                         b565[1]);
    int x = 0;
    do {
      const int p = CalculateFilteredOutput(src[x], ma565[0][x] + ma565[1][x],
                                            b565[0][x] + b565[1][x], 5);
      dst[x] = SelfGuidedSingleMultiplier(src[x], p, w0);
    } while (++x != width);
  }
}

inline void BoxFilterProcessPass2_C(
    const RestorationUnitInfo& restoration_info, const uint16_t* src,
    const uint16_t* const top_border, const uint16_t* bottom_border,
    const ptrdiff_t stride, const int width, const int height,
    SgrBuffer* const sgr_buffer, uint16_t* dst) {
  assert(restoration_info.sgr_proj_info.multiplier[0] == 0);
  const auto temp_stride = Align<ptrdiff_t>(width, 8);
  const ptrdiff_t sum_stride = temp_stride + 8;
  const int16_t w1 = restoration_info.sgr_proj_info.multiplier[1];
  const int16_t w0 = (1 << kSgrProjPrecisionBits) - w1;
  const int sgr_proj_index = restoration_info.sgr_proj_info.index;
  const uint32_t scale = kSgrScaleParameter[sgr_proj_index][1];  // < 2^12.
  uint16_t *sum3[3], *ma343[3], *ma444[2];
  uint32_t *square_sum3[3], *b343[3], *b444[2];
  sum3[0] = sgr_buffer->sum3;
  square_sum3[0] = sgr_buffer->square_sum3;
  ma343[0] = sgr_buffer->ma343;
  b343[0] = sgr_buffer->b343;
  for (int i = 1; i <= 2; ++i) {
    sum3[i] = sum3[i - 1] + sum_stride;
    square_sum3[i] = square_sum3[i - 1] + sum_stride;
    ma343[i] = ma343[i - 1] + temp_stride;
    b343[i] = b343[i - 1] + temp_stride;
  }
  ma444[0] = sgr_buffer->ma444;
  ma444[1] = ma444[0] + temp_stride;
  b444[0] = sgr_buffer->b444;
  b444[1] = b444[0] + temp_stride;
  assert(scale != 0);
  BoxSum<3>(top_border, stride, 2, width + 2, sum3, square_sum3);
  BoxSum<3>(src, stride, 1, width + 2, sum3 + 2, square_sum3 + 2);
  BoxFilterPreProcess3(sum3, square_sum3, width, scale, false, sgr_buffer,
                       ma343[0], b343[0], nullptr, nullptr);
  Circulate3PointersBy1<uint16_t>(sum3);
  Circulate3PointersBy1<uint32_t>(square_sum3);
  const uint16_t* s;
  if (height > 1) {
    s = src + stride;
  } else {
    s = bottom_border;
    bottom_border += stride;
  }
  BoxSum<3>(s, 0, 1, width + 2, sum3 + 2, square_sum3 + 2);
  BoxFilterPreProcess3(sum3, square_sum3, width, scale, true, sgr_buffer,
                       ma343[1], b343[1], ma444[0], b444[0]);

  for (int y = height - 2; y > 0; --y) {
    Circulate3PointersBy1<uint16_t>(sum3);
    Circulate3PointersBy1<uint32_t>(square_sum3);
    BoxFilterPass2(src + 2, src + 2 * stride, width, scale, w0, sum3,
                   square_sum3, sgr_buffer, ma343, ma444, b343, b444, dst);
    src += stride;
    dst += stride;
    Circulate3PointersBy1<uint16_t>(ma343);
    Circulate3PointersBy1<uint32_t>(b343);
    std::swap(ma444[0], ma444[1]);
    std::swap(b444[0], b444[1]);
  }

  src += 2;
  int y = std::min(height, 2);
  do {
    Circulate3PointersBy1<uint16_t>(sum3);
    Circulate3PointersBy1<uint32_t>(square_sum3);
    BoxFilterPass2(src, bottom_border, width, scale, w0, sum3, square_sum3,
                   sgr_buffer, ma343, ma444, b343, b444, dst);
    src += stride;
    dst += stride;
    bottom_border += stride;
    Circulate3PointersBy1<uint16_t>(ma343);
    Circulate3PointersBy1<uint32_t>(b343);
    std::swap(ma444[0], ma444[1]);
    std::swap(b444[0], b444[1]);
  } while (--y != 0);
}

void SelfGuidedFilter_AVX2(
    const RestorationUnitInfo& restoration_info, const void* const source,
    const void* const top_border, const void* const bottom_border,
    const ptrdiff_t stride, const int width, const int height,
    RestorationBuffer* const restoration_buffer, void* const dest) {
  const int index = restoration_info.sgr_proj_info.index;
  const int radius_pass_0 = kSgrProjParams[index][0];  // 2 or 0
  const int radius_pass_1 = kSgrProjParams[index][2];  // 1 or 0
  const auto* const src = static_cast<const uint16_t*>(source);
  const auto* const top = static_cast<const uint16_t*>(top_border);
  const auto* const bottom = static_cast<const uint16_t*>(bottom_border);
  auto* const dst = static_cast<uint16_t*>(dest);
  SgrBuffer* const sgr_buffer = &restoration_buffer->sgr_buffer;
  if (radius_pass_1 == 0) {
    // |radius_pass_0| and |radius_pass_1| cannot both be 0, so we have the
    // following assertion.
    assert(radius_pass_0 != 0);
    BoxFilterProcessPass1_C(restoration_info, src - 3, top - 3, bottom - 3,
                            stride, width, height, sgr_buffer, dst);
  } else if (radius_pass_0 == 0) {
    BoxFilterProcessPass2_C(restoration_info, src - 2, top - 2, bottom - 2,
                            stride, width, height, sgr_buffer, dst);
  } else {
    BoxFilterProcess_C(restoration_info, src - 3, top - 3, bottom - 3, stride,
                       width, height, sgr_buffer, dst);
  }
}

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth10);
  assert(dsp != nullptr);
#if DSP_ENABLED_10BPP_AVX2(WienerFilter)
  dsp->loop_restorations[0] = WienerFilter_AVX2;
#endif
#if DSP_ENABLED_10BPP_AVX2(SelfGuidedFilter)
  dsp->loop_restorations[1] = SelfGuidedFilter_AVX2;
#endif
}

}  // namespace

void LoopRestorationInit10bpp_AVX2() { Init10bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else  // !(LIBGAV1_TARGETING_AVX2 && LIBGAV1_MAX_BITDEPTH >= 10)
namespace libgav1 {
namespace dsp {

void LoopRestorationInit10bpp_AVX2() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_TARGETING_AVX2 && LIBGAV1_MAX_BITDEPTH >= 10
