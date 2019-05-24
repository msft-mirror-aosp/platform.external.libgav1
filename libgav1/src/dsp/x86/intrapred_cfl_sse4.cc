#include "src/dsp/dsp.h"
#include "src/dsp/x86/intrapred_sse4.h"

#if LIBGAV1_ENABLE_SSE4_1

#include <emmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/constants.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

//------------------------------------------------------------------------------
// CflIntraPredictor_SSE4_1

inline __m128i CflPredictUnclipped(const __m128i* input, __m128i alpha_q12,
                                   __m128i alpha_sign, __m128i dc_q0) {
  __m128i ac_q3 = LoadUnaligned16(input);
  __m128i ac_sign = _mm_sign_epi16(alpha_sign, ac_q3);
  __m128i scaled_luma_q0 = _mm_mulhrs_epi16(_mm_abs_epi16(ac_q3), alpha_q12);
  scaled_luma_q0 = _mm_sign_epi16(scaled_luma_q0, ac_sign);
  return _mm_add_epi16(scaled_luma_q0, dc_q0);
}

template <int width, int height>
void CflIntraPredictor_SSE4_1(
    void* const dest, ptrdiff_t stride,
    const int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int alpha) {
  auto* dst = reinterpret_cast<uint8_t*>(dest);
  const __m128i alpha_sign = _mm_set1_epi16(alpha);
  const __m128i alpha_q12 = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
  auto* row = reinterpret_cast<const __m128i*>(luma);
  const int kCflLumaBufferStrideLog2_16i = 5;
  const int kCflLumaBufferStrideLog2_128i = kCflLumaBufferStrideLog2_16i - 3;
  const __m128i* row_end = row + (height << kCflLumaBufferStrideLog2_128i);
  const __m128i dc_val = _mm_set1_epi16(dst[0]);
  do {
    __m128i res = CflPredictUnclipped(row, alpha_q12, alpha_sign, dc_val);
    if (width < 16) {
      res = _mm_packus_epi16(res, res);
      if (width == 4) {
        Store4(dst, res);
      } else {
        StoreLo8(dst, res);
      }
    } else {
      __m128i next =
          CflPredictUnclipped(row + 1, alpha_q12, alpha_sign, dc_val);
      res = _mm_packus_epi16(res, next);
      StoreUnaligned16(dst, res);
      if (width == 32) {
        res = CflPredictUnclipped(row + 2, alpha_q12, alpha_sign, dc_val);
        next = CflPredictUnclipped(row + 3, alpha_q12, alpha_sign, dc_val);
        res = _mm_packus_epi16(res, next);
        StoreUnaligned16(dst + 16, res);
      }
    }
    dst += stride;
  } while ((row += (1 << kCflLumaBufferStrideLog2_128i)) < row_end);
}

template <int block_height_log2>
void CflSubsampler444_4xH_SSE4_1(
    int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int /*max_luma_width*/, const int /*max_luma_height*/,
    const void* const source, ptrdiff_t stride) {
  const int block_height = 1 << block_height_log2;
  const auto* src = static_cast<const uint8_t*>(source);
  __m128i sum = _mm_setzero_si128();
  int16_t* luma_ptr = luma[0];
  const __m128i zero = _mm_setzero_si128();
  for (int y = 0; y < block_height; y += 4) {
    __m128i samples01 = Load4(src);
    src += stride;
    int src_bytes;
    memcpy(&src_bytes, src, 4);
    samples01 = _mm_insert_epi32(samples01, src_bytes, 1);
    src += stride;
    samples01 = _mm_slli_epi16(_mm_cvtepu8_epi16(samples01), 3);
    StoreLo8(luma_ptr, samples01);
    luma_ptr += kCflLumaBufferStride;
    StoreHi8(luma_ptr, samples01);
    luma_ptr += kCflLumaBufferStride;

    __m128i samples23 = Load4(src);
    src += stride;
    memcpy(&src_bytes, src, 4);
    samples23 = _mm_insert_epi32(samples23, src_bytes, 1);
    src += stride;
    samples23 = _mm_slli_epi16(_mm_cvtepu8_epi16(samples23), 3);
    StoreLo8(luma_ptr, samples23);
    luma_ptr += kCflLumaBufferStride;
    StoreHi8(luma_ptr, samples23);
    luma_ptr += kCflLumaBufferStride;

    const __m128i sample_sum = _mm_add_epi16(samples01, samples23);
    sum = _mm_add_epi32(sum, _mm_cvtepu16_epi32(sample_sum));
    sum = _mm_add_epi32(sum, _mm_unpackhi_epi16(sample_sum, zero));
  }
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));

  __m128i averages = RightShiftWithRounding_U32(
      sum, block_height_log2 + 2 /* log2 of width 4 */);
  averages = _mm_shufflelo_epi16(averages, 0);
  luma_ptr = luma[0];
  for (int y = 0; y < block_height; ++y, luma_ptr += kCflLumaBufferStride) {
    const __m128i samples = LoadLo8(luma_ptr);
    StoreLo8(luma_ptr, _mm_sub_epi16(samples, averages));
  }
}

template <int block_height_log2>
void CflSubsampler444_8xH_SSE4_1(
    int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int /*max_luma_width*/, const int /*max_luma_height*/,
    const void* const source, ptrdiff_t stride) {
  const int block_height = 1 << block_height_log2;
  const __m128i dup16 = _mm_set1_epi32(0x01000100);
  const auto* src = static_cast<const uint8_t*>(source);
  __m128i sum = _mm_setzero_si128();
  int16_t* luma_ptr = luma[0];
  const __m128i zero = _mm_setzero_si128();
  for (int y = 0; y < block_height; y += 2) {
    __m128i samples0 = LoadLo8(src);
    src += stride;
    samples0 = _mm_slli_epi16(_mm_cvtepu8_epi16(samples0), 3);
    StoreUnaligned16(luma_ptr, samples0);
    luma_ptr += kCflLumaBufferStride;

    __m128i samples1 = LoadLo8(src);
    src += stride;
    samples1 = _mm_slli_epi16(_mm_cvtepu8_epi16(samples1), 3);
    StoreUnaligned16(luma_ptr, samples1);
    luma_ptr += kCflLumaBufferStride;

    const __m128i sample_sum = _mm_add_epi16(samples0, samples1);
    sum = _mm_add_epi32(sum, _mm_cvtepu16_epi32(sample_sum));
    sum = _mm_add_epi32(sum, _mm_unpackhi_epi16(sample_sum, zero));
  }
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));

  __m128i averages = RightShiftWithRounding_U32(
      sum, block_height_log2 + 3 /* log2 of width 8 */);
  averages = _mm_shuffle_epi8(averages, dup16);
  luma_ptr = luma[0];
  for (int y = 0; y < block_height; ++y, luma_ptr += kCflLumaBufferStride) {
    const __m128i samples = LoadUnaligned16(luma_ptr);
    StoreUnaligned16(luma_ptr, _mm_sub_epi16(samples, averages));
  }
}

// This function will only work for block_width 16 and 32.
template <int block_width_log2, int block_height_log2>
void CflSubsampler444_SSE4_1(
    int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int /*max_luma_width*/, const int /*max_luma_height*/,
    const void* const source, ptrdiff_t stride) {
  const int block_height = 1 << block_height_log2;
  const int block_width = 1 << block_width_log2;
  const __m128i dup16 = _mm_set1_epi32(0x01000100);
  const __m128i zero = _mm_setzero_si128();
  const auto* src = static_cast<const uint8_t*>(source);
  int16_t* luma_ptr = luma[0];
  __m128i sum = _mm_setzero_si128();
  for (int y = 0; y < block_height;
       luma_ptr += kCflLumaBufferStride, src += stride, ++y) {
    const __m128i samples01 = LoadUnaligned16(src);
    const __m128i samples0 = _mm_slli_epi16(_mm_cvtepu8_epi16(samples01), 3);
    const __m128i samples1 =
        _mm_slli_epi16(_mm_unpackhi_epi8(samples01, zero), 3);
    StoreUnaligned16(luma_ptr, samples0);
    StoreUnaligned16(luma_ptr + 8, samples1);
    __m128i inner_sum = _mm_add_epi16(samples0, samples1);
    if (block_width == 32) {
      const __m128i samples23 = LoadUnaligned16(src + 16);
      const __m128i samples2 = _mm_slli_epi16(_mm_cvtepu8_epi16(samples23), 3);
      const __m128i samples3 =
          _mm_slli_epi16(_mm_unpackhi_epi8(samples23, zero), 3);
      StoreUnaligned16(luma_ptr + 16, samples2);
      StoreUnaligned16(luma_ptr + 24, samples3);
      inner_sum = _mm_add_epi16(samples2, inner_sum);
      inner_sum = _mm_add_epi16(samples3, inner_sum);
    }
    const __m128i inner_sum_lo = _mm_cvtepu16_epi32(inner_sum);
    const __m128i inner_sum_hi = _mm_unpackhi_epi16(inner_sum, zero);
    sum = _mm_add_epi32(sum, inner_sum_lo);
    sum = _mm_add_epi32(sum, inner_sum_hi);
  }
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));

  __m128i averages =
      RightShiftWithRounding_U32(sum, block_width_log2 + block_height_log2);
  averages = _mm_shuffle_epi8(averages, dup16);
  luma_ptr = luma[0];
  for (int y = 0; y < block_height; ++y, luma_ptr += kCflLumaBufferStride) {
    for (int x = 0; x < block_width; x += 8) {
      __m128i samples = LoadUnaligned16(&luma_ptr[x]);
      StoreUnaligned16(&luma_ptr[x], _mm_sub_epi16(samples, averages));
    }
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize4x4_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize4x4][kSubsamplingType444] =
      CflSubsampler444_4xH_SSE4_1<2>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize4x8_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize4x8][kSubsamplingType444] =
      CflSubsampler444_4xH_SSE4_1<3>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize4x16_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize4x16][kSubsamplingType444] =
      CflSubsampler444_4xH_SSE4_1<4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x4_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize8x4][kSubsamplingType444] =
      CflSubsampler444_8xH_SSE4_1<2>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x8_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize8x8][kSubsamplingType444] =
      CflSubsampler444_8xH_SSE4_1<3>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x16_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize8x16][kSubsamplingType444] =
      CflSubsampler444_8xH_SSE4_1<4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x32_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize8x32][kSubsamplingType444] =
      CflSubsampler444_8xH_SSE4_1<5>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x4_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize16x4][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<4, 2>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x8_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize16x8][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<4, 3>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x16_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize16x16][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<4, 4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x32_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize16x32][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<4, 5>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize32x8_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize32x8][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<5, 3>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize32x16_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize32x16][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<5, 4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize32x32_CflSubsampler444)
  dsp->cfl_subsamplers[kTransformSize32x32][kSubsamplingType444] =
      CflSubsampler444_SSE4_1<5, 5>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize4x4_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize4x4] = CflIntraPredictor_SSE4_1<4, 4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize4x8_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize4x8] = CflIntraPredictor_SSE4_1<4, 8>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize4x16_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize4x16] =
      CflIntraPredictor_SSE4_1<4, 16>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x4_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize8x4] = CflIntraPredictor_SSE4_1<8, 4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x8_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize8x8] = CflIntraPredictor_SSE4_1<8, 8>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x16_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize8x16] =
      CflIntraPredictor_SSE4_1<8, 16>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize8x32_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize8x32] =
      CflIntraPredictor_SSE4_1<8, 32>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x4_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize16x4] =
      CflIntraPredictor_SSE4_1<16, 4>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x8_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize16x8] =
      CflIntraPredictor_SSE4_1<16, 8>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x16_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize16x16] =
      CflIntraPredictor_SSE4_1<16, 16>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize16x32_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize16x32] =
      CflIntraPredictor_SSE4_1<16, 32>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize32x8_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize32x8] =
      CflIntraPredictor_SSE4_1<32, 8>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize32x16_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize32x16] =
      CflIntraPredictor_SSE4_1<32, 16>;
#endif
#if DSP_ENABLED_8BPP_SSE4_1(TransformSize32x32_CflIntraPredictor)
  dsp->cfl_intra_predictors[kTransformSize32x32] =
      CflIntraPredictor_SSE4_1<32, 32>;
#endif
}

}  // namespace
}  // namespace low_bitdepth

void IntraPredCflInit_SSE4_1() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else  // !LIBGAV1_ENABLE_SSE4_1

namespace libgav1 {
namespace dsp {

void IntraPredCflInit_SSE4_1() {}

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_ENABLE_SSE4_1
