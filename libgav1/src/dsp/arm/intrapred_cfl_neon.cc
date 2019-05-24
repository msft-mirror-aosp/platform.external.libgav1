#include "src/dsp/arm/intrapred_neon.h"
#include "src/dsp/dsp.h"

#if LIBGAV1_ENABLE_NEON

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/arm/common_neon.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

// Saturate |dc + ((alpha * luma) >> 6))| to uint8_t.
inline uint8x8_t Combine8(const int16x8_t luma, const int alpha,
                          const int16x8_t dc) {
  const int16x8_t la = vmulq_n_s16(luma, alpha);
  // Subtract the sign bit to round towards zero.
  const int16x8_t sub_sign = vsubq_s16(
      la, vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(la), 15)));
  // Shift and accumulate.
  const int16x8_t result = vrsraq_n_s16(dc, sub_sign, 6);
  return vqmovun_s16(result);
}

// The range of luma/alpha is not really important because it gets saturated to
// uint8_t. Saturated int16_t >> 6 outranges uint8_t.
template <int block_height>
inline void CflIntraPredictor4xN_NEON(
    void* const dest, const ptrdiff_t stride,
    const int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int alpha) {
  auto* dst = static_cast<uint8_t*>(dest);
  const int16x8_t dc = vdupq_n_s16(dst[0]);
  for (int y = 0; y < block_height; y += 2) {
    const int16x4_t luma_row0 = vld1_s16(luma[y]);
    const int16x4_t luma_row1 = vld1_s16(luma[y + 1]);
    const uint8x8_t sum =
        Combine8(vcombine_s16(luma_row0, luma_row1), alpha, dc);
    StoreLo4(dst, sum);
    dst += stride;
    StoreHi4(dst, sum);
    dst += stride;
  }
}

template <int block_height>
inline void CflIntraPredictor8xN_NEON(
    void* const dest, const ptrdiff_t stride,
    const int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int alpha) {
  auto* dst = static_cast<uint8_t*>(dest);
  const int16x8_t dc = vdupq_n_s16(dst[0]);
  for (int y = 0; y < block_height; ++y) {
    const int16x8_t luma_row = vld1q_s16(luma[y]);
    const uint8x8_t sum = Combine8(luma_row, alpha, dc);
    vst1_u8(dst, sum);
    dst += stride;
  }
}

template <int block_height>
inline void CflIntraPredictor16xN_NEON(
    void* const dest, const ptrdiff_t stride,
    const int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int alpha) {
  auto* dst = static_cast<uint8_t*>(dest);
  const int16x8_t dc = vdupq_n_s16(dst[0]);
  for (int y = 0; y < block_height; ++y) {
    const int16x8_t luma_row_0 = vld1q_s16(luma[y]);
    const int16x8_t luma_row_1 = vld1q_s16(luma[y] + 8);
    const uint8x8_t sum_0 = Combine8(luma_row_0, alpha, dc);
    const uint8x8_t sum_1 = Combine8(luma_row_1, alpha, dc);
    vst1_u8(dst, sum_0);
    vst1_u8(dst + 8, sum_1);
    dst += stride;
  }
}

template <int block_height>
inline void CflIntraPredictor32xN_NEON(
    void* const dest, const ptrdiff_t stride,
    const int16_t luma[kCflLumaBufferStride][kCflLumaBufferStride],
    const int alpha) {
  auto* dst = static_cast<uint8_t*>(dest);
  const int16x8_t dc = vdupq_n_s16(dst[0]);
  for (int y = 0; y < block_height; ++y) {
    const int16x8_t luma_row_0 = vld1q_s16(luma[y]);
    const int16x8_t luma_row_1 = vld1q_s16(luma[y] + 8);
    const int16x8_t luma_row_2 = vld1q_s16(luma[y] + 16);
    const int16x8_t luma_row_3 = vld1q_s16(luma[y] + 24);
    const uint8x8_t sum_0 = Combine8(luma_row_0, alpha, dc);
    const uint8x8_t sum_1 = Combine8(luma_row_1, alpha, dc);
    const uint8x8_t sum_2 = Combine8(luma_row_2, alpha, dc);
    const uint8x8_t sum_3 = Combine8(luma_row_3, alpha, dc);
    vst1_u8(dst, sum_0);
    vst1_u8(dst + 8, sum_1);
    vst1_u8(dst + 16, sum_2);
    vst1_u8(dst + 24, sum_3);
    dst += stride;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);

  dsp->cfl_intra_predictors[kTransformSize4x4] = CflIntraPredictor4xN_NEON<4>;
  dsp->cfl_intra_predictors[kTransformSize4x8] = CflIntraPredictor4xN_NEON<8>;
  dsp->cfl_intra_predictors[kTransformSize4x16] = CflIntraPredictor4xN_NEON<16>;

  dsp->cfl_intra_predictors[kTransformSize8x4] = CflIntraPredictor8xN_NEON<4>;
  dsp->cfl_intra_predictors[kTransformSize8x8] = CflIntraPredictor8xN_NEON<8>;
  dsp->cfl_intra_predictors[kTransformSize8x16] = CflIntraPredictor8xN_NEON<16>;
  dsp->cfl_intra_predictors[kTransformSize8x32] = CflIntraPredictor8xN_NEON<32>;

  dsp->cfl_intra_predictors[kTransformSize16x4] = CflIntraPredictor16xN_NEON<4>;
  dsp->cfl_intra_predictors[kTransformSize16x8] = CflIntraPredictor16xN_NEON<8>;
  dsp->cfl_intra_predictors[kTransformSize16x16] =
      CflIntraPredictor16xN_NEON<16>;
  dsp->cfl_intra_predictors[kTransformSize16x32] =
      CflIntraPredictor16xN_NEON<32>;

  dsp->cfl_intra_predictors[kTransformSize32x8] = CflIntraPredictor32xN_NEON<8>;
  dsp->cfl_intra_predictors[kTransformSize32x16] =
      CflIntraPredictor32xN_NEON<16>;
  dsp->cfl_intra_predictors[kTransformSize32x32] =
      CflIntraPredictor32xN_NEON<32>;
  // Max Cfl predictor size is 32x32.
}

}  // namespace
}  // namespace low_bitdepth

void IntraPredCflInit_NEON() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !LIBGAV1_ENABLE_NEON
namespace libgav1 {
namespace dsp {

void IntraPredCflInit_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON
