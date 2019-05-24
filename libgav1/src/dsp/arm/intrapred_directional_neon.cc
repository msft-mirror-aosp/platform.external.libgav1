#include "src/dsp/arm/intrapred_neon.h"
#include "src/dsp/dsp.h"

#if LIBGAV1_ENABLE_NEON

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>  // memset

#include "src/dsp/arm/common_neon.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

// Blend two values based on a 32 bit weight.
inline uint8x8_t WeightedBlend(const uint8x8_t a, const uint8x8_t b,
                               const uint8x8_t a_weight,
                               const uint8x8_t b_weight) {
  const uint16x8_t a_product = vmull_u8(a, a_weight);
  const uint16x8_t b_product = vmull_u8(b, b_weight);

  return vrshrn_n_u16(vaddq_u16(a_product, b_product), 5);
}

// For vertical operations the weights are one constant value.
inline uint8x8_t WeightedBlend(const uint8x8_t a, const uint8x8_t b,
                               const uint8_t weight) {
  const uint16x8_t a_product = vmull_u8(a, vdup_n_u8(32 - weight));
  const uint16x8_t b_product = vmull_u8(b, vdup_n_u8(weight));

  return vrshrn_n_u16(vaddq_u16(a_product, b_product), 5);
}

void DirectionalIntraPredictorZone1_NEON(void* const dest,
                                         const ptrdiff_t stride,
                                         const void* const top_row,
                                         const int width, const int height,
                                         const int xstep,
                                         const bool upsampled_top) {
  const uint8_t* const top = static_cast<const uint8_t*>(top_row);
  uint8_t* dst = static_cast<uint8_t*>(dest);

  assert(xstep > 0);

  const int upsample_shift = static_cast<int>(upsampled_top);
  const int scale_bits = 6 - upsample_shift;

  const uint8x8_t all = vcreate_u8(0x0706050403020100);
  const uint8x8_t even = vcreate_u8(0x0e0c0a0806040200);
  const uint8x8_t base_step = upsampled_top ? even : all;

  if (xstep == 64) {
    assert(!upsampled_top);
    const uint8_t* top_ptr = top + 1;
    for (int y = 0; y < height; y += 4) {
      memcpy(dst, top_ptr, width);
      memcpy(dst + stride, top_ptr + 1, width);
      memcpy(dst + 2 * stride, top_ptr + 2, width);
      memcpy(dst + 3 * stride, top_ptr + 3, width);
      dst += 4 * stride;
      top_ptr += 4;
    }
  } else if (width == 4) {
    const int max_base_x = ((width + height) - 1) << upsample_shift;
    const uint8x8_t max_base = vdup_n_u8(max_base_x);
    const uint8x8_t top_max_base = vdup_n_u8(top[max_base_x]);

    for (int y = 0, top_x = xstep; y < height;
         ++y, dst += stride, top_x += xstep) {
      const int top_base_x = top_x >> scale_bits;

      if (top_base_x >= max_base_x) {
        for (int i = y; i < height; ++i) {
          memset(dst, top[max_base_x], width);
          dst += stride;
        }
        return;
      }

      const uint8_t shift = ((top_x << upsample_shift) & 0x3F) >> 1;
      const uint8x8_t shift_mul = vdup_n_u8(shift);
      const uint8x8_t inv_shift_mul = vdup_n_u8(32 - shift);

      const uint8x8_t base_v = vadd_u8(vdup_n_u8(top_base_x), base_step);

      const uint8x8_t max_base_mask = vclt_u8(base_v, max_base);

      // Load 8 values because we will extract the output values based on
      // |upsampled_top| at the end.
      const uint8x8_t left_values = vld1_u8(top + top_base_x);
      const uint8x8_t right_values = RightShift<8>(left_values);

      const uint8x8_t value =
          WeightedBlend(left_values, right_values, inv_shift_mul, shift_mul);

      // If |upsampled_top| is true then extract every other value for output.
      const uint8x8_t value_stepped = vtbl1_u8(value, base_step);

      const uint8x8_t masked_value =
          vbsl_u8(max_base_mask, value_stepped, top_max_base);

      StoreLo4(dst, masked_value);
    }
  } else if (xstep > 51) {
    // 7.11.2.10. Intra edge upsample selection process
    // if ( d <= 0 || d >= 40 ) useUpsample = 0
    // For |upsample_top| the delta is from vertical so |prediction_angle - 90|.
    // In |kDirectionalIntraPredictorDerivative[]| angles less than 51 will meet
    // this criteria. The |xstep| value for angle 51 happens to be 51 as well.
    // Shallower angles have greater xstep values.
    assert(!upsampled_top);
    const int max_base_x = ((width + height) - 1);
    const uint8x8_t max_base = vdup_n_u8(max_base_x);
    const uint8x8_t top_max_base = vdup_n_u8(top[max_base_x]);
    const uint8x8_t block_step = vdup_n_u8(8);

    for (int y = 0, top_x = xstep; y < height;
         ++y, dst += stride, top_x += xstep) {
      const int top_base_x = top_x >> 6;

      if (top_base_x >= max_base_x) {
        for (int i = y; i < height; ++i) {
          memset(dst, top[max_base_x], width);
          dst += stride;
        }
        return;
      }

      const uint8_t shift = ((top_x << upsample_shift) & 0x3F) >> 1;
      const uint8x8_t shift_mul = vdup_n_u8(shift);
      const uint8x8_t inv_shift_mul = vdup_n_u8(32 - shift);

      uint8x8_t base_v = vadd_u8(vdup_n_u8(top_base_x), base_step);

      for (int x = 0; x < width; x += 8) {
        const uint8x8_t max_base_mask = vclt_u8(base_v, max_base);

        // Since these |xstep| values can not be upsampled the load is
        // simplified.
        const uint8x8_t left_values = vld1_u8(top + top_base_x + x);
        const uint8x8_t right_values = vld1_u8(top + top_base_x + x + 1);

        const uint8x8_t value =
            WeightedBlend(left_values, right_values, inv_shift_mul, shift_mul);

        const uint8x8_t masked_value =
            vbsl_u8(max_base_mask, value, top_max_base);

        vst1_u8(dst + x, masked_value);

        base_v = vadd_u8(base_v, block_step);
      }
    }
  } else {
    const int max_base_x = ((width + height) - 1) << upsample_shift;
    const uint8x8_t max_base = vdup_n_u8(max_base_x);
    const uint8x8_t top_max_base = vdup_n_u8(top[max_base_x]);
    const uint8x8_t right_step = vadd_u8(base_step, vdup_n_u8(1));
    const uint8x8_t block_step = vdup_n_u8(8 << upsample_shift);

    for (int y = 0, top_x = xstep; y < height;
         ++y, dst += stride, top_x += xstep) {
      const int top_base_x = top_x >> scale_bits;

      if (top_base_x >= max_base_x) {
        for (int i = y; i < height; ++i) {
          memset(dst, top[max_base_x], width);
          dst += stride;
        }
        return;
      }

      const uint8_t shift = ((top_x << upsample_shift) & 0x3F) >> 1;
      const uint8x8_t shift_mul = vdup_n_u8(shift);
      const uint8x8_t inv_shift_mul = vdup_n_u8(32 - shift);

      uint8x8_t base_v = vadd_u8(vdup_n_u8(top_base_x), base_step);

      for (int x = 0; x < width; x += 8) {
        const uint8x8_t max_base_mask = vclt_u8(base_v, max_base);

        // Extract the input values based on |upsampled_top| here to avoid doing
        // twice as many calculations.
        const uint8x16_t mixed_values = vld1q_u8(top + top_base_x + x);
        const uint8x8_t left_values = vtbl2_u8(
            {vget_low_u8(mixed_values), vget_high_u8(mixed_values)}, base_step);
        const uint8x8_t right_values =
            vtbl2_u8({vget_low_u8(mixed_values), vget_high_u8(mixed_values)},
                     right_step);

        const uint8x8_t value =
            WeightedBlend(left_values, right_values, inv_shift_mul, shift_mul);

        const uint8x8_t masked_value =
            vbsl_u8(max_base_mask, value, top_max_base);

        vst1_u8(dst + x, masked_value);

        base_v = vadd_u8(base_v, block_step);
      }
    }
  }
}

// Fill |left| and |right| with the appropriate values for a given |base_step|.
inline void LoadStepwise(const uint8_t* const source, const uint8x8_t left_step,
                         const uint8x8_t right_step, uint8x8_t* left,
                         uint8x8_t* right) {
  const uint8x16_t mixed = vld1q_u8(source);
  *left = vtbl2_u8({vget_low_u8(mixed), vget_high_u8(mixed)}, left_step);
  *right = vtbl2_u8({vget_low_u8(mixed), vget_high_u8(mixed)}, right_step);
}

void DirectionalIntraPredictorZone3_NEON(void* const dest,
                                         const ptrdiff_t stride,
                                         const void* const left_column,
                                         const int width, const int height,
                                         const int ystep,
                                         const bool upsampled_left) {
  const auto* const left = static_cast<const uint8_t*>(left_column);

  assert(ystep > 0);

  const int upsample_shift = static_cast<int>(upsampled_left);
  const int scale_bits = 6 - upsample_shift;
  const int base_step = 1 << upsample_shift;

  if (width == 4 || height == 4) {
    // This block can handle all sizes but the specializations for other sizes
    // are faster.
    const uint8x8_t all = vcreate_u8(0x0706050403020100);
    const uint8x8_t even = vcreate_u8(0x0e0c0a0806040200);
    const uint8x8_t base_step_v = upsampled_left ? even : all;
    const uint8x8_t right_step = vadd_u8(base_step_v, vdup_n_u8(1));

    for (int y = 0; y < height; y += 8) {
      for (int x = 0; x < width; x += 4) {
        uint8_t* dst = static_cast<uint8_t*>(dest);
        dst += y * stride + x;
        uint8x8_t left_v[4], right_v[4], value_v[4];
        const int ystep_base = ystep * x;
        const int offset = y * base_step;

        const int index_0 = ystep_base + ystep * 1;
        LoadStepwise(left + offset + (index_0 >> scale_bits), base_step_v,
                     right_step, &left_v[0], &right_v[0]);
        value_v[0] = WeightedBlend(left_v[0], right_v[0],
                                   ((index_0 << upsample_shift) & 0x3F) >> 1);

        const int index_1 = ystep_base + ystep * 2;
        LoadStepwise(left + offset + (index_1 >> scale_bits), base_step_v,
                     right_step, &left_v[1], &right_v[1]);
        value_v[1] = WeightedBlend(left_v[1], right_v[1],
                                   ((index_1 << upsample_shift) & 0x3F) >> 1);

        const int index_2 = ystep_base + ystep * 3;
        LoadStepwise(left + offset + (index_2 >> scale_bits), base_step_v,
                     right_step, &left_v[2], &right_v[2]);
        value_v[2] = WeightedBlend(left_v[2], right_v[2],
                                   ((index_2 << upsample_shift) & 0x3F) >> 1);

        const int index_3 = ystep_base + ystep * 4;
        LoadStepwise(left + offset + (index_3 >> scale_bits), base_step_v,
                     right_step, &left_v[3], &right_v[3]);
        value_v[3] = WeightedBlend(left_v[3], right_v[3],
                                   ((index_3 << upsample_shift) & 0x3F) >> 1);

        // 8x4 transpose.
        const uint8x8x2_t b0 = vtrn_u8(value_v[0], value_v[1]);
        const uint8x8x2_t b1 = vtrn_u8(value_v[2], value_v[3]);

        const uint16x4x2_t c0 = vtrn_u16(vreinterpret_u16_u8(b0.val[0]),
                                         vreinterpret_u16_u8(b1.val[0]));
        const uint16x4x2_t c1 = vtrn_u16(vreinterpret_u16_u8(b0.val[1]),
                                         vreinterpret_u16_u8(b1.val[1]));

        StoreLo4(dst, vreinterpret_u8_u16(c0.val[0]));
        dst += stride;
        StoreLo4(dst, vreinterpret_u8_u16(c1.val[0]));
        dst += stride;
        StoreLo4(dst, vreinterpret_u8_u16(c0.val[1]));
        dst += stride;
        StoreLo4(dst, vreinterpret_u8_u16(c1.val[1]));

        if (height > 4) {
          dst += stride;
          StoreHi4(dst, vreinterpret_u8_u16(c0.val[0]));
          dst += stride;
          StoreHi4(dst, vreinterpret_u8_u16(c1.val[0]));
          dst += stride;
          StoreHi4(dst, vreinterpret_u8_u16(c0.val[1]));
          dst += stride;
          StoreHi4(dst, vreinterpret_u8_u16(c1.val[1]));
        }
      }
    }
  } else {  // 8x8 at a time.
    // Limited improvement for 8x8. ~20% faster for 64x64.
    const uint8x8_t all = vcreate_u8(0x0706050403020100);
    const uint8x8_t even = vcreate_u8(0x0e0c0a0806040200);
    const uint8x8_t base_step_v = upsampled_left ? even : all;
    const uint8x8_t right_step = vadd_u8(base_step_v, vdup_n_u8(1));

    for (int y = 0; y < height; y += 8) {
      for (int x = 0; x < width; x += 8) {
        uint8_t* dst = static_cast<uint8_t*>(dest);
        dst += y * stride + x;
        uint8x8_t left_v[8], right_v[8], value_v[8];
        const int ystep_base = ystep * x;
        const int offset = y * base_step;

        const int index_0 = ystep_base + ystep * 1;
        LoadStepwise(left + offset + (index_0 >> scale_bits), base_step_v,
                     right_step, &left_v[0], &right_v[0]);
        value_v[0] = WeightedBlend(left_v[0], right_v[0],
                                   ((index_0 << upsample_shift) & 0x3F) >> 1);

        const int index_1 = ystep_base + ystep * 2;
        LoadStepwise(left + offset + (index_1 >> scale_bits), base_step_v,
                     right_step, &left_v[1], &right_v[1]);
        value_v[1] = WeightedBlend(left_v[1], right_v[1],
                                   ((index_1 << upsample_shift) & 0x3F) >> 1);

        const int index_2 = ystep_base + ystep * 3;
        LoadStepwise(left + offset + (index_2 >> scale_bits), base_step_v,
                     right_step, &left_v[2], &right_v[2]);
        value_v[2] = WeightedBlend(left_v[2], right_v[2],
                                   ((index_2 << upsample_shift) & 0x3F) >> 1);

        const int index_3 = ystep_base + ystep * 4;
        LoadStepwise(left + offset + (index_3 >> scale_bits), base_step_v,
                     right_step, &left_v[3], &right_v[3]);
        value_v[3] = WeightedBlend(left_v[3], right_v[3],
                                   ((index_3 << upsample_shift) & 0x3F) >> 1);

        const int index_4 = ystep_base + ystep * 5;
        LoadStepwise(left + offset + (index_4 >> scale_bits), base_step_v,
                     right_step, &left_v[4], &right_v[4]);
        value_v[4] = WeightedBlend(left_v[4], right_v[4],
                                   ((index_4 << upsample_shift) & 0x3F) >> 1);

        const int index_5 = ystep_base + ystep * 6;
        LoadStepwise(left + offset + (index_5 >> scale_bits), base_step_v,
                     right_step, &left_v[5], &right_v[5]);
        value_v[5] = WeightedBlend(left_v[5], right_v[5],
                                   ((index_5 << upsample_shift) & 0x3F) >> 1);

        const int index_6 = ystep_base + ystep * 7;
        LoadStepwise(left + offset + (index_6 >> scale_bits), base_step_v,
                     right_step, &left_v[6], &right_v[6]);
        value_v[6] = WeightedBlend(left_v[6], right_v[6],
                                   ((index_6 << upsample_shift) & 0x3F) >> 1);

        const int index_7 = ystep_base + ystep * 8;
        LoadStepwise(left + offset + (index_7 >> scale_bits), base_step_v,
                     right_step, &left_v[7], &right_v[7]);
        value_v[7] = WeightedBlend(left_v[7], right_v[7],
                                   ((index_7 << upsample_shift) & 0x3F) >> 1);

        // 8x8 transpose.
        const uint8x16x2_t b0 = vtrnq_u8(vcombine_u8(value_v[0], value_v[4]),
                                         vcombine_u8(value_v[1], value_v[5]));
        const uint8x16x2_t b1 = vtrnq_u8(vcombine_u8(value_v[2], value_v[6]),
                                         vcombine_u8(value_v[3], value_v[7]));

        const uint16x8x2_t c0 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[0]),
                                          vreinterpretq_u16_u8(b1.val[0]));
        const uint16x8x2_t c1 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[1]),
                                          vreinterpretq_u16_u8(b1.val[1]));

        const uint32x4x2_t d0 = vuzpq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                          vreinterpretq_u32_u16(c1.val[0]));
        const uint32x4x2_t d1 = vuzpq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                          vreinterpretq_u32_u16(c1.val[1]));

        vst1_u8(dst, vreinterpret_u8_u32(vget_low_u32(d0.val[0])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_high_u32(d0.val[0])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_low_u32(d1.val[0])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_high_u32(d1.val[0])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_low_u32(d0.val[1])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_high_u32(d0.val[1])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_low_u32(d1.val[1])));
        dst += stride;
        vst1_u8(dst, vreinterpret_u8_u32(vget_high_u32(d1.val[1])));
      }
    }
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
  dsp->directional_intra_predictor_zone1 = DirectionalIntraPredictorZone1_NEON;
  dsp->directional_intra_predictor_zone3 = DirectionalIntraPredictorZone3_NEON;
}

}  // namespace
}  // namespace low_bitdepth

void IntraPredDirectionalInit_NEON() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !LIBGAV1_ENABLE_NEON
namespace libgav1 {
namespace dsp {

void IntraPredDirectionalInit_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON
