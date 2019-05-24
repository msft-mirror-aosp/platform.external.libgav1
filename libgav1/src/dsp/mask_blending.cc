#include "src/dsp/mask_blending.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace {

template <int bitdepth, typename Pixel>
void MaskBlending_C(
    const uint16_t* prediction_0, const ptrdiff_t prediction_stride_0,
    const uint16_t* prediction_1, const ptrdiff_t prediction_stride_1,
    const uint8_t* mask, const ptrdiff_t mask_stride, const int width,
    const int height, const int subsampling_x, const int subsampling_y,
    const bool is_inter_intra, const bool is_wedge_inter_intra,
    const int inter_post_round_bits, void* dest, const ptrdiff_t dest_stride) {
  auto* dst = static_cast<Pixel*>(dest);
  const ptrdiff_t dst_stride = dest_stride / sizeof(Pixel);
  const int step_y = subsampling_y ? 2 : 1;
  const int mask_step_y =
      (is_inter_intra && !is_wedge_inter_intra) ? 1 : step_y;
  const uint8_t* mask_next_row = mask + mask_stride;
  // An offset to cancel offsets used in single predictor generation that
  // make intermediate computations non negative.
  const int single_round_offset = (1 << bitdepth) + (1 << (bitdepth - 1));
  // An offset to cancel offsets used in compound predictor generation that
  // make intermediate computations non negative.
  const int compound_round_offset =
      (1 << (bitdepth + 4)) + (1 << (bitdepth + 3));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t mask_value;
      if (((subsampling_x | subsampling_y) == 0) ||
          (is_inter_intra && !is_wedge_inter_intra)) {
        mask_value = mask[x];
      } else if (subsampling_x == 1 && subsampling_y == 0) {
        mask_value = static_cast<uint8_t>(RightShiftWithRounding(
            mask[MultiplyBy2(x)] + mask[MultiplyBy2(x) + 1], 1));
      } else if (subsampling_x == 0 && subsampling_y == 1) {
        mask_value = static_cast<uint8_t>(
            RightShiftWithRounding(mask[x] + mask_next_row[x], 1));
      } else {
        mask_value = static_cast<uint8_t>(RightShiftWithRounding(
            mask[MultiplyBy2(x)] + mask[MultiplyBy2(x) + 1] +
                mask_next_row[MultiplyBy2(x)] +
                mask_next_row[MultiplyBy2(x) + 1],
            2));
      }

      if (is_inter_intra) {
        // In inter intra prediction mode, the intra prediction (prediction_1)
        // values are valid pixel values: [0, (1 << bitdepth) - 1].
        // While the inter prediction values come from subpixel prediction
        // from another frame, which involves interpolation and rounding.
        // Therefore prediction_0 has to be clipped.
        dst[x] = static_cast<Pixel>(RightShiftWithRounding(
            mask_value * prediction_1[x] +
                (64 - mask_value) * Clip3(prediction_0[x] - single_round_offset,
                                          0, (1 << bitdepth) - 1),
            6));
      } else {
        int res = (mask_value * prediction_0[x] +
                   (64 - mask_value) * prediction_1[x]) >>
                  6;
        res -= compound_round_offset;
        dst[x] = static_cast<Pixel>(
            Clip3(RightShiftWithRounding(res, inter_post_round_bits), 0,
                  (1 << bitdepth) - 1));
      }
    }
    dst += dst_stride;
    mask += mask_stride * mask_step_y;
    mask_next_row += mask_stride * step_y;
    prediction_0 += prediction_stride_0;
    prediction_1 += prediction_stride_1;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
  dsp->mask_blend = MaskBlending_C<8, uint8_t>;
}

#if LIBGAV1_MAX_BITDEPTH >= 10
void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(10);
  assert(dsp != nullptr);
  dsp->mask_blend = MaskBlending_C<10, uint16_t>;
}
#endif

}  // namespace

void MaskBlendingInit_C() {
  Init8bpp();
#if LIBGAV1_MAX_BITDEPTH >= 10
  Init10bpp();
#endif
}

}  // namespace dsp
}  // namespace libgav1
