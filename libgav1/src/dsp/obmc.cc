#include "src/dsp/obmc.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace {

// 7.11.3.10.
template <typename Pixel>
void OverlapBlending_C(void* const prediction,
                       const ptrdiff_t prediction_stride, const int width,
                       const int height, const int blending_direction,
                       const uint8_t* const mask,
                       const void* const obmc_prediction,
                       const ptrdiff_t obmc_prediction_stride) {
  // 0 == kBlendFromAbove, 1 == kBlendFromLeft.
  assert(blending_direction == 0 || blending_direction == 1);
  auto* pred = static_cast<Pixel*>(prediction);
  const ptrdiff_t pred_stride = prediction_stride / sizeof(Pixel);
  const auto* obmc_pred = static_cast<const Pixel*>(obmc_prediction);
  const ptrdiff_t obmc_pred_stride = obmc_prediction_stride / sizeof(Pixel);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const uint8_t mask_value = (blending_direction == 0) ? mask[y] : mask[x];
      pred[x] = static_cast<Pixel>(RightShiftWithRounding(
          mask_value * pred[x] + (64 - mask_value) * obmc_pred[x], 6));
    }
    pred += pred_stride;
    obmc_pred += obmc_pred_stride;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
  dsp->obmc_blend = OverlapBlending_C<uint8_t>;
}

#if LIBGAV1_MAX_BITDEPTH >= 10
void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(10);
  assert(dsp != nullptr);
  dsp->obmc_blend = OverlapBlending_C<uint16_t>;
}
#endif

}  // namespace

void ObmcInit_C() {
  Init8bpp();
#if LIBGAV1_MAX_BITDEPTH >= 10
  Init10bpp();
#endif
}

}  // namespace dsp
}  // namespace libgav1
