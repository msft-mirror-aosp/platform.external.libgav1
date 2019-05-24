#include "src/dsp/average_blend.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/dsp.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace {

template <int bitdepth, typename Pixel>
void AverageBlending_C(const uint16_t* prediction_0,
                       const ptrdiff_t prediction_stride_0,
                       const uint16_t* prediction_1,
                       const ptrdiff_t prediction_stride_1,
                       const int inter_post_round_bit, const int width,
                       const int height, void* const dest,
                       const ptrdiff_t dest_stride) {
  // An offset to cancel offsets used in compound predictor generation that
  // make intermediate computations non negative.
  const int compound_round_offset =
      (1 << (bitdepth + 4)) + (1 << (bitdepth + 3));
  auto* dst = static_cast<Pixel*>(dest);
  const ptrdiff_t dst_stride = dest_stride / sizeof(Pixel);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int res = (prediction_0[x] + prediction_1[x]) >> 1;
      res -= compound_round_offset;
      dst[x] = static_cast<Pixel>(
          Clip3(RightShiftWithRounding(res, inter_post_round_bit), 0,
                (1 << bitdepth) - 1));
    }
    dst += dst_stride;
    prediction_0 += prediction_stride_0;
    prediction_1 += prediction_stride_1;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
  dsp->average_blend = AverageBlending_C<8, uint8_t>;
}

#if LIBGAV1_MAX_BITDEPTH >= 10
void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(10);
  assert(dsp != nullptr);
  dsp->average_blend = AverageBlending_C<10, uint16_t>;
}
#endif

}  // namespace

void AverageBlendInit_C() {
  Init8bpp();
#if LIBGAV1_MAX_BITDEPTH >= 10
  Init10bpp();
#endif
}

}  // namespace dsp
}  // namespace libgav1
