#include "src/dsp/warp.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

// Number of extra bits of precision in warped filtering.
constexpr int kWarpedDiffPrecisionBits = 10;

template <int bitdepth, typename Pixel>
void Warp_C(const void* const source, ptrdiff_t source_stride,
            const int source_width, const int source_height,
            const int* const warp_params, const int subsampling_x,
            const int subsampling_y, const uint8_t inter_round_bits[2],
            const int block_start_x, const int block_start_y,
            const int block_width, const int block_height, const int16_t alpha,
            const int16_t beta, const int16_t gamma, const int16_t delta,
            uint16_t* dest, const ptrdiff_t dest_stride) {
  // Intermediate_result is the output of the horizontal filtering.
  // The range is within 16 bits.
  uint16_t intermediate_result[15][8];  // 15 rows, 8 columns.
  const int horizontal_offset_bits = bitdepth + kFilterBits - 1;
  const int vertical_offset_bits =
      bitdepth + 2 * kFilterBits - inter_round_bits[0];
  const auto* const src = static_cast<const Pixel*>(source);
  source_stride /= sizeof(Pixel);

  // Warp process applies for each 8x8 block (or smaller).
  for (int start_y = block_start_y; start_y < block_start_y + block_height;
       start_y += 8) {
    for (int start_x = block_start_x; start_x < block_start_x + block_width;
         start_x += 8) {
      const int src_x = (start_x + 4) << subsampling_x;
      const int src_y = (start_y + 4) << subsampling_y;
      const int dst_x =
          src_x * warp_params[2] + src_y * warp_params[3] + warp_params[0];
      const int dst_y =
          src_x * warp_params[4] + src_y * warp_params[5] + warp_params[1];
      const int x4 = dst_x >> subsampling_x;
      const int y4 = dst_y >> subsampling_y;
      const int ix4 = x4 >> kWarpedModelPrecisionBits;
      const int sx4 = x4 & ((1 << kWarpedModelPrecisionBits) - 1);
      const int iy4 = y4 >> kWarpedModelPrecisionBits;
      const int sy4 = y4 & ((1 << kWarpedModelPrecisionBits) - 1);

      // Horizontal filter.
      for (int y = -7; y < 8; ++y) {
        // TODO(chenghchen):
        // Because of warping, the index could be out of frame boundary. Thus
        // clip is needed. However, can we remove or reduce usage of clip?
        // Besides, special cases exist, for example,
        // if iy4 - 7 >= source_height, there's no need to do the filtering.
        const int row = Clip3(iy4 + y, 0, source_height - 1);
        const Pixel* const src_row = src + row * source_stride;
        for (int x = -4; x < 4; ++x) {
          const int sx = sx4 + alpha * x + beta * y;
          const int offset =
              RightShiftWithRounding(sx, kWarpedDiffPrecisionBits) +
              kWarpedPixelPrecisionShifts;
          // For SIMD optimization:
          // For 8 bit, the range of sum is within uint16_t, if we add an
          // horizontal offset:
          int sum = 1 << horizontal_offset_bits;
          // Horizontal_offset guarantees sum is non negative.
          // If horizontal_offset is used, intermediate_result needs to be
          // uint16_t.
          // For 10/12 bit, the range of sum is within 32 bits.
          for (int k = 0; k < 8; ++k) {
            const int column = Clip3(ix4 + x + k - 3, 0, source_width - 1);
            sum += kWarpedFilters[offset][k] * src_row[column];
          }
          assert(sum >= 0 && sum < (1 << (horizontal_offset_bits + 2)));
          intermediate_result[y + 7][x + 4] = static_cast<uint16_t>(
              RightShiftWithRounding(sum, inter_round_bits[0]));
        }
      }

      // Vertical filter.
      uint16_t* dst_row = dest + start_x - block_start_x;
      for (int y = -4;
           y < std::min(4, block_start_y + block_height - start_y - 4); ++y) {
        for (int x = -4;
             x < std::min(4, block_start_x + block_width - start_x - 4); ++x) {
          const int sy = sy4 + gamma * x + delta * y;
          const int offset =
              RightShiftWithRounding(sy, kWarpedDiffPrecisionBits) +
              kWarpedPixelPrecisionShifts;
          // Similar to horizontal_offset, vertical_offset guarantees sum
          // before shifting is non negative:
          int sum = 1 << vertical_offset_bits;
          for (int k = 0; k < 8; ++k) {
            sum += kWarpedFilters[offset][k] *
                   intermediate_result[y + k + 4][x + 4];
          }
          assert(sum >= 0 && sum < (1 << (vertical_offset_bits + 2)));
          sum = RightShiftWithRounding(sum, inter_round_bits[1]);
          // Warp output is a predictor, whose type is uint16_t.
          // Do not clip it here. The clipping is applied at the stage of
          // final pixel value output.
          dst_row[x + 4] = static_cast<uint16_t>(sum);
        }
        dst_row += dest_stride;
      }
    }
    dest += 8 * dest_stride;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
  dsp->warp = Warp_C<8, uint8_t>;
}

#if LIBGAV1_MAX_BITDEPTH >= 10
void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(10);
  assert(dsp != nullptr);
  dsp->warp = Warp_C<10, uint16_t>;
}
#endif

}  // namespace

void WarpInit_C() {
  Init8bpp();
#if LIBGAV1_MAX_BITDEPTH >= 10
  Init10bpp();
#endif
}

}  // namespace dsp
}  // namespace libgav1
