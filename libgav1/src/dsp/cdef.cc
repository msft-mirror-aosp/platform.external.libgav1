#include "src/dsp/cdef.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/dsp/dsp.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace {

constexpr int16_t kDivisionTable[] = {0,   840, 420, 280, 210,
                                      168, 140, 120, 105};

constexpr uint8_t kPrimaryTaps[2][2] = {{4, 2}, {3, 3}};

constexpr uint8_t kSecondaryTaps[2][2] = {{2, 1}, {2, 1}};

constexpr int8_t kCdefDirections[8][2][2] = {
    {{-1, 1}, {-2, 2}}, {{0, 1}, {-1, 2}}, {{0, 1}, {0, 2}}, {{0, 1}, {1, 2}},
    {{1, 1}, {2, 2}},   {{1, 0}, {2, 1}},  {{1, 0}, {2, 0}}, {{1, 0}, {2, -1}}};

int Constrain(int diff, int threshold, int damping) {
  if (threshold == 0) return 0;
  damping = std::max(0, damping - FloorLog2(threshold));
  const int sign = (diff < 0) ? -1 : 1;
  return sign *
         Clip3(threshold - (std::abs(diff) >> damping), 0, std::abs(diff));
}

// 5.11.52.
bool InsideFrame(int x, int y, int subsampling_x, int subsampling_y,
                 int rows4x4, int columns4x4) {
  const int row = DivideBy4(LeftShift(y, subsampling_y));
  const int column = DivideBy4(LeftShift(x, subsampling_x));
  return row >= 0 && row < rows4x4 && column >= 0 && column < columns4x4;
}

int32_t Square(int32_t x) { return x * x; }

template <int bitdepth, typename Pixel>
void CdefDirection_C(const void* const source, ptrdiff_t stride,
                     int* const direction, int* const variance) {
  assert(direction != nullptr);
  assert(variance != nullptr);
  const auto* src = static_cast<const Pixel*>(source);
  stride /= sizeof(Pixel);
  int32_t cost[8] = {};
  // |partial| does not have to be int32_t for 8bpp. int16_t will suffice. We
  // use int32_t to keep it simple since |cost| will have to be int32_t.
  int32_t partial[8][15] = {};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int x = (src[j] >> (bitdepth - 8)) - 128;
      partial[0][i + j] += x;
      partial[1][i + j / 2] += x;
      partial[2][i] += x;
      partial[3][3 + i - j / 2] += x;
      partial[4][7 + i - j] += x;
      partial[5][3 - i / 2 + j] += x;
      partial[6][j] += x;
      partial[7][i / 2 + j] += x;
    }
    src += stride;
  }
  for (int i = 0; i < 8; ++i) {
    cost[2] += Square(partial[2][i]);
    cost[6] += Square(partial[6][i]);
  }
  cost[2] *= kDivisionTable[8];
  cost[6] *= kDivisionTable[8];
  for (int i = 0; i < 7; ++i) {
    cost[0] += (Square(partial[0][i]) + Square(partial[0][14 - i])) *
               kDivisionTable[i + 1];
    cost[4] += (Square(partial[4][i]) + Square(partial[4][14 - i])) *
               kDivisionTable[i + 1];
  }
  cost[0] += Square(partial[0][7]) * kDivisionTable[8];
  cost[4] += Square(partial[4][7]) * kDivisionTable[8];
  for (int i = 1; i < 8; i += 2) {
    for (int j = 0; j < 5; ++j) {
      cost[i] += Square(partial[i][3 + j]);
    }
    cost[i] *= kDivisionTable[8];
    for (int j = 0; j < 3; ++j) {
      cost[i] += (Square(partial[i][j]) + Square(partial[i][10 - j])) *
                 kDivisionTable[2 * j + 2];
    }
  }
  int32_t best_cost = 0;
  *direction = 0;
  for (int i = 0; i < 8; ++i) {
    if (cost[i] > best_cost) {
      best_cost = cost[i];
      *direction = i;
    }
  }
  *variance = (best_cost - cost[(*direction + 4) & 7]) >> 10;
}

template <int bitdepth, typename Pixel>
void CdefFiltering_C(const void* const source, const ptrdiff_t source_stride,
                     const int rows4x4, const int columns4x4, const int curr_x,
                     const int curr_y, const int subsampling_x,
                     const int subsampling_y, const int primary_strength,
                     const int secondary_strength, const int damping,
                     const int direction, void* const dest,
                     const ptrdiff_t dest_stride) {
  const int coeff_shift = bitdepth - 8;
  const int plane_width = MultiplyBy4(columns4x4) >> subsampling_x;
  const int plane_height = MultiplyBy4(rows4x4) >> subsampling_y;
  const int block_width = std::min(8 >> subsampling_x, plane_width - curr_x);
  const int block_height = std::min(8 >> subsampling_y, plane_height - curr_y);
  const auto* src = static_cast<const Pixel*>(source);
  const ptrdiff_t src_stride = source_stride / sizeof(Pixel);
  auto* dst = static_cast<Pixel*>(dest);
  const ptrdiff_t dst_stride = dest_stride / sizeof(Pixel);
  for (int y = 0; y < block_height; ++y) {
    for (int x = 0; x < block_width; ++x) {
      int16_t sum = 0;
      const Pixel pixel_value = src[x];
      Pixel max_value = pixel_value;
      Pixel min_value = pixel_value;
      for (int k = 0; k < 2; ++k) {
        const int signs[] = {-1, 1};
        for (const int& sign : signs) {
          int dy = sign * kCdefDirections[direction][k][0];
          int dx = sign * kCdefDirections[direction][k][1];
          int y0 = curr_y + y + dy;
          int x0 = curr_x + x + dx;
          // TODO(chengchen): Optimize cdef data fetching.
          // Cdef needs to get pixel values from 3x3 neighborhood.
          // It could happen that the target position is out of the frame.
          // When it's out of frame, that pixel should not be taken into
          // calculation.
          // In libaom's implementation, borders are padded around the whole
          // frame such that out of frame access gets a large value. The
          // large value is defined as 30000. This implementation has a problem
          // because 8-bit input can't represent 30000. It has to allocate a
          // 16-bit frame buffer to set large values for the borders.
          // In this implementation, we detect whether it's out of frame,
          // which is not friendly for SIMD implementation.
          // We can avoid the extra frame buffer by allocating a 16-bit block
          // buffer, like the implementation of loop restoration.
          if (InsideFrame(x0, y0, subsampling_x, subsampling_y, rows4x4,
                          columns4x4)) {
            const Pixel value = src[dy * src_stride + dx + x];
            sum += Constrain(value - pixel_value, primary_strength, damping) *
                   kPrimaryTaps[(primary_strength >> coeff_shift) & 1][k];
            max_value = std::max(value, max_value);
            min_value = std::min(value, min_value);
          }
          const int offsets[] = {-2, 2};
          for (const int& offset : offsets) {
            dy = sign * kCdefDirections[(direction + offset) & 7][k][0];
            dx = sign * kCdefDirections[(direction + offset) & 7][k][1];
            y0 = curr_y + y + dy;
            x0 = curr_x + x + dx;
            if (InsideFrame(x0, y0, subsampling_x, subsampling_y, rows4x4,
                            columns4x4)) {
              const Pixel value = src[dy * src_stride + dx + x];
              sum +=
                  Constrain(value - pixel_value, secondary_strength, damping) *
                  kSecondaryTaps[(primary_strength >> coeff_shift) & 1][k];
              max_value = std::max(value, max_value);
              min_value = std::min(value, min_value);
            }
          }
        }
      }

      dst[x] = Clip3(pixel_value + ((8 + sum - (sum < 0)) >> 4), min_value,
                     max_value);
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
#if LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
  dsp->cdef_direction = CdefDirection_C<8, uint8_t>;
  dsp->cdef_filter = CdefFiltering_C<8, uint8_t>;
#else  // !LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
#ifndef LIBGAV1_Dsp8bpp_CdefDirection
  dsp->cdef_direction = CdefDirection_C<8, uint8_t>;
  dsp->cdef_filter = CdefFiltering_C<8, uint8_t>;
#endif
#endif  // LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
}

#if LIBGAV1_MAX_BITDEPTH >= 10
void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(10);
  assert(dsp != nullptr);
#if LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
  dsp->cdef_direction = CdefDirection_C<10, uint16_t>;
  dsp->cdef_filter = CdefFiltering_C<10, uint16_t>;
#else  // !LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
#ifndef LIBGAV1_Dsp10bpp_CdefDirection
  dsp->cdef_direction = CdefDirection_C<10, uint16_t>;
  dsp->cdef_filter = CdefFiltering_C<10, uint16_t>;
#endif
#endif  // LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
}
#endif

}  // namespace

void CdefInit_C() {
  Init8bpp();
#if LIBGAV1_MAX_BITDEPTH >= 10
  Init10bpp();
#endif
}

}  // namespace dsp
}  // namespace libgav1
