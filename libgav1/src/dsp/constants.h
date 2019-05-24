#ifndef LIBGAV1_SRC_DSP_CONSTANTS_H_
#define LIBGAV1_SRC_DSP_CONSTANTS_H_

// This file contains DSP related constants that have a direct relationship with
// a DSP component.

#include <cstdint>

#include "src/utils/constants.h"

namespace libgav1 {

enum {
  // Weights are quadratic from '1' to '1 / block_size', scaled by
  // 2^kSmoothWeightScale.
  kSmoothWeightScale = 8,
  kCflLumaBufferStride = 32,
};  // anonymous enum

extern const int8_t kFilterIntraTaps[kNumFilterIntraPredictors][8][8];

// Values in this enum can be derived as the sum of subsampling_x and
// subsampling_y (since subsampling_x == 0 && subsampling_y == 1 case is never
// allowed by the bitstream).
enum SubsamplingType : uint8_t {
  kSubsamplingType444,  // subsampling_x = 0, subsampling_y = 0.
  kSubsamplingType422,  // subsampling_x = 1, subsampling_y = 0.
  kSubsamplingType420,  // subsampling_x = 1, subsampling_y = 1.
  kNumSubsamplingTypes
};

extern const uint16_t kSgrScaleParameter[16][2];

}  // namespace libgav1

#endif  // LIBGAV1_SRC_DSP_CONSTANTS_H_
