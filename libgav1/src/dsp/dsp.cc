#include "src/dsp/dsp.h"

#include <mutex>  // NOLINT (unapproved c++11 header)

#include "src/dsp/arm/intrapred_neon.h"
#include "src/dsp/arm/loop_filter_neon.h"
#include "src/dsp/arm/loop_restoration_neon.h"
#include "src/dsp/average_blend.h"
#include "src/dsp/cdef.h"
#include "src/dsp/convolve.h"
#include "src/dsp/cpu.h"
#include "src/dsp/distance_weighted_blend.h"
#include "src/dsp/film_grain.h"
#include "src/dsp/intra_edge.h"
#include "src/dsp/intrapred.h"
#include "src/dsp/inverse_transform.h"
#include "src/dsp/loop_filter.h"
#include "src/dsp/loop_restoration.h"
#include "src/dsp/mask_blending.h"
#include "src/dsp/obmc.h"
#include "src/dsp/warp.h"
#include "src/dsp/x86/intra_edge_sse4.h"
#include "src/dsp/x86/intrapred_smooth_sse4.h"
#include "src/dsp/x86/intrapred_sse4.h"
#include "src/dsp/x86/inverse_transform_sse4.h"
#include "src/dsp/x86/loop_filter_sse4.h"
#include "src/dsp/x86/loop_restoration_sse4.h"

namespace libgav1 {
namespace dsp_internal {

dsp::Dsp* GetWritableDspTable(int bitdepth) {
  switch (bitdepth) {
    case 8: {
      static dsp::Dsp dsp_8bpp;
      return &dsp_8bpp;
    }
#if LIBGAV1_MAX_BITDEPTH >= 10
    case 10: {
      static dsp::Dsp dsp_10bpp;
      return &dsp_10bpp;
    }
#endif
  }
  return nullptr;
}

}  // namespace dsp_internal

namespace dsp {

void DspInit() {
  static std::once_flag once;
  std::call_once(once, []() {
    AverageBlendInit_C();
    CdefInit_C();
    ConvolveInit_C();
    DistanceWeightedBlendInit_C();
    FilmGrainInit_C();
    IntraEdgeInit_C();
    IntraPredInit_C();
    InverseTransformInit_C();
    LoopFilterInit_C();
    LoopRestorationInit_C();
    MaskBlendingInit_C();
    ObmcInit_C();
    WarpInit_C();
#if LIBGAV1_ENABLE_SSE4_1
    const uint32_t cpu_features = GetCpuInfo();
    if ((cpu_features & kSSE4_1) != 0) {
      IntraEdgeInit_SSE4_1();
      IntraPredInit_SSE4_1();
      IntraPredCflInit_SSE4_1();
      IntraPredSmoothInit_SSE4_1();
      InverseTransformInit_SSE4_1();
      LoopFilterInit_SSE4_1();
      LoopRestorationInit_SSE4_1();
    }
#endif  // LIBGAV1_ENABLE_SSE4_1
#if LIBGAV1_ENABLE_NEON
    IntraPredCflInit_NEON();
    IntraPredDirectionalInit_NEON();
    IntraPredFilterIntraInit_NEON();
    IntraPredInit_NEON();
    IntraPredSmoothInit_NEON();
    LoopFilterInit_NEON();
    LoopRestorationInit_NEON();
#endif  // LIBGAV1_ENABLE_NEON
  });
}

const Dsp* GetDspTable(int bitdepth) {
  return dsp_internal::GetWritableDspTable(bitdepth);
}

}  // namespace dsp
}  // namespace libgav1
