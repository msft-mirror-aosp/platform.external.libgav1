#ifndef LIBGAV1_SRC_DSP_ARM_LOOP_RESTORATION_NEON_H_
#define LIBGAV1_SRC_DSP_ARM_LOOP_RESTORATION_NEON_H_

#include "src/dsp/cpu.h"
#include "src/dsp/dsp.h"
#include "src/dsp/loop_filter.h"

namespace libgav1 {
namespace dsp {

// Initializes Dsp::loop_restorations with neon implementations. This function
// is not thread-safe.
void LoopRestorationInit_NEON();

}  // namespace dsp
}  // namespace libgav1

#if LIBGAV1_ENABLE_NEON

#define LIBGAV1_Dsp8bpp_SelfGuidedFilter LIBGAV1_DSP_NEON
#define LIBGAV1_Dsp8bpp_WienerFilter LIBGAV1_DSP_NEON

#endif  // LIBGAV1_ENABLE_NEON

#endif  // LIBGAV1_SRC_DSP_ARM_LOOP_RESTORATION_NEON_H_
