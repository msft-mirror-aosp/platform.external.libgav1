#ifndef LIBGAV1_SRC_DSP_LOOP_RESTORATION_H_
#define LIBGAV1_SRC_DSP_LOOP_RESTORATION_H_

#include "src/dsp/arm/loop_restoration_neon.h"
#include "src/dsp/x86/loop_restoration_sse4.h"

namespace libgav1 {
namespace dsp {

// Initializes Dsp::loop_restorations with base implementations. This function
// is not thread-safe.
void LoopRestorationInit_C();

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_SRC_DSP_LOOP_RESTORATION_H_
