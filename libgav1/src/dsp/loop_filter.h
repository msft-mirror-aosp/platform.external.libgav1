#ifndef LIBGAV1_SRC_DSP_LOOP_FILTER_H_
#define LIBGAV1_SRC_DSP_LOOP_FILTER_H_

// Pull in LIBGAV1_DspXXX defines representing the implementation status
// of each function. The resulting value of each can be used by each module to
// determine whether an implementation is needed at compile time. The order of
// includes is important as each tests for a superior version before setting
// the base.

#include "src/dsp/arm/loop_filter_neon.h"
#include "src/dsp/x86/loop_filter_sse4.h"

namespace libgav1 {
namespace dsp {

// Initializes Dsp::loop_filters with base implementations. This function
// is not thread-safe.
void LoopFilterInit_C();

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_SRC_DSP_LOOP_FILTER_H_
