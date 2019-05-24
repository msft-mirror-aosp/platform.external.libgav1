#ifndef LIBGAV1_SRC_DSP_INTRAPRED_H_
#define LIBGAV1_SRC_DSP_INTRAPRED_H_

// Pull in LIBGAV1_DspXXX defines representing the implementation status
// of each function. The resulting value of each can be used by each module to
// determine whether an implementation is needed at compile time. The order of
// includes is important as each tests for a superior version before setting
// the base.

#include "src/dsp/arm/intrapred_neon.h"
#include "src/dsp/x86/intrapred_sse4.h"

namespace libgav1 {
namespace dsp {

// Initializes Dsp::intra_predictors and Dsp::filter_intra_predictor with base
// implementations. This function is not thread-safe.
void IntraPredInit_C();

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_SRC_DSP_INTRAPRED_H_
