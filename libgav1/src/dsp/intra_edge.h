#ifndef LIBGAV1_SRC_DSP_INTRA_EDGE_H_
#define LIBGAV1_SRC_DSP_INTRA_EDGE_H_

namespace libgav1 {
namespace dsp {

// Initializes Dsp::intra_edge_filter and Dsp::intra_edge_upsampler. This
// function is not thread-safe.
void IntraEdgeInit_C();

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_SRC_DSP_INTRA_EDGE_H_
