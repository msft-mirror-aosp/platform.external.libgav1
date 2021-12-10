/*
 * Copyright 2019 The libgav1 Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LIBGAV1_SRC_DSP_ARM_LOOP_FILTER_NEON_H_
#define LIBGAV1_SRC_DSP_ARM_LOOP_FILTER_NEON_H_

#include "src/dsp/dsp.h"
#include "src/utils/cpu.h"

namespace libgav1 {
namespace dsp {

// Initializes Dsp::loop_filters, see the defines below for specifics. This
// function is not thread-safe.
void LoopFilterInit_NEON();

}  // namespace dsp
}  // namespace libgav1

#if LIBGAV1_ENABLE_NEON

#define LIBGAV1_Dsp8bpp_LoopFilterSize4_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp8bpp_LoopFilterSize4_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp8bpp_LoopFilterSize6_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp8bpp_LoopFilterSize6_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp8bpp_LoopFilterSize8_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp8bpp_LoopFilterSize8_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp8bpp_LoopFilterSize14_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp8bpp_LoopFilterSize14_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp10bpp_LoopFilterSize4_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp10bpp_LoopFilterSize4_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp10bpp_LoopFilterSize6_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp10bpp_LoopFilterSize6_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp10bpp_LoopFilterSize8_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp10bpp_LoopFilterSize8_LoopFilterTypeVertical LIBGAV1_CPU_NEON

#define LIBGAV1_Dsp10bpp_LoopFilterSize14_LoopFilterTypeHorizontal \
  LIBGAV1_CPU_NEON
#define LIBGAV1_Dsp10bpp_LoopFilterSize14_LoopFilterTypeVertical \
  LIBGAV1_CPU_NEON

#endif  // LIBGAV1_ENABLE_NEON

#endif  // LIBGAV1_SRC_DSP_ARM_LOOP_FILTER_NEON_H_
