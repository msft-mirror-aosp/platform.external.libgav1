#ifndef LIBGAV1_SRC_DSP_ARM_COMMON_NEON_H_
#define LIBGAV1_SRC_DSP_ARM_COMMON_NEON_H_

#include "src/dsp/dsp.h"

#if LIBGAV1_ENABLE_NEON

#include <arm_neon.h>

#include <cstdint>
#include <cstring>

namespace libgav1 {
namespace dsp {

//------------------------------------------------------------------------------
// Load functions.

// Load 4 uint8_t values into the low half of a uint8x8_t register.
inline uint8x8_t LoadLo4(const uint8_t* const buf, uint8x8_t val) {
  uint32_t temp;
  memcpy(&temp, buf, 4);
  return vreinterpret_u8_u32(vld1_lane_u32(&temp, vreinterpret_u32_u8(val), 0));
}

// Load 4 uint8_t values into the high half of a uint8x8_t register.
inline uint8x8_t LoadHi4(const uint8_t* const buf, uint8x8_t val) {
  uint32_t temp;
  memcpy(&temp, buf, 4);
  return vreinterpret_u8_u32(vld1_lane_u32(&temp, vreinterpret_u32_u8(val), 1));
}

//------------------------------------------------------------------------------
// Store functions.

// Propagate type information to the compiler. Without this the compiler may
// assume the required alignment of uint32_t (4 bytes) and add alignment hints
// to the memory access.
inline void Uint32ToMem(uint8_t* const buf, uint32_t val) {
  memcpy(buf, &val, 4);
}

// Store 4 uint8_t values from the low half of a uint8x8_t register.
inline void StoreLo4(uint8_t* const buf, const uint8x8_t val) {
  Uint32ToMem(buf, vget_lane_u32(vreinterpret_u32_u8(val), 0));
}

// Store 4 uint8_t values from the high half of a uint8x8_t register.
inline void StoreHi4(uint8_t* const buf, const uint8x8_t val) {
  Uint32ToMem(buf, vget_lane_u32(vreinterpret_u32_u8(val), 1));
}

//------------------------------------------------------------------------------
// Bit manipulation.

// vshXX_n_XX() requires an immediate.
template <int shift>
inline uint8x8_t RightShift(const uint8x8_t vector) {
  return vreinterpret_u8_u64(vshr_n_u64(vreinterpret_u64_u8(vector), shift));
}

template <int shift>
inline int8x8_t RightShift(const int8x8_t vector) {
  return vreinterpret_s8_u64(vshr_n_u64(vreinterpret_u64_s8(vector), shift));
}

//------------------------------------------------------------------------------
// Transpose.

// Implement vtrnq_s64().
// Input:
// a0: 00 01 02 03 04 05 06 07
// a1: 16 17 18 19 20 21 22 23
// Output:
// b0.val[0]: 00 01 02 03 16 17 18 19
// b0.val[1]: 04 05 06 07 20 21 22 23
inline int16x8x2_t VtrnqS64(int32x4_t a0, int32x4_t a1) {
  int16x8x2_t b0;
  b0.val[0] = vcombine_s16(vreinterpret_s16_s32(vget_low_s32(a0)),
                           vreinterpret_s16_s32(vget_low_s32(a1)));
  b0.val[1] = vcombine_s16(vreinterpret_s16_s32(vget_high_s32(a0)),
                           vreinterpret_s16_s32(vget_high_s32(a1)));
  return b0;
}

// Input:
// a: 00 01 02 03 10 11 12 13
// b: 20 21 22 23 30 31 32 33
// Output:
// Note that columns [1] and [2] are transposed.
// a: 00 10 20 30 02 12 22 32
// b: 01 11 21 31 03 13 23 33
inline void Transpose4x4(uint8x8_t* a, uint8x8_t* b) {
  const uint16x4x2_t c =
      vtrn_u16(vreinterpret_u16_u8(*a), vreinterpret_u16_u8(*b));
  const uint32x2x2_t d =
      vtrn_u32(vreinterpret_u32_u16(c.val[0]), vreinterpret_u32_u16(c.val[1]));
  const uint8x8x2_t e =
      vtrn_u8(vreinterpret_u8_u32(d.val[0]), vreinterpret_u8_u32(d.val[1]));
  *a = e.val[0];
  *b = e.val[1];
}

// Reversible if the x4 values are packed next to each other.
// x4 input / x8 output:
// a0: 00 01 02 03 40 41 42 43 44
// a1: 10 11 12 13 50 51 52 53 54
// a2: 20 21 22 23 60 61 62 63 64
// a3: 30 31 32 33 70 71 72 73 74
// x8 input / x4 output:
// a0: 00 10 20 30 40 50 60 70
// a1: 01 11 21 31 41 51 61 71
// a2: 02 12 22 32 42 52 62 72
// a3: 03 13 23 33 43 53 63 73
inline void Transpose8x4(uint8x8_t* a0, uint8x8_t* a1, uint8x8_t* a2,
                         uint8x8_t* a3) {
  const uint8x8x2_t b0 = vtrn_u8(*a0, *a1);
  const uint8x8x2_t b1 = vtrn_u8(*a2, *a3);

  const uint16x4x2_t c0 =
      vtrn_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
  const uint16x4x2_t c1 =
      vtrn_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));

  *a0 = vreinterpret_u8_u16(c0.val[0]);
  *a1 = vreinterpret_u8_u16(c1.val[0]);
  *a2 = vreinterpret_u8_u16(c0.val[1]);
  *a3 = vreinterpret_u8_u16(c1.val[1]);
}

// Input:
// a0: 00 01 02 03 04 05 06 07
// a1: 10 11 12 13 14 15 16 17
// a2: 20 21 22 23 24 25 26 27
// a3: 30 31 32 33 34 35 36 37
// a4: 40 41 42 43 44 45 46 47
// a5: 50 51 52 53 54 55 56 57
// a6: 60 61 62 63 64 65 66 67
// a7: 70 71 72 73 74 75 76 77

// Output:
// a0: 00 10 20 30 40 50 60 70
// a1: 01 11 21 31 41 51 61 71
// a2: 02 12 22 32 42 52 62 72
// a3: 03 13 23 33 43 53 63 73
// a4: 04 14 24 34 44 54 64 74
// a5: 05 15 25 35 45 55 65 75
// a6: 06 16 26 36 46 56 66 76
// a7: 07 17 27 37 47 57 67 77
inline void Transpose8x8(int16x8_t* a0, int16x8_t* a1, int16x8_t* a2,
                         int16x8_t* a3, int16x8_t* a4, int16x8_t* a5,
                         int16x8_t* a6, int16x8_t* a7) {
  const int16x8x2_t b0 = vtrnq_s16(*a0, *a1);
  const int16x8x2_t b1 = vtrnq_s16(*a2, *a3);
  const int16x8x2_t b2 = vtrnq_s16(*a4, *a5);
  const int16x8x2_t b3 = vtrnq_s16(*a6, *a7);

  const int32x4x2_t c0 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[0]),
                                   vreinterpretq_s32_s16(b1.val[0]));
  const int32x4x2_t c1 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[1]),
                                   vreinterpretq_s32_s16(b1.val[1]));
  const int32x4x2_t c2 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[0]),
                                   vreinterpretq_s32_s16(b3.val[0]));
  const int32x4x2_t c3 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[1]),
                                   vreinterpretq_s32_s16(b3.val[1]));

  const int16x8x2_t d0 = VtrnqS64(c0.val[0], c2.val[0]);
  const int16x8x2_t d1 = VtrnqS64(c1.val[0], c3.val[0]);
  const int16x8x2_t d2 = VtrnqS64(c0.val[1], c2.val[1]);
  const int16x8x2_t d3 = VtrnqS64(c1.val[1], c3.val[1]);

  *a0 = d0.val[0];
  *a1 = d1.val[0];
  *a2 = d2.val[0];
  *a3 = d3.val[0];
  *a4 = d0.val[1];
  *a5 = d1.val[1];
  *a6 = d2.val[1];
  *a7 = d3.val[1];
}

// Input:
// i0: 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
// i1: 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
// i2: 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
// i3: 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
// i4: 40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
// i5: 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
// i6: 60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
// i7: 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f

// Output:
// o00: 00 10 20 30 40 50 60 70
// o01: 01 11 21 31 41 51 61 71
// o02: 02 12 22 32 42 52 62 72
// o03: 03 13 23 33 43 53 63 73
// o04: 04 14 24 34 44 54 64 74
// o05: 05 15 25 35 45 55 65 75
// o06: 06 16 26 36 46 56 66 76
// o07: 07 17 27 37 47 57 67 77
// o08: 08 18 28 38 48 58 68 78
// o09: 09 19 29 39 49 59 69 79
// o0a: 0a 1a 2a 3a 4a 5a 6a 7a
// o0b: 0b 1b 2b 3b 4b 5b 6b 7b
// o0c: 0c 1c 2c 3c 4c 5c 6c 7c
// o0d: 0d 1d 2d 3d 4d 5d 6d 7d
// o0e: 0e 1e 2e 3e 4e 5e 6e 7e
// o0f: 0f 1f 2f 3f 4f 5f 6f 7f
inline void Transpose16x8(const uint8x16_t i0, const uint8x16_t i1,
                          const uint8x16_t i2, const uint8x16_t i3,
                          const uint8x16_t i4, const uint8x16_t i5,
                          const uint8x16_t i6, const uint8x16_t i7,
                          uint8x8_t* o00, uint8x8_t* o01, uint8x8_t* o02,
                          uint8x8_t* o03, uint8x8_t* o04, uint8x8_t* o05,
                          uint8x8_t* o06, uint8x8_t* o07, uint8x8_t* o08,
                          uint8x8_t* o09, uint8x8_t* o10, uint8x8_t* o11,
                          uint8x8_t* o12, uint8x8_t* o13, uint8x8_t* o14,
                          uint8x8_t* o15) {
  const uint8x16x2_t b0 = vtrnq_u8(i0, i1);
  const uint8x16x2_t b1 = vtrnq_u8(i2, i3);
  const uint8x16x2_t b2 = vtrnq_u8(i4, i5);
  const uint8x16x2_t b3 = vtrnq_u8(i6, i7);

  const uint16x8x2_t c0 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[0]),
                                    vreinterpretq_u16_u8(b1.val[0]));
  const uint16x8x2_t c1 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[1]),
                                    vreinterpretq_u16_u8(b1.val[1]));
  const uint16x8x2_t c2 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[0]),
                                    vreinterpretq_u16_u8(b3.val[0]));
  const uint16x8x2_t c3 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[1]),
                                    vreinterpretq_u16_u8(b3.val[1]));

  const uint32x4x2_t d0 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                    vreinterpretq_u32_u16(c2.val[0]));
  const uint32x4x2_t d1 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                    vreinterpretq_u32_u16(c2.val[1]));
  const uint32x4x2_t d2 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[0]),
                                    vreinterpretq_u32_u16(c3.val[0]));
  const uint32x4x2_t d3 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[1]),
                                    vreinterpretq_u32_u16(c3.val[1]));

  *o00 = vget_low_u8(vreinterpretq_u8_u32(d0.val[0]));
  *o01 = vget_low_u8(vreinterpretq_u8_u32(d2.val[0]));
  *o02 = vget_low_u8(vreinterpretq_u8_u32(d1.val[0]));
  *o03 = vget_low_u8(vreinterpretq_u8_u32(d3.val[0]));
  *o04 = vget_low_u8(vreinterpretq_u8_u32(d0.val[1]));
  *o05 = vget_low_u8(vreinterpretq_u8_u32(d2.val[1]));
  *o06 = vget_low_u8(vreinterpretq_u8_u32(d1.val[1]));
  *o07 = vget_low_u8(vreinterpretq_u8_u32(d3.val[1]));
  *o08 = vget_high_u8(vreinterpretq_u8_u32(d0.val[0]));
  *o09 = vget_high_u8(vreinterpretq_u8_u32(d2.val[0]));
  *o10 = vget_high_u8(vreinterpretq_u8_u32(d1.val[0]));
  *o11 = vget_high_u8(vreinterpretq_u8_u32(d3.val[0]));
  *o12 = vget_high_u8(vreinterpretq_u8_u32(d0.val[1]));
  *o13 = vget_high_u8(vreinterpretq_u8_u32(d2.val[1]));
  *o14 = vget_high_u8(vreinterpretq_u8_u32(d1.val[1]));
  *o15 = vget_high_u8(vreinterpretq_u8_u32(d3.val[1]));
}

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_ENABLE_NEON
#endif  // LIBGAV1_SRC_DSP_ARM_COMMON_NEON_H_
