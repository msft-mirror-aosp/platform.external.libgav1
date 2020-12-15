// Copyright 2020 The libgav1 Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/dsp/weight_mask.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/cpu.h"
#include "src/utils/memory.h"
#include "tests/third_party/libvpx/acm_random.h"
#include "tests/utils.h"

namespace libgav1 {
namespace dsp {
namespace {

constexpr int kNumSpeedTests = 50000;
constexpr int kMaxPredictionSize = 128;

const char* GetDigest8bpp(int id) {
  static const char* const kDigest[] = {
      "38612f8031608c5bac93fb8895369524",
      "bcac2573f67771f85f5f46e1bb54b1a9",
      "cad4fea48e4b9f6e6a8d064208b74041",
      "" /*kBlock16x4*/,
      "1a6542fab173873f0bc10e0dc8b4cf02",
      "2d26e918ee368ed1b7bf5df7cd8169ae",
      "4403df26ae363fcb6494e80025cf08a3",
      "281eb71746cf0b256c4acff65bf419d8",
      "7fab69156d6d4d87c8297c37ca4c1dad",
      "9f5cb7b3904135762e8b85d514fe4c12",
      "3a2deabd94e76b812e2755a38827e3c4",
      "a861b154179b98ec8defeec6cc8bb6cb",
      "9088c68e89a80f2da0b3dcceaa1097d6",
      "a0aef02974e1e100fd958e7deb07bee2",
      "b9f07f4300f8cb6aae92ce16e8d30b75",
      "c443ab805597240cddc0bb38af45860f",
      "6a7216117e428ebb13d9e11f8c1e31a8",
      "c463a6eec677e24e0069ed5d7b33f287",

      // mask_is_inverse = true.
      "7642e2a5ff77b77d38c99e196c50080a",
      "07c7cbe5200db76657eba8c3608cf861",
      "ef83d1d598cae50fe0101d7d2fc4b165",
      "" /*kBlock16x4*/,
      "2c100309913583a52d3b8993a8ea1269",
      "fc35db47c6e62a1c7d510dc7cd7fa07b",
      "986b7d8906d1af573cb3fc39ec00662b",
      "27361ab074d4a03a1523f96581ed6d28",
      "2bd0dc7977b1084c9a47ef0a44fef3b6",
      "94b8b7774fe52f040a492f333ad5ccf7",
      "18aefe53f097bf6387047ae014ea74eb",
      "8dabaa9364db5e699c6a6c9028fdf5f1",
      "ccbea5db7091cb6de386729c703f2b32",
      "508fa4c464224b997962ba84ea865f76",
      "5e86b08d53e094d3f8988e0fb2cbdff3",
      "7ac71c7b22bc8fe072c370f927e6bce8",
      "e9ea26ea2f0bc076622763710a86e1a2",
      "2047f322011c891a6086601a44e8bea1",
  };
  return kDigest[id];
}

#if LIBGAV1_MAX_BITDEPTH >= 10
const char* GetDigest10bpp(int id) {
  static const char* const kDigest[] = {
      "38612f8031608c5bac93fb8895369524",
      "bcac2573f67771f85f5f46e1bb54b1a9",
      "cad4fea48e4b9f6e6a8d064208b74041",
      "" /*kBlock16x4*/,
      "1a6542fab173873f0bc10e0dc8b4cf02",
      "2d26e918ee368ed1b7bf5df7cd8169ae",
      "4403df26ae363fcb6494e80025cf08a3",
      "281eb71746cf0b256c4acff65bf419d8",
      "7fab69156d6d4d87c8297c37ca4c1dad",
      "9f5cb7b3904135762e8b85d514fe4c12",
      "3a2deabd94e76b812e2755a38827e3c4",
      "44dcc5b27a316c46f7aa7b0411172164",
      "9088c68e89a80f2da0b3dcceaa1097d6",
      "f3cfb00ba4e4843e8bb695449410bd70",
      "6d21e77659c3a725afee1a189c7ca7be",
      "ecaccea3dd268c7ae7ba8b8aa4124360",
      "fe901e763b1ace9a3bca63c8537f37ee",
      "26b4feee26c40242598a49816a35a932",

      // mask_is_inverse = true.
      "7642e2a5ff77b77d38c99e196c50080a",
      "07c7cbe5200db76657eba8c3608cf861",
      "ef83d1d598cae50fe0101d7d2fc4b165",
      "" /*kBlock4x16*/,
      "2c100309913583a52d3b8993a8ea1269",
      "fc35db47c6e62a1c7d510dc7cd7fa07b",
      "986b7d8906d1af573cb3fc39ec00662b",
      "27361ab074d4a03a1523f96581ed6d28",
      "2bd0dc7977b1084c9a47ef0a44fef3b6",
      "94b8b7774fe52f040a492f333ad5ccf7",
      "18aefe53f097bf6387047ae014ea74eb",
      "f08920bdaacc5ba703b35cb0e6ce1479",
      "ccbea5db7091cb6de386729c703f2b32",
      "cd50c06e62d00b313a761d46f490a849",
      "1ed97f8ae0ab711687dd3d6adc9e1625",
      "cae5c363df44009f305ed85eb0ff9f8a",
      "5a665e46f7871382e7d1a359285ea301",
      "60a6a5e927c6b494fe78fe4bbb22e2ac",
  };
  return kDigest[id];
}
#endif  // LIBGAV1_MAX_BITDEPTH >= 10

struct WeightMaskTestParam {
  WeightMaskTestParam(int width, int height, bool mask_is_inverse)
      : width(width), height(height), mask_is_inverse(mask_is_inverse) {}
  int width;
  int height;
  bool mask_is_inverse;
};

std::ostream& operator<<(std::ostream& os, const WeightMaskTestParam& param) {
  return os << param.width << "x" << param.height
            << ", mask_is_inverse: " << param.mask_is_inverse;
}

template <int bitdepth>
class WeightMaskTest : public ::testing::TestWithParam<WeightMaskTestParam>,
                       public test_utils::MaxAlignedAllocable {
 public:
  WeightMaskTest() = default;
  ~WeightMaskTest() override = default;

  void SetUp() override {
    test_utils::ResetDspTable(bitdepth);
    WeightMaskInit_C();
    const dsp::Dsp* const dsp = dsp::GetDspTable(bitdepth);
    ASSERT_NE(dsp, nullptr);
    const int width_index = FloorLog2(width_) - 3;
    const int height_index = FloorLog2(height_) - 3;
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const char* const test_case = test_info->test_suite_name();
    if (absl::StartsWith(test_case, "C/")) {
    } else if (absl::StartsWith(test_case, "NEON/")) {
      WeightMaskInit_NEON();
    } else if (absl::StartsWith(test_case, "SSE41/")) {
      WeightMaskInit_SSE4_1();
    }
    func_ = dsp->weight_mask[width_index][height_index][mask_is_inverse_];
  }

 protected:
  void SetInputData(bool use_fixed_values, int value_1, int value_2);
  void Test(int num_runs, bool use_fixed_values, int value_1, int value_2);

 private:
  const int width_ = GetParam().width;
  const int height_ = GetParam().height;
  const bool mask_is_inverse_ = GetParam().mask_is_inverse;
  alignas(
      kMaxAlignment) uint16_t block_1_[kMaxPredictionSize * kMaxPredictionSize];
  alignas(
      kMaxAlignment) uint16_t block_2_[kMaxPredictionSize * kMaxPredictionSize];
  uint8_t mask_[kMaxPredictionSize * kMaxPredictionSize] = {};
  dsp::WeightMaskFunc func_;
};

template <int bitdepth>
void WeightMaskTest<bitdepth>::SetInputData(const bool use_fixed_values,
                                            const int value_1,
                                            const int value_2) {
  if (use_fixed_values) {
    std::fill(block_1_, block_1_ + kMaxPredictionSize * kMaxPredictionSize,
              value_1);
    std::fill(block_2_, block_2_ + kMaxPredictionSize * kMaxPredictionSize,
              value_2);
  } else {
    libvpx_test::ACMRandom rnd(libvpx_test::ACMRandom::DeterministicSeed());
    const int mask = (1 << bitdepth) - 1;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        block_1_[y * width_ + x] = rnd.Rand16() & mask;
        block_2_[y * width_ + x] = rnd.Rand16() & mask;
      }
    }
  }
}

BlockSize DimensionsToBlockSize(int width, int height) {
  if (width == 4) {
    if (height == 4) return kBlock4x4;
    if (height == 8) return kBlock4x8;
    if (height == 16) return kBlock4x16;
    return kBlockInvalid;
  }
  if (width == 8) {
    if (height == 4) return kBlock8x4;
    if (height == 8) return kBlock8x8;
    if (height == 16) return kBlock8x16;
    if (height == 32) return kBlock8x32;
    return kBlockInvalid;
  }
  if (width == 16) {
    if (height == 4) return kBlock16x4;
    if (height == 8) return kBlock16x8;
    if (height == 16) return kBlock16x16;
    if (height == 32) return kBlock16x32;
    if (height == 64) return kBlock16x64;
    return kBlockInvalid;
  }
  if (width == 32) {
    if (height == 8) return kBlock32x8;
    if (height == 16) return kBlock32x16;
    if (height == 32) return kBlock32x32;
    if (height == 64) return kBlock32x64;
    return kBlockInvalid;
  }
  if (width == 64) {
    if (height == 16) return kBlock64x16;
    if (height == 32) return kBlock64x32;
    if (height == 64) return kBlock64x64;
    if (height == 128) return kBlock64x128;
    return kBlockInvalid;
  }
  if (width == 128) {
    if (height == 64) return kBlock128x64;
    if (height == 128) return kBlock128x128;
    return kBlockInvalid;
  }
  return kBlockInvalid;
}

template <int bitdepth>
void WeightMaskTest<bitdepth>::Test(const int num_runs,
                                    const bool use_fixed_values,
                                    const int value_1, const int value_2) {
  if (func_ == nullptr) return;
  SetInputData(use_fixed_values, value_1, value_2);
  const absl::Time start = absl::Now();
  for (int i = 0; i < num_runs; ++i) {
    func_(block_1_, block_2_, mask_, kMaxPredictionSize);
  }
  const absl::Duration elapsed_time = absl::Now() - start;
  if (use_fixed_values) {
    const int max_pixel_value = (1 << bitdepth) - 1;
    int fixed_value = 38;
    if ((value_1 == 0 && value_2 == max_pixel_value) ||
        (value_1 == max_pixel_value && value_2 == 0)) {
      fixed_value = 39;
    }
    if (mask_is_inverse_) fixed_value = 64 - fixed_value;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        EXPECT_EQ(static_cast<int>(mask_[y * kMaxPredictionSize + x]),
                  fixed_value);
      }
    }
  } else {
    const int id_offset = mask_is_inverse_ ? kMaxBlockSizes - 4 : 0;
    const int id = id_offset +
                   static_cast<int>(DimensionsToBlockSize(width_, height_)) - 4;
    if (bitdepth == 8) {
      test_utils::CheckMd5Digest(
          absl::StrFormat("BlockSize %dx%d", width_, height_).c_str(),
          "WeightMask", GetDigest8bpp(id), mask_, sizeof(mask_), elapsed_time);
#if LIBGAV1_MAX_BITDEPTH >= 10
    } else {
      test_utils::CheckMd5Digest(
          absl::StrFormat("BlockSize %dx%d", width_, height_).c_str(),
          "WeightMask", GetDigest10bpp(id), mask_, sizeof(mask_), elapsed_time);
#endif
    }
  }
}

const WeightMaskTestParam weight_mask_test_param[] = {
    WeightMaskTestParam(8, 8, false),     WeightMaskTestParam(8, 16, false),
    WeightMaskTestParam(8, 32, false),    WeightMaskTestParam(16, 8, false),
    WeightMaskTestParam(16, 16, false),   WeightMaskTestParam(16, 32, false),
    WeightMaskTestParam(16, 64, false),   WeightMaskTestParam(32, 8, false),
    WeightMaskTestParam(32, 16, false),   WeightMaskTestParam(32, 32, false),
    WeightMaskTestParam(32, 64, false),   WeightMaskTestParam(64, 16, false),
    WeightMaskTestParam(64, 32, false),   WeightMaskTestParam(64, 64, false),
    WeightMaskTestParam(64, 128, false),  WeightMaskTestParam(128, 64, false),
    WeightMaskTestParam(128, 128, false), WeightMaskTestParam(8, 8, true),
    WeightMaskTestParam(8, 16, true),     WeightMaskTestParam(8, 32, true),
    WeightMaskTestParam(16, 8, true),     WeightMaskTestParam(16, 16, true),
    WeightMaskTestParam(16, 32, true),    WeightMaskTestParam(16, 64, true),
    WeightMaskTestParam(32, 8, true),     WeightMaskTestParam(32, 16, true),
    WeightMaskTestParam(32, 32, true),    WeightMaskTestParam(32, 64, true),
    WeightMaskTestParam(64, 16, true),    WeightMaskTestParam(64, 32, true),
    WeightMaskTestParam(64, 64, true),    WeightMaskTestParam(64, 128, true),
    WeightMaskTestParam(128, 64, true),   WeightMaskTestParam(128, 128, true),
};

using WeightMaskTest8bpp = WeightMaskTest<8>;

TEST_P(WeightMaskTest8bpp, FixedValues) {
  Test(1, true, 0, 0);
  Test(1, true, 0, 255);
  Test(1, true, 255, 0);
  Test(1, true, 255, 255);
}

TEST_P(WeightMaskTest8bpp, RandomValues) { Test(1, false, -1, -1); }

TEST_P(WeightMaskTest8bpp, DISABLED_Speed) {
  Test(kNumSpeedTests, false, -1, -1);
}

INSTANTIATE_TEST_SUITE_P(C, WeightMaskTest8bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#if LIBGAV1_ENABLE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, WeightMaskTest8bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif
#if LIBGAV1_ENABLE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE41, WeightMaskTest8bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif

#if LIBGAV1_MAX_BITDEPTH >= 10
using WeightMaskTest10bpp = WeightMaskTest<10>;

TEST_P(WeightMaskTest10bpp, FixedValues) {
  Test(1, true, 0, 0);
  Test(1, true, 0, (1 << 10) - 1);
  Test(1, true, (1 << 10) - 1, 0);
  Test(1, true, (1 << 10) - 1, (1 << 10) - 1);
}

TEST_P(WeightMaskTest10bpp, RandomValues) { Test(1, false, -1, -1); }

TEST_P(WeightMaskTest10bpp, DISABLED_Speed) {
  Test(kNumSpeedTests, false, -1, -1);
}

INSTANTIATE_TEST_SUITE_P(C, WeightMaskTest10bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#if LIBGAV1_ENABLE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, WeightMaskTest10bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif
#if LIBGAV1_ENABLE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE41, WeightMaskTest10bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif
#endif  // LIBGAV1_MAX_BITDEPTH >= 10

}  // namespace
}  // namespace dsp
}  // namespace libgav1
