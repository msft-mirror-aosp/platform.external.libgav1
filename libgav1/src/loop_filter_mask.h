#ifndef LIBGAV1_SRC_LOOP_FILTER_MASK_H_
#define LIBGAV1_SRC_LOOP_FILTER_MASK_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/obu_parser.h"
#include "src/utils/array_2d.h"
#include "src/utils/block_parameters_holder.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/segmentation.h"
#include "src/utils/types.h"

namespace libgav1 {

class LoopFilterMask {
 public:
  // This structure holds loop filter bit masks for a 64x64 block.
  // 64x64 block contains kNum4x4In64x64 = (64x64 / (4x4) = 256)
  // 4x4 blocks. It requires kNumLoopFilterMasks = 4 uint64_t to represent them.
  struct Data : public Allocable {
    uint8_t level[kMaxPlanes][kNumLoopFilterTypes][kNum4x4In64x64];
    uint64_t left[kMaxPlanes][kNumTransformSizesLoopFilter]
                 [kNumLoopFilterMasks];
    uint64_t top[kMaxPlanes][kNumTransformSizesLoopFilter][kNumLoopFilterMasks];
  };

  LoopFilterMask() = default;

  // Loop filter mask is built and used for each superblock individually.
  // Thus not copyable/movable.
  LoopFilterMask(const LoopFilterMask&) = delete;
  LoopFilterMask& operator=(const LoopFilterMask&) = delete;
  LoopFilterMask(LoopFilterMask&&) = delete;
  LoopFilterMask& operator=(LoopFilterMask&&) = delete;

  // Allocates the loop filter masks for the given |width| and
  // |height| if necessary and zeros out the appropriate number of
  // entries. Returns true on success.
  bool Reset(int width, int height);

  // Builds bit masks for tile boundaries.
  // This function is called after the frame has been decoded so that
  // information across tiles is available.
  // Before this function call, bit masks of transform edges other than those
  // on tile boundaries are built together with tile decoding, in
  // Tile::BuildBitMask().
  bool Build(const ObuSequenceHeader& sequence_header,
             const ObuFrameHeader& frame_header, int tile_group_start,
             int tile_group_end, BlockParametersHolder* block_parameters_holder,
             const Array2D<TransformSize>& inter_transform_sizes);

  uint8_t GetLevel(int mask_id, int plane, LoopFilterType type,
                   int offset) const {
    return loop_filter_masks_[mask_id].level[plane][type][offset];
  }

  uint64_t GetLeft(int mask_id, int plane, int tx_size_id, int index) const {
    return loop_filter_masks_[mask_id].left[plane][tx_size_id][index];
  }

  uint64_t GetTop(int mask_id, int plane, int tx_size_id, int index) const {
    return loop_filter_masks_[mask_id].top[plane][tx_size_id][index];
  }

  int num_64x64_blocks_per_row() const { return num_64x64_blocks_per_row_; }

  void SetLeft(uint64_t new_mask, int mask_id, int plane, int transform_size_id,
               int index) {
    loop_filter_masks_[mask_id].left[plane][transform_size_id][index] |=
        new_mask;
  }

  void SetTop(uint64_t new_mask, int mask_id, int plane, int transform_size_id,
              int index) {
    loop_filter_masks_[mask_id].top[plane][transform_size_id][index] |=
        new_mask;
  }

  void SetLevel(uint8_t level, int mask_id, int plane, LoopFilterType type,
                int offset) {
    loop_filter_masks_[mask_id].level[plane][type][offset] = level;
  }

  static int GetIndex(int row4x4) { return row4x4 >> 2; }

  static int GetShift(int row4x4, int column4x4) {
    return ((row4x4 & 3) << 4) | column4x4;
  }

  static int GetLevelOffset(int row4x4, int column4x4) {
    assert(row4x4 < 16);
    assert(column4x4 < 16);
    return (row4x4 << 4) | column4x4;
  }

  // 7.14.5.
  static uint8_t GetDeblockFilterLevel(const ObuFrameHeader& frame_header,
                                       const BlockParameters& bp, Plane plane,
                                       int pass,
                                       const int8_t delta_lf[kFrameLfCount]) {
    const int filter_level_delta = (plane == kPlaneY) ? pass : plane + 1;
    const int delta = frame_header.delta_lf.multi ? delta_lf[filter_level_delta]
                                                  : delta_lf[0];
    // TODO(chengchen): Could we reduce number of clips?
    int level =
        Clip3(frame_header.loop_filter.level[filter_level_delta] + delta, 0,
              kMaxLoopFilterValue);
    const auto feature = static_cast<SegmentFeature>(
        kSegmentFeatureLoopFilterYVertical + filter_level_delta);
    if (frame_header.segmentation.FeatureActive(bp.segment_id, feature)) {
      level = Clip3(
          level +
              frame_header.segmentation.feature_data[bp.segment_id][feature],
          0, kMaxLoopFilterValue);
    }
    if (frame_header.loop_filter.delta_enabled) {
      const int shift = level >> 5;
      if (bp.reference_frame[0] == kReferenceFrameIntra) {
        level += LeftShift(
            frame_header.loop_filter.ref_deltas[kReferenceFrameIntra], shift);
      } else {
        const int mode_id = kPredictionModeDeltasLookup[bp.y_mode];
        level += LeftShift(
            frame_header.loop_filter.ref_deltas[bp.reference_frame[0]] +
                frame_header.loop_filter.mode_deltas[mode_id],
            shift);
      }
      level = Clip3(level, 0, kMaxLoopFilterValue);
    }
    return level;
  }

  bool IsValid(int mask_id) const;

 private:
  std::unique_ptr<Data[]> loop_filter_masks_;
  int num_64x64_blocks_ = -1;
  int num_64x64_blocks_per_row_;
  int num_64x64_blocks_per_column_;
};

}  // namespace libgav1

#endif  // LIBGAV1_SRC_LOOP_FILTER_MASK_H_
