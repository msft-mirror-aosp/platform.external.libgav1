#include "src/tile.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "src/motion_vector.h"
#include "src/reconstruction.h"
#include "src/utils/logging.h"
#include "src/utils/scan.h"
#include "src/utils/segmentation.h"

namespace libgav1 {
namespace {

// Import all the constants in the anonymous namespace.
#include "src/quantizer_tables.inc"

// Precision bits when scaling reference frames.
constexpr int kReferenceScaleShift = 14;
// Range above kNumQuantizerBaseLevels which the exponential golomb coding
// process is activated.
constexpr int kQuantizerCoefficientBaseRange = 12;
constexpr int kNumQuantizerBaseLevels = 2;
constexpr int kQuantizerCoefficientBaseRangeContextClamp =
    kQuantizerCoefficientBaseRange + kNumQuantizerBaseLevels + 1;
constexpr int kCoeffBaseRangeMaxIterations =
    kQuantizerCoefficientBaseRange / (kCoeffBaseRangeSymbolCount - 1);

const uint8_t kAllZeroContextsByMinMax[5][5] = {{1, 2, 2, 2, 3},
                                                {1, 4, 4, 4, 5},
                                                {1, 4, 4, 4, 5},
                                                {1, 4, 4, 4, 5},
                                                {1, 4, 4, 4, 6}};

// The space complexity of DFS is O(branching_factor * max_depth). For the
// parameter tree, branching_factor = 4 (there could be up to 4 children for
// every node) and max_depth = 6 (to go from a 128x128 block all the way to a
// 4x4 block).
constexpr int kDfsStackSize = 24;

// Mask indicating whether the transform sets contain a particular transform
// type. If |tx_type| is present in |tx_set|, then the |tx_type|th LSB is set.
constexpr uint16_t kTransformTypeInSetMask[kNumTransformSets] = {
    0x1, 0xE0F, 0x20F, 0xFFFF, 0xFFF, 0x201};

const PredictionMode
    kFilterIntraModeToIntraPredictor[kNumFilterIntraPredictors] = {
        kPredictionModeDc, kPredictionModeVertical, kPredictionModeHorizontal,
        kPredictionModeD157, kPredictionModeDc};

// This is computed as:
// min(transform_width_log2, 5) + min(transform_height_log2, 5) - 4.
const uint8_t kEobMultiSizeLookup[kNumTransformSizes] = {
    0, 1, 2, 1, 2, 3, 4, 2, 3, 4, 5, 5, 4, 5, 6, 6, 5, 6, 6};

/* clang-format off */
const uint8_t kCoeffBaseContextOffset[kNumTransformSizes][5][5] = {
    {{0, 1, 6, 6, 0}, {1, 6, 6, 21, 0}, {6, 6, 21, 21, 0}, {6, 21, 21, 21, 0},
     {0, 0, 0, 0, 0}},
    {{0, 11, 11, 11, 0}, {11, 11, 11, 11, 0}, {6, 6, 21, 21, 0},
     {6, 21, 21, 21, 0}, {21, 21, 21, 21, 0}},
    {{0, 11, 11, 11, 0}, {11, 11, 11, 11, 0}, {6, 6, 21, 21, 0},
     {6, 21, 21, 21, 0}, {21, 21, 21, 21, 0}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {0, 0, 0, 0, 0}},
    {{0, 1, 6, 6, 21}, {1, 6, 6, 21, 21}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 11, 11, 11, 11}, {11, 11, 11, 11, 11}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 11, 11, 11, 11}, {11, 11, 11, 11, 11}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {0, 0, 0, 0, 0}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {16, 16, 21, 21, 21}},
    {{0, 1, 6, 6, 21}, {1, 6, 6, 21, 21}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 11, 11, 11, 11}, {11, 11, 11, 11, 11}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 11, 11, 11, 11}, {11, 11, 11, 11, 11}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {16, 16, 21, 21, 21}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {16, 16, 21, 21, 21}},
    {{0, 1, 6, 6, 21}, {1, 6, 6, 21, 21}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 11, 11, 11, 11}, {11, 11, 11, 11, 11}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {16, 16, 21, 21, 21}},
    {{0, 16, 6, 6, 21}, {16, 16, 6, 21, 21}, {16, 16, 21, 21, 21},
     {16, 16, 21, 21, 21}, {16, 16, 21, 21, 21}},
    {{0, 1, 6, 6, 21}, {1, 6, 6, 21, 21}, {6, 6, 21, 21, 21},
     {6, 21, 21, 21, 21}, {21, 21, 21, 21, 21}}};
/* clang-format on */

const uint8_t kCoeffBasePositionContextOffset[3] = {26, 31, 36};

const PredictionMode kInterIntraToIntraMode[kNumInterIntraModes] = {
    kPredictionModeDc, kPredictionModeVertical, kPredictionModeHorizontal,
    kPredictionModeSmooth};

// Number of horizontal luma samples before intra block copy can be used.
constexpr int kIntraBlockCopyDelayPixels = 256;
// Number of 64 by 64 blocks before intra block copy can be used.
constexpr int kIntraBlockCopyDelay64x64Blocks = kIntraBlockCopyDelayPixels / 64;

// Index [i][j] corresponds to the transform size of width 1 << (i + 2) and
// height 1 << (j + 2).
constexpr TransformSize k4x4SizeToTransformSize[5][5] = {
    {kTransformSize4x4, kTransformSize4x8, kTransformSize4x16,
     kNumTransformSizes, kNumTransformSizes},
    {kTransformSize8x4, kTransformSize8x8, kTransformSize8x16,
     kTransformSize8x32, kNumTransformSizes},
    {kTransformSize16x4, kTransformSize16x8, kTransformSize16x16,
     kTransformSize16x32, kTransformSize16x64},
    {kNumTransformSizes, kTransformSize32x8, kTransformSize32x16,
     kTransformSize32x32, kTransformSize32x64},
    {kNumTransformSizes, kNumTransformSizes, kTransformSize64x16,
     kTransformSize64x32, kTransformSize64x64}};

// Maps compound prediction modes into single modes. For e.g.
// kPredictionModeNearestNewMv will map to kPredictionModeNearestMv for index 0
// and kPredictionModeNewMv for index 1. It is used to simplify the logic in
// AssignMv (and avoid duplicate code). This is section 5.11.30. in the spec.
inline PredictionMode GetSinglePredictionMode(int index,
                                              PredictionMode y_mode) {
  if (index == 0) {
    if (y_mode < kPredictionModeNearestNearestMv) {
      return y_mode;
    }
    if (y_mode == kPredictionModeNewNewMv ||
        y_mode == kPredictionModeNewNearestMv ||
        y_mode == kPredictionModeNewNearMv) {
      return kPredictionModeNewMv;
    }
    if (y_mode == kPredictionModeNearestNearestMv ||
        y_mode == kPredictionModeNearestNewMv) {
      return kPredictionModeNearestMv;
    }
    if (y_mode == kPredictionModeNearNearMv ||
        y_mode == kPredictionModeNearNewMv) {
      return kPredictionModeNearMv;
    }
    return kPredictionModeGlobalMv;
  }
  if (y_mode == kPredictionModeNewNewMv ||
      y_mode == kPredictionModeNearestNewMv ||
      y_mode == kPredictionModeNearNewMv) {
    return kPredictionModeNewMv;
  }
  if (y_mode == kPredictionModeNearestNearestMv ||
      y_mode == kPredictionModeNewNearestMv) {
    return kPredictionModeNearestMv;
  }
  if (y_mode == kPredictionModeNearNearMv ||
      y_mode == kPredictionModeNewNearMv) {
    return kPredictionModeNearMv;
  }
  return kPredictionModeGlobalMv;
}

// log2(dqDenom) in section 7.12.3 of the spec. We use the log2 value because
// dqDenom is always a power of two and hence right shift can be used instead of
// division.
int GetQuantizationShift(TransformSize tx_size) {
  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  const int max_tx_dimension = std::max(tx_width, tx_height);
  const int min_tx_dimension = std::min(tx_width, tx_height);
  if (max_tx_dimension == 64 && min_tx_dimension >= 32) {
    return 2;
  }
  if (max_tx_dimension >= 32 && min_tx_dimension >= 16) {
    return 1;
  }
  return 0;
}

// Input: 1d array index |index|, which indexes into a 2d array of width
//     1 << |tx_width_log2|.
// Output: 1d array index which indexes into a 2d array of width
//     (1 << |tx_width_log2|) + kQuantizedCoefficientBufferPadding.
int PaddedIndex(int index, int tx_width_log2) {
  return index + MultiplyBy4(index >> tx_width_log2);
}

}  // namespace

Tile::Tile(
    int tile_number, const uint8_t* const data, size_t size,
    const ObuSequenceHeader& sequence_header,
    const ObuFrameHeader& frame_header, RefCountedBuffer* const current_frame,
    const std::array<bool, kNumReferenceFrameTypes>& reference_frame_sign_bias,
    const std::array<RefCountedBufferPtr, kNumReferenceFrameTypes>&
        reference_frames,
    Array2D<TemporalMotionVector>* const motion_field_mv,
    const std::array<uint8_t, kNumReferenceFrameTypes>& reference_order_hint,
    const std::array<uint8_t, kWedgeMaskSize>& wedge_masks,
    const SymbolDecoderContext& symbol_decoder_context,
    SymbolDecoderContext* const saved_symbol_decoder_context,
    const SegmentationMap* prev_segment_ids, PostFilter* const post_filter,
    BlockParametersHolder* const block_parameters_holder,
    Array2D<int16_t>* const cdef_index,
    Array2D<TransformSize>* const inter_transform_sizes,
    const dsp::Dsp* const dsp, ThreadPool* const thread_pool,
    ResidualBufferPool* const residual_buffer_pool)
    : number_(tile_number),
      data_(data),
      size_(size),
      read_deltas_(false),
      current_quantizer_index_(frame_header.quantizer.base_index),
      sequence_header_(sequence_header),
      frame_header_(frame_header),
      current_frame_(*current_frame),
      reference_frame_sign_bias_(reference_frame_sign_bias),
      reference_frames_(reference_frames),
      motion_field_mv_(motion_field_mv),
      reference_order_hint_(reference_order_hint),
      wedge_masks_(wedge_masks),
      reader_(data_, size_, frame_header_.enable_cdf_update),
      symbol_decoder_context_(symbol_decoder_context),
      saved_symbol_decoder_context_(saved_symbol_decoder_context),
      prev_segment_ids_(prev_segment_ids),
      dsp_(*dsp),
      post_filter_(*post_filter),
      block_parameters_holder_(*block_parameters_holder),
      quantizer_(sequence_header_.color_config.bitdepth,
                 &frame_header_.quantizer),
      residual_size_((sequence_header_.color_config.bitdepth == 8)
                         ? sizeof(int16_t)
                         : sizeof(int32_t)),
      intra_block_copy_lag_(
          frame_header_.allow_intrabc
              ? (sequence_header_.use_128x128_superblock ? 3 : 5)
              : 1),
      cdef_index_(*cdef_index),
      inter_transform_sizes_(*inter_transform_sizes),
      thread_pool_(thread_pool),
      residual_buffer_pool_(residual_buffer_pool) {
  row_ = number_ / frame_header.tile_info.tile_columns;
  column_ = number_ % frame_header.tile_info.tile_columns;
  row4x4_start_ = frame_header.tile_info.tile_row_start[row_];
  row4x4_end_ = frame_header.tile_info.tile_row_start[row_ + 1];
  column4x4_start_ = frame_header.tile_info.tile_column_start[column_];
  column4x4_end_ = frame_header.tile_info.tile_column_start[column_ + 1];
  for (size_t i = 0; i < entropy_contexts_.size(); ++i) {
    const int contexts_per_plane = (i == EntropyContext::kLeft)
                                       ? frame_header_.rows4x4
                                       : frame_header_.columns4x4;
    if (!entropy_contexts_[i].Reset(PlaneCount(), contexts_per_plane)) {
      LIBGAV1_DLOG(ERROR, "entropy_contexts_[%zu].Reset() failed.", i);
    }
  }
  const int block_width4x4 = kNum4x4BlocksWide[SuperBlockSize()];
  const int block_width4x4_log2 = k4x4HeightLog2[SuperBlockSize()];
  superblock_rows_ =
      (row4x4_end_ - row4x4_start_ + block_width4x4 - 1) >> block_width4x4_log2;
  superblock_columns_ =
      (column4x4_end_ - column4x4_start_ + block_width4x4 - 1) >>
      block_width4x4_log2;
  // Enable multi-threading within a tile only if there are at least as many
  // superblock columns as |intra_block_copy_lag_|.
  split_parse_and_decode_ =
      thread_pool_ != nullptr && superblock_columns_ > intra_block_copy_lag_;
  if (split_parse_and_decode_) {
    assert(residual_buffer_pool != nullptr);
    if (!residual_buffer_threaded_.Reset(superblock_rows_, superblock_columns_,
                                         /*zero_initialize=*/false)) {
      LIBGAV1_DLOG(ERROR, "residual_buffer_threaded_.Reset() failed.");
    }
  } else {
    residual_buffer_ = MakeAlignedUniquePtr<uint8_t>(32, 4096 * residual_size_);
    prediction_parameters_.reset(new (std::nothrow) PredictionParameters());
  }
  memset(delta_lf_, 0, sizeof(delta_lf_));
  YuvBuffer* const buffer = current_frame->buffer();
  for (int plane = 0; plane < PlaneCount(); ++plane) {
    buffer_[plane].Reset(buffer->height(plane) + buffer->bottom_border(plane),
                         buffer->stride(plane), buffer->data(plane));
  }
}

bool Tile::Decode() {
  if (frame_header_.use_ref_frame_mvs) {
    SetupMotionField(sequence_header_, frame_header_, current_frame_,
                     reference_frames_, motion_field_mv_, row4x4_start_,
                     row4x4_end_, column4x4_start_, column4x4_end_);
  }
  ResetLoopRestorationParams();
  if (split_parse_and_decode_) {
    if (!ThreadedDecode()) return false;
  } else {
    const int block_width4x4 = kNum4x4BlocksWide[SuperBlockSize()];
    SuperBlockBuffer sb_buffer;
    for (int row4x4 = row4x4_start_; row4x4 < row4x4_end_;
         row4x4 += block_width4x4) {
      for (int column4x4 = column4x4_start_; column4x4 < column4x4_end_;
           column4x4 += block_width4x4) {
        if (!ProcessSuperBlock(row4x4, column4x4, block_width4x4, &sb_buffer,
                               kProcessingModeParseAndDecode)) {
          LIBGAV1_DLOG(ERROR, "Error decoding super block row: %d column: %d",
                       row4x4, column4x4);
          return false;
        }
      }
    }
  }
  if (frame_header_.enable_frame_end_update_cdf &&
      number_ == frame_header_.tile_info.context_update_id) {
    *saved_symbol_decoder_context_ = symbol_decoder_context_;
  }
  return true;
}

bool Tile::ThreadedDecode() {
  ThreadingParameters threading;
  {
    std::lock_guard<std::mutex> lock(threading.mutex);
    if (!threading.sb_state.Reset(superblock_rows_, superblock_columns_)) {
      LIBGAV1_DLOG(ERROR, "threading.sb_state.Reset() failed.");
      return false;
    }
  }

  const int block_width4x4 = kNum4x4BlocksWide[SuperBlockSize()];

  // Begin parsing.
  SuperBlockBuffer sb_buffer;
  for (int row4x4 = row4x4_start_, row_index = 0; row4x4 < row4x4_end_;
       row4x4 += block_width4x4, ++row_index) {
    for (int column4x4 = column4x4_start_, column_index = 0;
         column4x4 < column4x4_end_;
         column4x4 += block_width4x4, ++column_index) {
      if (!ProcessSuperBlock(row4x4, column4x4, block_width4x4, &sb_buffer,
                             kProcessingModeParseOnly)) {
        std::lock_guard<std::mutex> lock(threading.mutex);
        threading.abort = true;
        break;
      }
      std::lock_guard<std::mutex> lock(threading.mutex);
      if (threading.abort) break;
      threading.sb_state[row_index][column_index] = kSuperBlockStateParsed;
      // Schedule the decoding of this superblock if it is allowed.
      if (CanDecode(row_index, column_index, threading.sb_state)) {
        ++threading.pending_jobs;
        threading.sb_state[row_index][column_index] = kSuperBlockStateScheduled;
        thread_pool_->Schedule([this, row_index, column_index, block_width4x4,
                                &threading]() {
          DecodeSuperBlock(row_index, column_index, block_width4x4, &threading);
        });
      }
    }
    std::lock_guard<std::mutex> lock(threading.mutex);
    if (threading.abort) break;
  }

  // Wait for the decode jobs to finish.
  std::unique_lock<std::mutex> lock(threading.mutex);
  while (threading.pending_jobs != 0) {
    threading.pending_jobs_zero_condvar.wait(lock);
  }

  return !threading.abort;
}

bool Tile::CanDecode(int row_index, int column_index,
                     const Array2D<SuperBlockState>& sb_state) {
  // If |sb_state| is not equal to kSuperBlockStateParsed, then return false.
  // This is ok because if |sb_state| is equal to:
  //   kSuperBlockStateNone - then the superblock is not yet parsed.
  //   kSuperBlockStateScheduled - then the superblock is already scheduled for
  //                               decode.
  //   kSuperBlockStateDecoded - then the superblock has already been decoded.
  if (row_index < 0 || column_index < 0 || row_index >= superblock_rows_ ||
      column_index >= superblock_columns_ ||
      sb_state[row_index][column_index] != kSuperBlockStateParsed) {
    return false;
  }
  // First superblock has no dependencies.
  if (row_index == 0 && column_index == 0) {
    return true;
  }
  // Superblocks in the first row only depend on the superblock to the left of
  // it.
  if (row_index == 0) {
    return sb_state[0][column_index - 1] == kSuperBlockStateDecoded;
  }
  // All other superblocks depend on superblock to the left of it (if one
  // exists) and superblock to the top right with a lag of
  // |intra_block_copy_lag_| (if one exists).
  const int top_right_column_index =
      std::min(column_index + intra_block_copy_lag_, superblock_columns_ - 1);
  return sb_state[row_index - 1][top_right_column_index] ==
             kSuperBlockStateDecoded &&
         (column_index == 0 ||
          sb_state[row_index][column_index - 1] == kSuperBlockStateDecoded);
}

void Tile::DecodeSuperBlock(int row_index, int column_index, int block_width4x4,
                            ThreadingParameters* const threading) {
  SuperBlockBuffer sb_buffer;
  const int row4x4 = row4x4_start_ + (row_index * block_width4x4);
  const int column4x4 = column4x4_start_ + (column_index * block_width4x4);
  const bool ok = ProcessSuperBlock(row4x4, column4x4, block_width4x4,
                                    &sb_buffer, kProcessingModeDecodeOnly);
  std::lock_guard<std::mutex> lock(threading->mutex);
  if (ok) {
    threading->sb_state[row_index][column_index] = kSuperBlockStateDecoded;
    // Candidate rows and columns that we could potentially begin the decoding
    // (if it is allowed to do so). The candidates are:
    //   1) The superblock to the bottom-left of the current superblock with a
    //   lag of |intra_block_copy_lag_| (or the beginning of the next superblock
    //   row in case there are less than |intra_block_copy_lag_| superblock
    //   columns in the Tile).
    //   2) The superblock to the right of the current superblock.
    const int candidate_row_indices[] = {row_index + 1, row_index};
    const int candidate_column_indices[] = {
        std::max(0, column_index - intra_block_copy_lag_), column_index + 1};
    for (size_t i = 0; i < std::extent<decltype(candidate_row_indices)>::value;
         ++i) {
      const int candidate_row_index = candidate_row_indices[i];
      const int candidate_column_index = candidate_column_indices[i];
      if (!CanDecode(candidate_row_index, candidate_column_index,
                     threading->sb_state)) {
        continue;
      }
      ++threading->pending_jobs;
      threading->sb_state[candidate_row_index][candidate_column_index] =
          kSuperBlockStateScheduled;
      thread_pool_->Schedule([this, candidate_row_index, candidate_column_index,
                              block_width4x4, threading]() {
        DecodeSuperBlock(candidate_row_index, candidate_column_index,
                         block_width4x4, threading);
      });
    }
  } else {
    threading->abort = true;
  }
  if (--threading->pending_jobs == 0) {
    // TODO(jzern): the mutex doesn't need to be locked to signal the
    // condition.
    threading->pending_jobs_zero_condvar.notify_one();
  }
}

bool Tile::IsInside(int row4x4, int column4x4) const {
  return row4x4 >= row4x4_start_ && row4x4 < row4x4_end_ &&
         column4x4 >= column4x4_start_ && column4x4 < column4x4_end_;
}

int Tile::GetTransformAllZeroContext(const Block& block, Plane plane,
                                     TransformSize tx_size, int x4, int y4,
                                     int w4, int h4) {
  const int max_x4x4 = frame_header_.columns4x4 >> SubsamplingX(plane);
  const int max_y4x4 = frame_header_.rows4x4 >> SubsamplingY(plane);

  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  const BlockSize plane_block_size =
      kPlaneResidualSize[block.size][SubsamplingX(plane)][SubsamplingY(plane)];
  assert(plane_block_size != kBlockInvalid);
  const int block_width = kBlockWidthPixels[plane_block_size];
  const int block_height = kBlockHeightPixels[plane_block_size];

  int top = 0;
  int left = 0;
  if (plane == kPlaneY) {
    if (block_width == tx_width && block_height == tx_height) return 0;
    for (int i = 0; i < w4 && (i + x4 < max_x4x4); ++i) {
      top = std::max(top,
                     static_cast<int>(
                         entropy_contexts_[EntropyContext::kTop][plane][x4 + i]
                             .coefficient_level));
    }
    for (int i = 0; i < h4 && (i + y4 < max_y4x4); ++i) {
      left = std::max(
          left, static_cast<int>(
                    entropy_contexts_[EntropyContext::kLeft][plane][y4 + i]
                        .coefficient_level));
    }
    top = std::min(top, 255);
    left = std::min(left, 255);
    const int min = std::min({top, left, 4});
    const int max = std::min(std::max(top, left), 4);
    // kAllZeroContextsByMinMax is pre-computed based on the logic in the spec
    // for top and left.
    return kAllZeroContextsByMinMax[min][max];
  }
  for (int i = 0; i < w4 && (i + x4 < max_x4x4); ++i) {
    top |= entropy_contexts_[EntropyContext::kTop][plane][x4 + i]
               .coefficient_level;
    top |= entropy_contexts_[EntropyContext::kTop][plane][x4 + i].dc_category;
  }
  for (int i = 0; i < h4 && (i + y4 < max_y4x4); ++i) {
    left |= entropy_contexts_[EntropyContext::kLeft][plane][y4 + i]
                .coefficient_level;
    left |= entropy_contexts_[EntropyContext::kLeft][plane][y4 + i].dc_category;
  }
  int context = static_cast<int>(top != 0) + static_cast<int>(left != 0) + 7;
  if (block_width * block_height > tx_width * tx_height) context += 3;
  return context;
}

TransformSet Tile::GetTransformSet(TransformSize tx_size, bool is_inter) const {
  const TransformSize tx_size_square_min = kTransformSizeSquareMin[tx_size];
  const TransformSize tx_size_square_max = kTransformSizeSquareMax[tx_size];
  if (tx_size_square_max == kTransformSize64x64) return kTransformSetDctOnly;
  if (is_inter) {
    if (frame_header_.reduced_tx_set ||
        tx_size_square_max == kTransformSize32x32) {
      return kTransformSetInter3;
    }
    if (tx_size_square_min == kTransformSize16x16) return kTransformSetInter2;
    return kTransformSetInter1;
  }
  if (tx_size_square_max == kTransformSize32x32) return kTransformSetDctOnly;
  if (frame_header_.reduced_tx_set ||
      tx_size_square_min == kTransformSize16x16) {
    return kTransformSetIntra2;
  }
  return kTransformSetIntra1;
}

TransformType Tile::ComputeTransformType(const Block& block, Plane plane,
                                         TransformSize tx_size, int block_x,
                                         int block_y) {
  const BlockParameters& bp = *block.bp;
  const TransformSize tx_size_square_max = kTransformSizeSquareMax[tx_size];
  if (frame_header_.segmentation.lossless[bp.segment_id] ||
      tx_size_square_max == kTransformSize64x64) {
    return kTransformTypeDctDct;
  }
  if (plane == kPlaneY) {
    return transform_types_[block_y - block.row4x4][block_x - block.column4x4];
  }
  const TransformSet tx_set = GetTransformSet(tx_size, bp.is_inter);
  TransformType tx_type;
  if (bp.is_inter) {
    const int x4 =
        std::max(block.column4x4,
                 block_x << sequence_header_.color_config.subsampling_x);
    const int y4 = std::max(
        block.row4x4, block_y << sequence_header_.color_config.subsampling_y);
    tx_type = transform_types_[y4 - block.row4x4][x4 - block.column4x4];
  } else {
    tx_type = kModeToTransformType[bp.uv_mode];
  }
  return static_cast<bool>((kTransformTypeInSetMask[tx_set] >> tx_type) & 1)
             ? tx_type
             : kTransformTypeDctDct;
}

void Tile::ReadTransformType(const Block& block, int x4, int y4,
                             TransformSize tx_size) {
  BlockParameters& bp = *block.bp;
  const TransformSet tx_set = GetTransformSet(tx_size, bp.is_inter);

  TransformType tx_type = kTransformTypeDctDct;
  if (tx_set != kTransformSetDctOnly &&
      frame_header_.segmentation.qindex[bp.segment_id] > 0) {
    const int cdf_index = SymbolDecoderContext::TxTypeIndex(tx_set);
    const int cdf_tx_size_index =
        TransformSizeToSquareTransformIndex(kTransformSizeSquareMin[tx_size]);
    uint16_t* cdf;
    if (bp.is_inter) {
      cdf = symbol_decoder_context_
                .inter_tx_type_cdf[cdf_index][cdf_tx_size_index];
    } else {
      const PredictionMode intra_direction =
          block.bp->prediction_parameters->use_filter_intra
              ? kFilterIntraModeToIntraPredictor[block.bp->prediction_parameters
                                                     ->filter_intra_mode]
              : bp.y_mode;
      cdf =
          symbol_decoder_context_
              .intra_tx_type_cdf[cdf_index][cdf_tx_size_index][intra_direction];
    }
    tx_type = static_cast<TransformType>(
        reader_.ReadSymbol(cdf, kNumTransformTypesInSet[tx_set]));
    // This array does not contain an entry for kTransformSetDctOnly, so the
    // first dimension needs to be offset by 1.
    tx_type = kInverseTransformTypeBySet[tx_set - 1][tx_type];
  }
  transform_types_[y4 - block.row4x4][x4 - block.column4x4] = tx_type;
  for (int i = 0; i < DivideBy4(kTransformWidth[tx_size]); ++i) {
    for (int j = 0; j < DivideBy4(kTransformHeight[tx_size]); ++j) {
      transform_types_[y4 + j - block.row4x4][x4 + i - block.column4x4] =
          tx_type;
    }
  }
}

// Section 8.3.2 in the spec, under coeff_base_eob.
int Tile::GetCoeffBaseContextEob(TransformSize tx_size, int index) {
  if (index == 0) return 0;
  const TransformSize adjusted_tx_size = kAdjustedTransformSize[tx_size];
  const int tx_width_log2 = kTransformWidthLog2[adjusted_tx_size];
  const int tx_height = kTransformHeight[adjusted_tx_size];
  if (index <= DivideBy8(tx_height << tx_width_log2)) return 1;
  if (index <= DivideBy4(tx_height << tx_width_log2)) return 2;
  return 3;
}

// Section 8.3.2 in the spec, under coeff_base.
int Tile::GetCoeffBaseContext2D(TransformSize tx_size,
                                int adjusted_tx_width_log2, uint16_t pos) {
  if (pos == 0) return 0;
  const int tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  int32_t* const quantized =
      &quantized_[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      4, DivideBy2(1 + (std::min(quantized[1], 3) +                    // {0, 1}
                        std::min(quantized[padded_tx_width], 3) +      // {1, 0}
                        std::min(quantized[padded_tx_width + 1], 3) +  // {1, 1}
                        std::min(quantized[2], 3) +                    // {0, 2}
                        std::min(quantized[MultiplyBy2(padded_tx_width)],
                                 3))));  // {2, 0}
  const int row = pos >> adjusted_tx_width_log2;
  const int column = pos & (tx_width - 1);
  return context + kCoeffBaseContextOffset[tx_size][std::min(row, 4)]
                                          [std::min(column, 4)];
}

// Section 8.3.2 in the spec, under coeff_base.
int Tile::GetCoeffBaseContextHorizontal(TransformSize /*tx_size*/,
                                        int adjusted_tx_width_log2,
                                        uint16_t pos) {
  const int tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  int32_t* const quantized =
      &quantized_[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      4, DivideBy2(1 + (std::min(quantized[1], 3) +                // {0, 1}
                        std::min(quantized[padded_tx_width], 3) +  // {1, 0}
                        std::min(quantized[2], 3) +                // {0, 2}
                        std::min(quantized[3], 3) +                // {0, 3}
                        std::min(quantized[4], 3))));              // {0, 4}
  const int index = pos & (tx_width - 1);
  return context + kCoeffBasePositionContextOffset[std::min(index, 2)];
}

// Section 8.3.2 in the spec, under coeff_base.
int Tile::GetCoeffBaseContextVertical(TransformSize /*tx_size*/,
                                      int adjusted_tx_width_log2,
                                      uint16_t pos) {
  const int tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  int32_t* const quantized =
      &quantized_[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      4, DivideBy2(1 + (std::min(quantized[1], 3) +                // {0, 1}
                        std::min(quantized[padded_tx_width], 3) +  // {1, 0}
                        std::min(quantized[MultiplyBy2(padded_tx_width)],
                                 3) +                                  // {2, 0}
                        std::min(quantized[padded_tx_width * 3], 3) +  // {3, 0}
                        std::min(quantized[MultiplyBy4(padded_tx_width)],
                                 3))));  // {4, 0}

  const int index = pos >> adjusted_tx_width_log2;
  return context + kCoeffBasePositionContextOffset[std::min(index, 2)];
}

// Section 8.3.2 in the spec, under coeff_br.
int Tile::GetCoeffBaseRangeContext2D(int adjusted_tx_width_log2, int pos) {
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  int32_t* const quantized =
      &quantized_[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      6, DivideBy2(
             1 +
             std::min(quantized[1],
                      kQuantizerCoefficientBaseRangeContextClamp) +  // {0, 1}
             std::min(quantized[padded_tx_width],
                      kQuantizerCoefficientBaseRangeContextClamp) +  // {1, 0}
             std::min(quantized[padded_tx_width + 1],
                      kQuantizerCoefficientBaseRangeContextClamp)));  // {1, 1}
  if (pos == 0) return context;
  const int row = pos >> adjusted_tx_width_log2;
  const int column = pos & (tx_width - 1);
  return context + (((row | column) < 2) ? 7 : 14);
}

// Section 8.3.2 in the spec, under coeff_br.
int Tile::GetCoeffBaseRangeContextHorizontal(int adjusted_tx_width_log2,
                                             int pos) {
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  int32_t* const quantized =
      &quantized_[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      6, DivideBy2(
             1 +
             std::min(quantized[1],
                      kQuantizerCoefficientBaseRangeContextClamp) +  // {0, 1}
             std::min(quantized[padded_tx_width],
                      kQuantizerCoefficientBaseRangeContextClamp) +  // {1, 0}
             std::min(quantized[2],
                      kQuantizerCoefficientBaseRangeContextClamp)));  // {0, 2}
  if (pos == 0) return context;
  const int column = pos & (tx_width - 1);
  return context + ((column == 0) ? 7 : 14);
}

// Section 8.3.2 in the spec, under coeff_br.
int Tile::GetCoeffBaseRangeContextVertical(int adjusted_tx_width_log2,
                                           int pos) {
  const uint8_t tx_width = 1 << adjusted_tx_width_log2;
  const int padded_tx_width = tx_width + kQuantizedCoefficientBufferPadding;
  int32_t* const quantized =
      &quantized_[PaddedIndex(pos, adjusted_tx_width_log2)];
  const int context = std::min(
      6, DivideBy2(
             1 +
             std::min(quantized[1],
                      kQuantizerCoefficientBaseRangeContextClamp) +  // {0, 1}
             std::min(quantized[padded_tx_width],
                      kQuantizerCoefficientBaseRangeContextClamp) +  // {1, 0}
             std::min(quantized[MultiplyBy2(padded_tx_width)],
                      kQuantizerCoefficientBaseRangeContextClamp)));  // {2, 0}
  if (pos == 0) return context;
  const int row = pos >> adjusted_tx_width_log2;
  return context + ((row == 0) ? 7 : 14);
}

int Tile::GetDcSignContext(int x4, int y4, int w4, int h4, Plane plane) {
  const int max_x4x4 = frame_header_.columns4x4 >> SubsamplingX(plane);
  const int max_y4x4 = frame_header_.rows4x4 >> SubsamplingY(plane);
  int dc_sign = 0;
  for (int i = 0; i < w4 && (i + x4 < max_x4x4); ++i) {
    const int sign =
        entropy_contexts_[EntropyContext::kTop][plane][x4 + i].dc_category;
    if (sign == 1) {
      dc_sign--;
    } else if (sign == 2) {
      dc_sign++;
    }
  }
  for (int i = 0; i < h4 && (i + y4 < max_y4x4); ++i) {
    const int sign =
        entropy_contexts_[EntropyContext::kLeft][plane][y4 + i].dc_category;
    if (sign == 1) {
      dc_sign--;
    } else if (sign == 2) {
      dc_sign++;
    }
  }
  if (dc_sign < 0) return 1;
  if (dc_sign > 0) return 2;
  return 0;
}

void Tile::SetEntropyContexts(int x4, int y4, int w4, int h4, Plane plane,
                              uint8_t coefficient_level, uint8_t dc_category) {
  const int max_x4x4 = frame_header_.columns4x4 >> SubsamplingX(plane);
  const int max_y4x4 = frame_header_.rows4x4 >> SubsamplingY(plane);
  for (int i = 0; i < w4 && (i + x4 < max_x4x4); ++i) {
    entropy_contexts_[EntropyContext::kTop][plane][x4 + i].coefficient_level =
        coefficient_level;
    entropy_contexts_[EntropyContext::kTop][plane][x4 + i].dc_category =
        dc_category;
  }
  for (int i = 0; i < h4 && (i + y4 < max_y4x4); ++i) {
    entropy_contexts_[EntropyContext::kLeft][plane][y4 + i].coefficient_level =
        coefficient_level;
    entropy_contexts_[EntropyContext::kLeft][plane][y4 + i].dc_category =
        dc_category;
  }
}

void Tile::ScaleMotionVector(const MotionVector& mv, const Plane plane,
                             const int reference_frame_index, const int x,
                             const int y, int* const start_x,
                             int* const start_y, int* const step_x,
                             int* const step_y) {
  const int reference_upscaled_width =
      (reference_frame_index == -1)
          ? frame_header_.upscaled_width
          : reference_frames_[reference_frame_index]->upscaled_width();
  const int reference_height =
      (reference_frame_index == -1)
          ? frame_header_.height
          : reference_frames_[reference_frame_index]->frame_height();
  assert(2 * frame_header_.width >= reference_upscaled_width &&
         2 * frame_header_.height >= reference_height &&
         frame_header_.width <= 16 * reference_upscaled_width &&
         frame_header_.height <= 16 * reference_height);
  const bool is_scaled_x = reference_upscaled_width != frame_header_.width;
  const bool is_scaled_y = reference_height != frame_header_.height;
  const int half_sample = 1 << (kSubPixelBits - 1);
  int orig_x = (x << kSubPixelBits) + ((2 * mv.mv[1]) >> SubsamplingX(plane));
  int orig_y = (y << kSubPixelBits) + ((2 * mv.mv[0]) >> SubsamplingY(plane));
  const int rounding_offset =
      DivideBy2(1 << (kScaleSubPixelBits - kSubPixelBits));
  if (is_scaled_x) {
    const int scale_x = ((reference_upscaled_width << kReferenceScaleShift) +
                         DivideBy2(frame_header_.width)) /
                        frame_header_.width;
    *step_x = RightShiftWithRoundingSigned(
        scale_x, kReferenceScaleShift - kScaleSubPixelBits);
    orig_x += half_sample;
    // When frame size is 4k and above, orig_x can be above 16 bits, scale_x can
    // be up to 15 bits. So we use int64_t to hold base_x.
    const int64_t base_x = static_cast<int64_t>(orig_x) * scale_x -
                           (half_sample << kReferenceScaleShift);
    *start_x =
        RightShiftWithRoundingSigned(
            base_x, kReferenceScaleShift + kSubPixelBits - kScaleSubPixelBits) +
        rounding_offset;
  } else {
    *step_x = 1 << kScaleSubPixelBits;
    *start_x = LeftShift(orig_x, 6) + rounding_offset;
  }
  if (is_scaled_y) {
    const int scale_y = ((reference_height << kReferenceScaleShift) +
                         DivideBy2(frame_header_.height)) /
                        frame_header_.height;
    *step_y = RightShiftWithRoundingSigned(
        scale_y, kReferenceScaleShift - kScaleSubPixelBits);
    orig_y += half_sample;
    const int64_t base_y = static_cast<int64_t>(orig_y) * scale_y -
                           (half_sample << kReferenceScaleShift);
    *start_y =
        RightShiftWithRoundingSigned(
            base_y, kReferenceScaleShift + kSubPixelBits - kScaleSubPixelBits) +
        rounding_offset;
  } else {
    *step_y = 1 << kScaleSubPixelBits;
    *start_y = LeftShift(orig_y, 6) + rounding_offset;
  }
}

int16_t Tile::ReadTransformCoefficients(const Block& block, Plane plane,
                                        int start_x, int start_y,
                                        TransformSize tx_size,
                                        TransformType* const tx_type) {
  const int x4 = DivideBy4(start_x);
  const int y4 = DivideBy4(start_y);
  const int w4 = DivideBy4(kTransformWidth[tx_size]);
  const int h4 = DivideBy4(kTransformHeight[tx_size]);

  const int tx_size_square_min =
      TransformSizeToSquareTransformIndex(kTransformSizeSquareMin[tx_size]);
  const int tx_size_square_max =
      TransformSizeToSquareTransformIndex(kTransformSizeSquareMax[tx_size]);
  const int tx_size_context =
      DivideBy2(tx_size_square_min + tx_size_square_max + 1);
  int context =
      GetTransformAllZeroContext(block, plane, tx_size, x4, y4, w4, h4);
  const bool all_zero = reader_.ReadSymbol(
      symbol_decoder_context_.all_zero_cdf[tx_size_context][context]);
  if (all_zero) {
    if (plane == kPlaneY) {
      for (int i = 0; i < w4; ++i) {
        for (int j = 0; j < h4; ++j) {
          transform_types_[y4 + j - block.row4x4][x4 + i - block.column4x4] =
              kTransformTypeDctDct;
        }
      }
    }
    SetEntropyContexts(x4, y4, w4, h4, plane, 0, 0);
    // This is not used in this case, so it can be set to any value.
    *tx_type = kNumTransformTypes;
    return 0;
  }
  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  memset(block.sb_buffer->residual, 0, tx_width * tx_height * residual_size_);
  const int clamped_tx_width = std::min(tx_width, 32);
  const int clamped_tx_height = std::min(tx_height, 32);
  const int padded_tx_width =
      clamped_tx_width + kQuantizedCoefficientBufferPadding;
  const int padded_tx_height =
      clamped_tx_height + kQuantizedCoefficientBufferPadding;
  // Only the first |padded_tx_width| * |padded_tx_height| values of
  // |quantized_| will be used by this function. So we simply need to zero out
  // those values before it is being used (instead of zeroing the entire array).
  memset(quantized_, 0,
         padded_tx_width * padded_tx_height * sizeof(quantized_[0]));
  if (plane == kPlaneY) {
    ReadTransformType(block, x4, y4, tx_size);
  }
  BlockParameters& bp = *block.bp;
  *tx_type = ComputeTransformType(block, plane, tx_size, x4, y4);
  const int eob_multi_size = kEobMultiSizeLookup[tx_size];
  const PlaneType plane_type = GetPlaneType(plane);
  context = static_cast<int>(GetTransformClass(*tx_type) != kTransformClass2D);
  uint16_t* cdf;
  switch (eob_multi_size) {
    case 0:
      cdf = symbol_decoder_context_.eob_pt_16_cdf[plane_type][context];
      break;
    case 1:
      cdf = symbol_decoder_context_.eob_pt_32_cdf[plane_type][context];
      break;
    case 2:
      cdf = symbol_decoder_context_.eob_pt_64_cdf[plane_type][context];
      break;
    case 3:
      cdf = symbol_decoder_context_.eob_pt_128_cdf[plane_type][context];
      break;
    case 4:
      cdf = symbol_decoder_context_.eob_pt_256_cdf[plane_type][context];
      break;
    case 5:
      cdf = symbol_decoder_context_.eob_pt_512_cdf[plane_type];
      break;
    case 6:
    default:
      cdf = symbol_decoder_context_.eob_pt_1024_cdf[plane_type];
      break;
  }
  const int16_t eob_pt =
      1 + reader_.ReadSymbol(cdf, kEobPtSymbolCount[eob_multi_size]);
  int16_t eob = (eob_pt < 2) ? eob_pt : ((1 << (eob_pt - 2)) + 1);
  int coefficient_level = 0;
  uint8_t dc_category = 0;
  if (eob_pt >= 3) {
    context = eob_pt - 3;
    const bool eob_extra = reader_.ReadSymbol(
        symbol_decoder_context_
            .eob_extra_cdf[tx_size_context][plane_type][context]);
    if (eob_extra) eob += 1 << (eob_pt - 3);
    for (int i = 1; i < eob_pt - 2; ++i) {
      assert(eob_pt - i >= 3);
      assert(eob_pt <= kEobPtSymbolCount[6]);
      if (static_cast<bool>(reader_.ReadBit())) {
        eob += 1 << (eob_pt - i - 3);
      }
    }
  }
  const uint16_t* scan = GetScan(tx_size, *tx_type);
  const TransformSize adjusted_tx_size = kAdjustedTransformSize[tx_size];
  const int adjusted_tx_width_log2 = kTransformWidthLog2[adjusted_tx_size];
  const TransformClass tx_class = GetTransformClass(*tx_type);
  // Lookup used to call the right variant of GetCoeffBaseContext*() based on
  // the transform class.
  static constexpr int (Tile::*kGetCoeffBaseContextFunc[])(
      TransformSize, int, uint16_t) = {&Tile::GetCoeffBaseContext2D,
                                       &Tile::GetCoeffBaseContextHorizontal,
                                       &Tile::GetCoeffBaseContextVertical};
  auto get_coeff_base_context_func = kGetCoeffBaseContextFunc[tx_class];
  // Lookup used to call the right variant of GetCoeffBaseRangeContext*() based
  // on the transform class.
  static constexpr int (Tile::*kGetCoeffBaseRangeContextFunc[])(int, int) = {
      &Tile::GetCoeffBaseRangeContext2D,
      &Tile::GetCoeffBaseRangeContextHorizontal,
      &Tile::GetCoeffBaseRangeContextVertical};
  auto get_coeff_base_range_context_func =
      kGetCoeffBaseRangeContextFunc[tx_class];
  for (int i = eob - 1; i >= 0; --i) {
    const uint16_t pos = scan[i];
    int level;
    int symbol_count;
    if (i == eob - 1) {
      level = 1;
      context = GetCoeffBaseContextEob(tx_size, i);
      cdf = symbol_decoder_context_
                .coeff_base_eob_cdf[tx_size_context][plane_type][context];
      symbol_count = kCoeffBaseEobSymbolCount;
    } else {
      level = 0;
      context = (this->*get_coeff_base_context_func)(
          tx_size, adjusted_tx_width_log2, pos);
      cdf = symbol_decoder_context_
                .coeff_base_cdf[tx_size_context][plane_type][context];
      symbol_count = kCoeffBaseSymbolCount;
    }
    level += reader_.ReadSymbol(cdf, symbol_count);
    if (level > kNumQuantizerBaseLevels) {
      context = (this->*get_coeff_base_range_context_func)(
          adjusted_tx_width_log2, pos);
      for (int j = 0; j < kCoeffBaseRangeMaxIterations; ++j) {
        const int coeff_base_range = reader_.ReadSymbol(
            symbol_decoder_context_.coeff_base_range_cdf[std::min(
                tx_size_context, 3)][plane_type][context],
            kCoeffBaseRangeSymbolCount);
        level += coeff_base_range;
        if (coeff_base_range < (kCoeffBaseRangeSymbolCount - 1)) break;
      }
    }
    quantized_[PaddedIndex(pos, adjusted_tx_width_log2)] = level;
  }
  const int min_value = -(1 << (7 + sequence_header_.color_config.bitdepth));
  const int max_value = (1 << (7 + sequence_header_.color_config.bitdepth)) - 1;
  const int current_quantizer_index = GetQIndex(
      frame_header_.segmentation, bp.segment_id, current_quantizer_index_);
  const int dc_q_value = quantizer_.GetDcValue(plane, current_quantizer_index);
  const int ac_q_value = quantizer_.GetAcValue(plane, current_quantizer_index);
  const int shift = GetQuantizationShift(tx_size);
  for (int i = 0; i < eob; ++i) {
    int pos = scan[i];
    const int pos_index = PaddedIndex(pos, adjusted_tx_width_log2);
    bool sign = false;
    if (quantized_[pos_index] != 0) {
      if (i == 0) {
        context = GetDcSignContext(x4, y4, w4, h4, plane);
        sign = reader_.ReadSymbol(
            symbol_decoder_context_.dc_sign_cdf[plane_type][context]);
      } else {
        sign = static_cast<bool>(reader_.ReadBit());
      }
    }
    if (quantized_[pos_index] >
        kNumQuantizerBaseLevels + kQuantizerCoefficientBaseRange) {
      int length = 0;
      bool golomb_length_bit = false;
      do {
        golomb_length_bit = static_cast<bool>(reader_.ReadBit());
        ++length;
        if (length > 20) {
          LIBGAV1_DLOG(ERROR, "Invalid golomb_length %d", length);
          return -1;
        }
      } while (!golomb_length_bit);
      int x = 1;
      for (int i = length - 2; i >= 0; --i) {
        x = (x << 1) | reader_.ReadBit();
      }
      quantized_[pos_index] += x - 1;
    }
    if (pos == 0 && quantized_[pos_index] > 0) {
      dc_category = sign ? 1 : 2;
    }
    quantized_[pos_index] &= 0xfffff;
    coefficient_level += quantized_[pos_index];
    // Apply dequantization. Step 1 of section 7.12.3 in the spec.
    int q = (pos == 0) ? dc_q_value : ac_q_value;
    if (frame_header_.quantizer.use_matrix &&
        *tx_type < kTransformTypeIdentityIdentity &&
        !frame_header_.segmentation.lossless[bp.segment_id] &&
        frame_header_.quantizer.matrix_level[plane] < 15) {
      q *= kQuantizerMatrix[frame_header_.quantizer.matrix_level[plane]]
                           [plane_type][kQuantizerMatrixOffset[tx_size] + pos];
      q = RightShiftWithRounding(q, 5);
    }
    // The intermediate multiplication can exceed 32 bits, so it has to be
    // performed by promoting one of the values to int64_t.
    int32_t dequantized_value =
        (static_cast<int64_t>(q) * quantized_[pos_index]) & 0xffffff;
    dequantized_value >>= shift;
    if (sign) {
      dequantized_value = -dequantized_value;
    }
    // Inverse transform process assumes that the quantized coefficients are
    // stored as a virtual 2d array of size |tx_width| x |tx_height|. If
    // transform width is 64, then this assumption is broken because the scan
    // order used for populating the coefficients for such transforms is the
    // same as the one used for corresponding transform with width 32 (e.g. the
    // scan order used for 64x16 is the same as the one used for 32x16). So we
    // have to recompute the value of pos so that it reflects the index of the
    // 2d array of size 64 x |tx_height|.
    if (tx_width == 64) {
      const int row_index = DivideBy32(pos);
      const int column_index = Mod32(pos);
      pos = MultiplyBy64(row_index) + column_index;
    }
    if (sequence_header_.color_config.bitdepth == 8) {
      auto* const residual_buffer =
          reinterpret_cast<int16_t*>(block.sb_buffer->residual);
      residual_buffer[pos] = Clip3(dequantized_value, min_value, max_value);
#if LIBGAV1_MAX_BITDEPTH >= 10
    } else {
      auto* const residual_buffer =
          reinterpret_cast<int32_t*>(block.sb_buffer->residual);
      residual_buffer[pos] = Clip3(dequantized_value, min_value, max_value);
#endif
    }
  }
  SetEntropyContexts(x4, y4, w4, h4, plane, std::min(63, coefficient_level),
                     dc_category);
  if (split_parse_and_decode_) {
    block.sb_buffer->residual += tx_width * tx_height * residual_size_;
  }
  return eob;
}

bool Tile::TransformBlock(const Block& block, Plane plane, int base_x,
                          int base_y, TransformSize tx_size, int x, int y,
                          ProcessingMode mode) {
  BlockParameters& bp = *block.bp;
  const int subsampling_x = SubsamplingX(plane);
  const int subsampling_y = SubsamplingY(plane);
  const int start_x = base_x + MultiplyBy4(x);
  const int start_y = base_y + MultiplyBy4(y);
  const int max_x = MultiplyBy4(frame_header_.columns4x4) >> subsampling_x;
  const int max_y = MultiplyBy4(frame_header_.rows4x4) >> subsampling_y;
  if (start_x >= max_x || start_y >= max_y) return true;
  const int row = DivideBy4(start_y << subsampling_y);
  const int column = DivideBy4(start_x << subsampling_x);
  const int mask = sequence_header_.use_128x128_superblock ? 31 : 15;
  const int sub_block_row4x4 = row & mask;
  const int sub_block_column4x4 = column & mask;
  const int step_x = DivideBy4(kTransformWidth[tx_size]);
  const int step_y = DivideBy4(kTransformHeight[tx_size]);
  const bool do_decode = mode == kProcessingModeDecodeOnly ||
                         mode == kProcessingModeParseAndDecode;
  if (do_decode && !bp.is_inter) {
    if (bp.palette_mode_info.size[GetPlaneType(plane)] > 0) {
      if (sequence_header_.color_config.bitdepth == 8) {
        PalettePrediction<uint8_t>(block, plane, start_x, start_y, x, y,
                                   tx_size);
#if LIBGAV1_MAX_BITDEPTH >= 10
      } else {
        PalettePrediction<uint16_t>(block, plane, start_x, start_y, x, y,
                                    tx_size);
#endif
      }
    } else {
      const PredictionMode mode =
          (plane == kPlaneY)
              ? bp.y_mode
              : (bp.uv_mode == kPredictionModeChromaFromLuma ? kPredictionModeDc
                                                             : bp.uv_mode);
      const int tr_row4x4 = (sub_block_row4x4 >> subsampling_y) - 1;
      const int tr_column4x4 = (sub_block_column4x4 >> subsampling_x) + step_x;
      const int bl_row4x4 = (sub_block_row4x4 >> subsampling_y) + step_y;
      const int bl_column4x4 = (sub_block_column4x4 >> subsampling_x) - 1;
      const bool has_left =
          x > 0 || (plane == kPlaneY ? block.left_available
                                     : block.LeftAvailableChroma());
      const bool has_top =
          y > 0 ||
          (plane == kPlaneY ? block.top_available : block.TopAvailableChroma());
      if (sequence_header_.color_config.bitdepth == 8) {
        IntraPrediction<uint8_t>(
            block, plane, start_x, start_y, has_left, has_top,
            BlockDecoded(block, plane, tr_row4x4, tr_column4x4, has_top),
            BlockDecoded(block, plane, bl_row4x4, bl_column4x4, has_left), mode,
            tx_size);
#if LIBGAV1_MAX_BITDEPTH >= 10
      } else {
        IntraPrediction<uint16_t>(
            block, plane, start_x, start_y, has_left, has_top,
            BlockDecoded(block, plane, tr_row4x4, tr_column4x4, has_top),
            BlockDecoded(block, plane, bl_row4x4, bl_column4x4, has_left), mode,
            tx_size);
#endif
      }
      if (plane != kPlaneY && bp.uv_mode == kPredictionModeChromaFromLuma) {
        if (sequence_header_.color_config.bitdepth == 8) {
          ChromaFromLumaPrediction<uint8_t>(block, plane, start_x, start_y,
                                            tx_size);
#if LIBGAV1_MAX_BITDEPTH >= 10
        } else {
          ChromaFromLumaPrediction<uint16_t>(block, plane, start_x, start_y,
                                             tx_size);
#endif
        }
      }
    }
    if (plane == kPlaneY) {
      block.bp->prediction_parameters->max_luma_width =
          start_x + MultiplyBy4(step_x);
      block.bp->prediction_parameters->max_luma_height =
          start_y + MultiplyBy4(step_y);
      block.sb_buffer->cfl_luma_buffer_valid = false;
    }
  }
  if (!bp.skip) {
    switch (mode) {
      case kProcessingModeParseAndDecode: {
        TransformType tx_type;
        const int16_t non_zero_coeff_count = ReadTransformCoefficients(
            block, plane, start_x, start_y, tx_size, &tx_type);
        if (non_zero_coeff_count < 0) return false;
        ReconstructBlock(block, plane, start_x, start_y, tx_size, tx_type,
                         non_zero_coeff_count);
        break;
      }
      case kProcessingModeParseOnly: {
        TransformType tx_type;
        const int16_t non_zero_coeff_count = ReadTransformCoefficients(
            block, plane, start_x, start_y, tx_size, &tx_type);
        if (non_zero_coeff_count < 0) return false;
        block.sb_buffer->transform_parameters->Push(non_zero_coeff_count,
                                                    tx_type);
        break;
      }
      case kProcessingModeDecodeOnly: {
        ReconstructBlock(
            block, plane, start_x, start_y, tx_size,
            block.sb_buffer->transform_parameters->Type(),
            block.sb_buffer->transform_parameters->NonZeroCoeffCount());
        block.sb_buffer->transform_parameters->Pop();
        break;
      }
    }
  }
  if (do_decode) {
    for (int i = 0; i < step_y; ++i) {
      for (int j = 0; j < step_x; ++j) {
        block.sb_buffer
            ->block_decoded[plane][(sub_block_row4x4 >> subsampling_y) + i]
                           [(sub_block_column4x4 >> subsampling_x) + j] = true;
      }
    }
  }
  return true;
}

bool Tile::TransformTree(const Block& block, int start_x, int start_y,
                         int width, int height, ProcessingMode mode) {
  const int row = DivideBy4(start_y);
  const int column = DivideBy4(start_x);
  if (row >= frame_header_.rows4x4 || column >= frame_header_.columns4x4) {
    return true;
  }
  const TransformSize inter_tx_size = inter_transform_sizes_[row][column];
  if (width <= kTransformWidth[inter_tx_size] &&
      height <= kTransformHeight[inter_tx_size]) {
    TransformSize tx_size = kNumTransformSizes;
    for (int i = 0; i < kNumTransformSizes; ++i) {
      if (kTransformWidth[i] == width && kTransformHeight[i] == height) {
        tx_size = static_cast<TransformSize>(i);
        break;
      }
    }
    assert(tx_size < kNumTransformSizes);
    return TransformBlock(block, kPlaneY, start_x, start_y, tx_size, 0, 0,
                          mode);
  }
  const int half_width = DivideBy2(width);
  const int half_height = DivideBy2(height);
  if (width > height) {
    return TransformTree(block, start_x, start_y, half_width, height, mode) &&
           TransformTree(block, start_x + half_width, start_y, half_width,
                         height, mode);
  }
  if (width < height) {
    return TransformTree(block, start_x, start_y, width, half_height, mode) &&
           TransformTree(block, start_x, start_y + half_height, width,
                         half_height, mode);
  }
  return TransformTree(block, start_x, start_y, half_width, half_height,
                       mode) &&
         TransformTree(block, start_x + half_width, start_y, half_width,
                       half_height, mode) &&
         TransformTree(block, start_x, start_y + half_height, half_width,
                       half_height, mode) &&
         TransformTree(block, start_x + half_width, start_y + half_height,
                       half_width, half_height, mode);
}

void Tile::ReconstructBlock(const Block& block, Plane plane, int start_x,
                            int start_y, TransformSize tx_size,
                            TransformType tx_type,
                            int16_t non_zero_coeff_count) {
  assert(non_zero_coeff_count >= 0);
  if (non_zero_coeff_count == 0) return;
  // Reconstruction process. Steps 2 and 3 of Section 7.12.3 in the spec.
  if (sequence_header_.color_config.bitdepth == 8) {
    Reconstruct(dsp_, tx_type, tx_size, sequence_header_.color_config.bitdepth,
                frame_header_.segmentation.lossless[block.bp->segment_id],
                reinterpret_cast<int16_t*>(block.sb_buffer->residual), start_x,
                start_y, &buffer_[plane], non_zero_coeff_count);
#if LIBGAV1_MAX_BITDEPTH >= 10
  } else {
    Array2DView<uint16_t> buffer(
        buffer_[plane].rows(), buffer_[plane].columns() / sizeof(uint16_t),
        reinterpret_cast<uint16_t*>(&buffer_[plane][0][0]));
    Reconstruct(dsp_, tx_type, tx_size, sequence_header_.color_config.bitdepth,
                frame_header_.segmentation.lossless[block.bp->segment_id],
                reinterpret_cast<int32_t*>(block.sb_buffer->residual), start_x,
                start_y, &buffer, non_zero_coeff_count);
#endif
  }
  if (split_parse_and_decode_) {
    block.sb_buffer->residual +=
        kTransformWidth[tx_size] * kTransformHeight[tx_size] * residual_size_;
  }
}

bool Tile::Residual(const Block& block, ProcessingMode mode) {
  const int width_chunks = std::max(1, kBlockWidthPixels[block.size] >> 6);
  const int height_chunks = std::max(1, kBlockHeightPixels[block.size] >> 6);
  const BlockSize size_chunk4x4 =
      (width_chunks > 1 || height_chunks > 1) ? kBlock64x64 : block.size;
  const BlockParameters& bp = *block.bp;

  for (int chunk_y = 0; chunk_y < height_chunks; ++chunk_y) {
    for (int chunk_x = 0; chunk_x < width_chunks; ++chunk_x) {
      for (int plane = 0; plane < (block.HasChroma() ? PlaneCount() : 1);
           ++plane) {
        const int subsampling_x = SubsamplingX(static_cast<Plane>(plane));
        const int subsampling_y = SubsamplingY(static_cast<Plane>(plane));
        const TransformSize tx_size =
            GetTransformSize(frame_header_.segmentation.lossless[bp.segment_id],
                             block.size, static_cast<Plane>(plane),
                             bp.transform_size, subsampling_x, subsampling_y);
        const BlockSize plane_size =
            kPlaneResidualSize[size_chunk4x4][subsampling_x][subsampling_y];
        assert(plane_size != kBlockInvalid);
        if (bp.is_inter &&
            !frame_header_.segmentation.lossless[bp.segment_id] &&
            plane == kPlaneY) {
          const int row_chunk4x4 = block.row4x4 + MultiplyBy16(chunk_y);
          const int column_chunk4x4 = block.column4x4 + MultiplyBy16(chunk_x);
          const int base_x = MultiplyBy4(column_chunk4x4 >> subsampling_x);
          const int base_y = MultiplyBy4(row_chunk4x4 >> subsampling_y);
          if (!TransformTree(block, base_x, base_y,
                             kBlockWidthPixels[plane_size],
                             kBlockHeightPixels[plane_size], mode)) {
            return false;
          }
        } else {
          const int base_x = MultiplyBy4(block.column4x4 >> subsampling_x);
          const int base_y = MultiplyBy4(block.row4x4 >> subsampling_y);
          const int step_x = DivideBy4(kTransformWidth[tx_size]);
          const int step_y = DivideBy4(kTransformHeight[tx_size]);
          const int num4x4_wide = kNum4x4BlocksWide[plane_size];
          const int num4x4_high = kNum4x4BlocksHigh[plane_size];
          for (int y = 0; y < num4x4_high; y += step_y) {
            for (int x = 0; x < num4x4_wide; x += step_x) {
              if (!TransformBlock(
                      block, static_cast<Plane>(plane), base_x, base_y, tx_size,
                      x + (MultiplyBy16(chunk_x) >> subsampling_x),
                      y + (MultiplyBy16(chunk_y) >> subsampling_y), mode)) {
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

// The purpose of this function is to limit the maximum size of motion vectors
// and also, if use_intra_block_copy is true, to additionally constrain the
// motion vector so that the data is fetched from parts of the tile that have
// already been decoded and are not too close to the current block (in order to
// make a pipelined decoder implementation feasible).
bool Tile::IsMvValid(const Block& block, bool is_compound) const {
  const BlockParameters& bp = *block.bp;
  for (int i = 0; i < 1 + static_cast<int>(is_compound); ++i) {
    for (int mv_component : bp.mv[i].mv) {
      if (std::abs(mv_component) >= (1 << 14)) {
        return false;
      }
    }
  }
  if (!block.bp->prediction_parameters->use_intra_block_copy) {
    return true;
  }
  const int block_width = kBlockWidthPixels[block.size];
  const int block_height = kBlockHeightPixels[block.size];
  if ((bp.mv[0].mv[0] & 7) != 0 || (bp.mv[0].mv[1] & 7) != 0) {
    return false;
  }
  const int delta_row = bp.mv[0].mv[0] >> 3;
  const int delta_column = bp.mv[0].mv[1] >> 3;
  int src_top_edge = MultiplyBy4(block.row4x4) + delta_row;
  int src_left_edge = MultiplyBy4(block.column4x4) + delta_column;
  const int src_bottom_edge = src_top_edge + block_height;
  const int src_right_edge = src_left_edge + block_width;
  if (block.HasChroma()) {
    if (block_width < 8 && sequence_header_.color_config.subsampling_x != 0) {
      src_left_edge -= 4;
    }
    if (block_height < 8 && sequence_header_.color_config.subsampling_y != 0) {
      src_top_edge -= 4;
    }
  }
  if (src_top_edge < MultiplyBy4(row4x4_start_) ||
      src_left_edge < MultiplyBy4(column4x4_start_) ||
      src_bottom_edge > MultiplyBy4(row4x4_end_) ||
      src_right_edge > MultiplyBy4(column4x4_end_)) {
    return false;
  }
  // sb_height_log2 = use_128x128_superblock ? log2(128) : log2(64)
  const int sb_height_log2 =
      6 + static_cast<int>(sequence_header_.use_128x128_superblock);
  const int active_sb_row = MultiplyBy4(block.row4x4) >> sb_height_log2;
  const int active_64x64_block_column = MultiplyBy4(block.column4x4) >> 6;
  const int src_sb_row = (src_bottom_edge - 1) >> sb_height_log2;
  const int src_64x64_block_column = (src_right_edge - 1) >> 6;
  const int total_64x64_blocks_per_row =
      ((column4x4_end_ - column4x4_start_ - 1) >> 4) + 1;
  const int active_64x64_block =
      active_sb_row * total_64x64_blocks_per_row + active_64x64_block_column;
  const int src_64x64_block =
      src_sb_row * total_64x64_blocks_per_row + src_64x64_block_column;
  if (src_64x64_block >= active_64x64_block - kIntraBlockCopyDelay64x64Blocks) {
    return false;
  }

  // Wavefront constraint: use only top left area of frame for reference.
  if (src_sb_row > active_sb_row) return false;
  const int gradient =
      1 + kIntraBlockCopyDelay64x64Blocks +
      static_cast<int>(sequence_header_.use_128x128_superblock);
  const int wavefront_offset = gradient * (active_sb_row - src_sb_row);
  return src_64x64_block_column < active_64x64_block_column -
                                      kIntraBlockCopyDelay64x64Blocks +
                                      wavefront_offset;
}

bool Tile::AssignMv(const Block& block, bool is_compound) {
  MotionVector predicted_mv[2] = {};
  BlockParameters& bp = *block.bp;
  for (int i = 0; i < 1 + static_cast<int>(is_compound); ++i) {
    const PredictionParameters& prediction_parameters =
        *block.bp->prediction_parameters;
    const PredictionMode mode = prediction_parameters.use_intra_block_copy
                                    ? kPredictionModeNewMv
                                    : GetSinglePredictionMode(i, bp.y_mode);
    if (prediction_parameters.use_intra_block_copy) {
      predicted_mv[0] = prediction_parameters.ref_mv_stack[0].mv[0];
      if (predicted_mv[0].mv[0] == 0 && predicted_mv[0].mv[1] == 0) {
        predicted_mv[0] = prediction_parameters.ref_mv_stack[1].mv[0];
      }
      if (predicted_mv[0].mv[0] == 0 && predicted_mv[0].mv[1] == 0) {
        const int super_block_size4x4 = kNum4x4BlocksHigh[SuperBlockSize()];
        if (block.row4x4 - super_block_size4x4 < row4x4_start_) {
          predicted_mv[0].mv[1] = -MultiplyBy8(
              MultiplyBy4(super_block_size4x4) + kIntraBlockCopyDelayPixels);
        } else {
          predicted_mv[0].mv[0] = -MultiplyBy32(super_block_size4x4);
        }
      }
    } else if (mode == kPredictionModeGlobalMv) {
      predicted_mv[i] = prediction_parameters.global_mv[i];
    } else {
      const int ref_mv_index = (mode == kPredictionModeNearestMv ||
                                (mode == kPredictionModeNewMv &&
                                 prediction_parameters.ref_mv_count <= 1))
                                   ? 0
                                   : prediction_parameters.ref_mv_index;
      predicted_mv[i] = prediction_parameters.ref_mv_stack[ref_mv_index].mv[i];
    }
    if (mode == kPredictionModeNewMv) {
      ReadMotionVector(block, i);
      bp.mv[i].mv[0] += predicted_mv[i].mv[0];
      bp.mv[i].mv[1] += predicted_mv[i].mv[1];
    } else {
      bp.mv[i] = predicted_mv[i];
    }
  }
  return IsMvValid(block, is_compound);
}

void Tile::ResetEntropyContext(const Block& block) {
  const int block_width4x4 = kNum4x4BlocksWide[block.size];
  const int block_height4x4 = kNum4x4BlocksHigh[block.size];
  for (int plane = 0; plane < (block.HasChroma() ? PlaneCount() : 1); ++plane) {
    const int subsampling_x = SubsamplingX(static_cast<Plane>(plane));
    const int start_x = block.column4x4 >> subsampling_x;
    const int end_x =
        std::min((block.column4x4 + block_width4x4) >> subsampling_x,
                 frame_header_.columns4x4);
    for (int x = start_x; x < end_x; ++x) {
      entropy_contexts_[EntropyContext::kTop][plane][x] = {};
    }
    const int subsampling_y = SubsamplingY(static_cast<Plane>(plane));
    const int start_y = block.row4x4 >> subsampling_y;
    const int end_y =
        std::min((block.row4x4 + block_height4x4) >> subsampling_y,
                 frame_header_.rows4x4);
    for (int y = start_y; y < end_y; ++y) {
      entropy_contexts_[EntropyContext::kLeft][plane][y] = {};
    }
  }
}

bool Tile::ComputePrediction(const Block& block) {
  const int mask =
      (1 << (4 + static_cast<int>(sequence_header_.use_128x128_superblock))) -
      1;
  const int sub_block_row4x4 = block.row4x4 & mask;
  const int sub_block_column4x4 = block.column4x4 & mask;
  // Returns true if this block applies local warping. The state is determined
  // in the Y plane and carried for use in the U/V planes.
  // But the U/V planes will not apply warping when the block size is smaller
  // than 8x8, even if this variable is true.
  bool is_local_valid = false;
  // Local warping parameters, similar usage as is_local_valid.
  GlobalMotion local_warp_params;
  for (int plane = 0; plane < (block.HasChroma() ? PlaneCount() : 1); ++plane) {
    const int8_t subsampling_x = SubsamplingX(static_cast<Plane>(plane));
    const int8_t subsampling_y = SubsamplingY(static_cast<Plane>(plane));
    const BlockSize plane_size =
        kPlaneResidualSize[block.size][subsampling_x][subsampling_y];
    assert(plane_size != kBlockInvalid);
    const int block_width4x4 = kNum4x4BlocksWide[plane_size];
    const int block_height4x4 = kNum4x4BlocksHigh[plane_size];
    const int block_width = kBlockWidthPixels[plane_size];
    const int block_height = kBlockHeightPixels[plane_size];
    const int base_x = MultiplyBy4(block.column4x4 >> subsampling_x);
    const int base_y = MultiplyBy4(block.row4x4 >> subsampling_y);
    const BlockParameters& bp = *block.bp;
    if (bp.is_inter && bp.reference_frame[1] == kReferenceFrameIntra) {
      const int tr_row4x4 = (sub_block_row4x4 >> subsampling_y) - 1;
      const int tr_column4x4 =
          (sub_block_column4x4 >> subsampling_x) + block_width4x4;
      const int bl_row4x4 =
          (sub_block_row4x4 >> subsampling_y) + block_height4x4;
      const int bl_column4x4 = (sub_block_column4x4 >> subsampling_x) - 1;
      const TransformSize tx_size =
          k4x4SizeToTransformSize[k4x4WidthLog2[plane_size]]
                                 [k4x4HeightLog2[plane_size]];
      const bool has_left =
          plane == kPlaneY ? block.left_available : block.LeftAvailableChroma();
      const bool has_top =
          plane == kPlaneY ? block.top_available : block.TopAvailableChroma();
      if (sequence_header_.color_config.bitdepth == 8) {
        IntraPrediction<uint8_t>(
            block, static_cast<Plane>(plane), base_x, base_y, has_left, has_top,
            BlockDecoded(block, static_cast<Plane>(plane), tr_row4x4,
                         tr_column4x4, has_top),
            BlockDecoded(block, static_cast<Plane>(plane), bl_row4x4,
                         bl_column4x4, has_left),
            kInterIntraToIntraMode[block.bp->prediction_parameters
                                       ->inter_intra_mode],
            tx_size);
#if LIBGAV1_MAX_BITDEPTH >= 10
      } else {
        IntraPrediction<uint16_t>(
            block, static_cast<Plane>(plane), base_x, base_y, has_left, has_top,
            BlockDecoded(block, static_cast<Plane>(plane), tr_row4x4,
                         tr_column4x4, has_top),
            BlockDecoded(block, static_cast<Plane>(plane), bl_row4x4,
                         bl_column4x4, has_left),
            kInterIntraToIntraMode[block.bp->prediction_parameters
                                       ->inter_intra_mode],
            tx_size);
#endif
      }
    }
    if (bp.is_inter) {
      int candidate_row = (block.row4x4 >> subsampling_y) << subsampling_y;
      int candidate_column = (block.column4x4 >> subsampling_x)
                             << subsampling_x;
      bool some_use_intra = false;
      for (int r = 0; r < (block_height4x4 << subsampling_y); ++r) {
        for (int c = 0; c < (block_width4x4 << subsampling_x); ++c) {
          auto* const bp = block_parameters_holder_.Find(candidate_row + r,
                                                         candidate_column + c);
          if (bp != nullptr && bp->reference_frame[0] == kReferenceFrameIntra) {
            some_use_intra = true;
            break;
          }
        }
        if (some_use_intra) break;
      }
      int prediction_width;
      int prediction_height;
      if (some_use_intra) {
        candidate_row = block.row4x4;
        candidate_column = block.column4x4;
        prediction_width = block_width;
        prediction_height = block_height;
      } else {
        prediction_width = kBlockWidthPixels[block.size] >> subsampling_x;
        prediction_height = kBlockHeightPixels[block.size] >> subsampling_y;
      }
      for (int r = 0, y = 0; y < block_height; y += prediction_height, ++r) {
        for (int c = 0, x = 0; x < block_width; x += prediction_width, ++c) {
          if (!InterPrediction(block, static_cast<Plane>(plane), base_x + x,
                               base_y + y, prediction_width, prediction_height,
                               candidate_row + r, candidate_column + c,
                               &is_local_valid, &local_warp_params)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void Tile::ComputeDeblockFilterLevel(const Block& block) {
  BlockParameters& bp = *block.bp;
  for (int plane = kPlaneY; plane < PlaneCount(); ++plane) {
    for (int i = kLoopFilterTypeVertical; i < kNumLoopFilterTypes; ++i) {
      bp.deblock_filter_level[plane][i] = LoopFilterMask::GetDeblockFilterLevel(
          frame_header_, bp, static_cast<Plane>(plane), i, delta_lf_);
    }
  }
}

bool Tile::ProcessBlock(int row4x4, int column4x4, BlockSize block_size,
                        ParameterTree* const tree,
                        SuperBlockBuffer* const sb_buffer) {
  // Do not process the block if the starting point is beyond the visible frame.
  // This is equivalent to the has_row/has_column check in the
  // decode_partition() section of the spec when partition equals
  // kPartitionHorizontal or kPartitionVertical.
  if (row4x4 >= frame_header_.rows4x4 ||
      column4x4 >= frame_header_.columns4x4) {
    return true;
  }
  Block block(*this, row4x4, column4x4, block_size, sb_buffer,
              tree->parameters());
  block.bp->size = block_size;
  block_parameters_holder_.FillCache(row4x4, column4x4, block_size,
                                     tree->parameters());
  block.bp->prediction_parameters =
      split_parse_and_decode_ ? std::unique_ptr<PredictionParameters>(
                                    new (std::nothrow) PredictionParameters())
                              : std::move(prediction_parameters_);
  if (block.bp->prediction_parameters == nullptr) return false;
  if (!DecodeModeInfo(block)) return false;
  ComputeDeblockFilterLevel(block);
  ReadPaletteTokens(block);
  DecodeTransformSize(block);
  const BlockParameters& bp = *block.bp;
  if (bp.skip) ResetEntropyContext(block);
  const int block_width4x4 = kNum4x4BlocksWide[block_size];
  const int block_height4x4 = kNum4x4BlocksHigh[block_size];
  if (split_parse_and_decode_) {
    if (!Residual(block, kProcessingModeParseOnly)) return false;
  } else {
    if (!ComputePrediction(block) ||
        !Residual(block, kProcessingModeParseAndDecode)) {
      return false;
    }
  }
  // If frame_header_.segmentation.enabled is false, bp.segment_id is 0 for all
  // blocks. We don't need to call save bp.segment_id in the current frame
  // because the current frame's segmentation map will be cleared to all 0s.
  //
  // If frame_header_.segmentation.enabled is true and
  // frame_header_.segmentation.update_map is false, we will copy the previous
  // frame's segmentation map to the current frame. So we don't need to call
  // save bp.segment_id in the current frame.
  if (frame_header_.segmentation.enabled &&
      frame_header_.segmentation.update_map) {
    const int x_limit =
        std::min(frame_header_.columns4x4 - column4x4, block_width4x4);
    const int y_limit =
        std::min(frame_header_.rows4x4 - row4x4, block_height4x4);
    current_frame_.segmentation_map()->FillBlock(row4x4, column4x4, x_limit,
                                                 y_limit, bp.segment_id);
  }
  if (!split_parse_and_decode_) {
    BuildBitMask(row4x4, column4x4, block_size);
    StoreMotionFieldMvsIntoCurrentFrame(block);
    prediction_parameters_ = std::move(block.bp->prediction_parameters);
  }
  return true;
}

bool Tile::DecodeBlock(ParameterTree* const tree,
                       SuperBlockBuffer* const sb_buffer) {
  const int row4x4 = tree->row4x4();
  const int column4x4 = tree->column4x4();
  if (row4x4 >= frame_header_.rows4x4 ||
      column4x4 >= frame_header_.columns4x4) {
    return true;
  }
  const BlockSize block_size = tree->block_size();
  Block block(*this, row4x4, column4x4, block_size, sb_buffer,
              tree->parameters());
  if (!ComputePrediction(block) ||
      !Residual(block, kProcessingModeDecodeOnly)) {
    return false;
  }
  BuildBitMask(row4x4, column4x4, block_size);
  StoreMotionFieldMvsIntoCurrentFrame(block);
  block.bp->prediction_parameters.reset(nullptr);
  return true;
}

bool Tile::ProcessPartition(int row4x4_start, int column4x4_start,
                            ParameterTree* const root,
                            SuperBlockBuffer* const sb_buffer) {
  std::vector<ParameterTree*> stack;
  stack.reserve(kDfsStackSize);

  // Set up the first iteration.
  ParameterTree* node = root;
  int row4x4 = row4x4_start;
  int column4x4 = column4x4_start;
  BlockSize block_size = SuperBlockSize();

  // DFS loop. If it sees a terminal node (leaf node), ProcessBlock is invoked.
  // Otherwise, the children are pushed into the stack for future processing.
  do {
    if (!stack.empty()) {
      // Set up subsequent iterations.
      node = stack.back();
      stack.pop_back();
      row4x4 = node->row4x4();
      column4x4 = node->column4x4();
      block_size = node->block_size();
    }
    if (row4x4 >= frame_header_.rows4x4 ||
        column4x4 >= frame_header_.columns4x4) {
      continue;
    }
    const int block_width4x4 = kNum4x4BlocksWide[block_size];
    assert(block_width4x4 == kNum4x4BlocksHigh[block_size]);
    const int half_block4x4 = block_width4x4 >> 1;
    const bool has_rows = (row4x4 + half_block4x4) < frame_header_.rows4x4;
    const bool has_columns =
        (column4x4 + half_block4x4) < frame_header_.columns4x4;
    Partition partition;
    if (!ReadPartition(row4x4, column4x4, block_size, has_rows, has_columns,
                       &partition)) {
      LIBGAV1_DLOG(ERROR, "Failed to read partition for row: %d column: %d",
                   row4x4, column4x4);
      return false;
    }
    const BlockSize sub_size = kSubSize[partition][block_size];
    // Section 6.10.4: It is a requirement of bitstream conformance that
    // get_plane_residual_size( subSize, 1 ) is not equal to BLOCK_INVALID
    // every time subSize is computed.
    if (sub_size == kBlockInvalid ||
        kPlaneResidualSize[sub_size]
                          [sequence_header_.color_config.subsampling_x]
                          [sequence_header_.color_config.subsampling_y] ==
            kBlockInvalid) {
      LIBGAV1_DLOG(
          ERROR,
          "Invalid sub-block/plane size for row: %d column: %d partition: "
          "%d block_size: %d sub_size: %d subsampling_x/y: %d, %d",
          row4x4, column4x4, partition, block_size, sub_size,
          sequence_header_.color_config.subsampling_x,
          sequence_header_.color_config.subsampling_y);
      return false;
    }
    node->SetPartitionType(partition);
    switch (partition) {
      case kPartitionNone:
        if (!ProcessBlock(row4x4, column4x4, sub_size, node, sb_buffer)) {
          return false;
        }
        break;
      case kPartitionSplit:
        // The children must be added in reverse order since a stack is being
        // used.
        for (int i = 3; i >= 0; --i) {
          ParameterTree* const child = node->children(i);
          assert(child != nullptr);
          stack.push_back(child);
        }
        break;
      case kPartitionHorizontal:
      case kPartitionVertical:
      case kPartitionHorizontalWithTopSplit:
      case kPartitionHorizontalWithBottomSplit:
      case kPartitionVerticalWithLeftSplit:
      case kPartitionVerticalWithRightSplit:
      case kPartitionHorizontal4:
      case kPartitionVertical4:
        for (int i = 0; i < 4; ++i) {
          ParameterTree* const child = node->children(i);
          // Once a null child is seen, all the subsequent children will also be
          // null.
          if (child == nullptr) break;
          if (!ProcessBlock(child->row4x4(), child->column4x4(),
                            child->block_size(), child, sb_buffer)) {
            return false;
          }
        }
        break;
    }
  } while (!stack.empty());
  return true;
}

void Tile::ResetLoopRestorationParams() {
  for (int plane = kPlaneY; plane < kMaxPlanes; ++plane) {
    for (int i = WienerInfo::kVertical; i <= WienerInfo::kHorizontal; ++i) {
      reference_unit_info_[plane].sgr_proj_info.multiplier[i] =
          kSgrProjDefaultMultiplier[i];
      for (int j = 0; j < kNumWienerCoefficients; ++j) {
        reference_unit_info_[plane].wiener_info.filter[i][j] =
            kWienerDefaultFilter[j];
      }
    }
  }
}

void Tile::ResetCdef(const int row4x4, const int column4x4) {
  if (cdef_index_[0] == nullptr) return;
  const int row = DivideBy16(row4x4);
  const int column = DivideBy16(column4x4);
  cdef_index_[row][column] = -1;
  if (sequence_header_.use_128x128_superblock) {
    const int cdef_size4x4 = kNum4x4BlocksWide[kBlock64x64];
    const int border_row = DivideBy16(row4x4 + cdef_size4x4);
    const int border_column = DivideBy16(column4x4 + cdef_size4x4);
    cdef_index_[row][border_column] = -1;
    cdef_index_[border_row][column] = -1;
    cdef_index_[border_row][border_column] = -1;
  }
}

bool Tile::ProcessSuperBlock(int row4x4, int column4x4, int block_width4x4,
                             SuperBlockBuffer* const sb_buffer,
                             ProcessingMode mode) {
  const bool parsing =
      mode == kProcessingModeParseOnly || mode == kProcessingModeParseAndDecode;
  const bool decoding = mode == kProcessingModeDecodeOnly ||
                        mode == kProcessingModeParseAndDecode;
  if (parsing) {
    read_deltas_ = frame_header_.delta_q.present;
    ResetCdef(row4x4, column4x4);
  }
  if (decoding) {
    memset(sb_buffer->block_decoded, 0,
           sizeof(sb_buffer->block_decoded));  // Section 5.11.3.
    sb_buffer->block_decoded_width_threshold = column4x4_end_ - column4x4;
    sb_buffer->block_decoded_height_threshold = row4x4_end_ - row4x4;
  }
  const BlockSize block_size = SuperBlockSize();
  if (parsing) {
    ReadLoopRestorationCoefficients(row4x4, column4x4, block_size);
  }
  const int row = row4x4 / block_width4x4;
  const int column = column4x4 / block_width4x4;
  if (parsing && decoding) {
    sb_buffer->residual = residual_buffer_.get();
    if (!ProcessPartition(row4x4, column4x4,
                          block_parameters_holder_.Tree(row, column),
                          sb_buffer)) {
      LIBGAV1_DLOG(ERROR, "Error decoding partition row: %d column: %d", row4x4,
                   column4x4);
      return false;
    }
    return true;
  }
  const int sb_row_index = (row4x4 - row4x4_start_) / block_width4x4;
  const int sb_column_index = (column4x4 - column4x4_start_) / block_width4x4;
  if (parsing) {
    residual_buffer_threaded_[sb_row_index][sb_column_index] =
        residual_buffer_pool_->Get();
    if (residual_buffer_threaded_[sb_row_index][sb_column_index] == nullptr) {
      LIBGAV1_DLOG(ERROR, "Failed to get residual buffer.");
      return false;
    }
    sb_buffer->residual =
        residual_buffer_threaded_[sb_row_index][sb_column_index]->buffer.get();
    sb_buffer->transform_parameters =
        &residual_buffer_threaded_[sb_row_index][sb_column_index]
             ->transform_parameters;
    if (!ProcessPartition(row4x4, column4x4,
                          block_parameters_holder_.Tree(row, column),
                          sb_buffer)) {
      LIBGAV1_DLOG(ERROR, "Error parsing partition row: %d column: %d", row4x4,
                   column4x4);
      return false;
    }
  } else {
    sb_buffer->residual =
        residual_buffer_threaded_[sb_row_index][sb_column_index]->buffer.get();
    sb_buffer->transform_parameters =
        &residual_buffer_threaded_[sb_row_index][sb_column_index]
             ->transform_parameters;
    if (!DecodeSuperBlock(block_parameters_holder_.Tree(row, column),
                          sb_buffer)) {
      LIBGAV1_DLOG(ERROR, "Error decoding superblock row: %d column: %d",
                   row4x4, column4x4);
      return false;
    }
    residual_buffer_pool_->Release(
        std::move(residual_buffer_threaded_[sb_row_index][sb_column_index]));
  }
  return true;
}

bool Tile::DecodeSuperBlock(ParameterTree* const tree,
                            SuperBlockBuffer* const sb_buffer) {
  std::vector<ParameterTree*> stack;
  stack.reserve(kDfsStackSize);
  stack.push_back(tree);
  while (!stack.empty()) {
    ParameterTree* const node = stack.back();
    stack.pop_back();
    if (node->partition() != kPartitionNone) {
      for (int i = 3; i >= 0; --i) {
        if (node->children(i) == nullptr) continue;
        stack.push_back(node->children(i));
      }
      continue;
    }
    if (!DecodeBlock(node, sb_buffer)) {
      LIBGAV1_DLOG(ERROR, "Error decoding block row: %d column: %d",
                   node->row4x4(), node->column4x4());
      return false;
    }
  }
  return true;
}

void Tile::ReadLoopRestorationCoefficients(int row4x4, int column4x4,
                                           BlockSize block_size) {
  if (frame_header_.allow_intrabc) return;
  LoopRestorationInfo* const restoration_info = post_filter_.restoration_info();
  const bool is_superres_scaled =
      frame_header_.width != frame_header_.upscaled_width;
  for (int plane = kPlaneY; plane < PlaneCount(); ++plane) {
    LoopRestorationUnitInfo unit_info;
    if (restoration_info->PopulateUnitInfoForSuperBlock(
            static_cast<Plane>(plane), block_size, is_superres_scaled,
            frame_header_.superres_scale_denominator, row4x4, column4x4,
            &unit_info)) {
      for (int unit_row = unit_info.row_start; unit_row < unit_info.row_end;
           ++unit_row) {
        for (int unit_column = unit_info.column_start;
             unit_column < unit_info.column_end; ++unit_column) {
          const int unit_id = unit_row * restoration_info->num_horizontal_units(
                                             static_cast<Plane>(plane)) +
                              unit_column;
          restoration_info->ReadUnitCoefficients(
              &reader_, &symbol_decoder_context_, static_cast<Plane>(plane),
              unit_id, &reference_unit_info_);
        }
      }
    }
  }
}

void Tile::BuildBitMask(int row4x4, int column4x4, BlockSize block_size) {
  if (!post_filter_.DoDeblock()) return;
  const int block_width4x4 = kNum4x4BlocksWide[block_size];
  const int block_height4x4 = kNum4x4BlocksHigh[block_size];
  if (block_width4x4 <= kNum4x4BlocksWide[kBlock64x64] &&
      block_height4x4 <= kNum4x4BlocksHigh[kBlock64x64]) {
    BuildBitMaskHelper(row4x4, column4x4, block_size, true, true);
  } else {
    for (int y = 0; y < block_height4x4; y += kNum4x4BlocksHigh[kBlock64x64]) {
      for (int x = 0; x < block_width4x4; x += kNum4x4BlocksWide[kBlock64x64]) {
        BuildBitMaskHelper(row4x4 + y, column4x4 + x, kBlock64x64, x == 0,
                           y == 0);
      }
    }
  }
}

void Tile::BuildBitMaskHelper(int row4x4, int column4x4, BlockSize block_size,
                              const bool is_vertical_block_border,
                              const bool is_horizontal_block_border) {
  const int block_width4x4 = kNum4x4BlocksWide[block_size];
  const int block_height4x4 = kNum4x4BlocksHigh[block_size];
  BlockParameters& bp = *block_parameters_holder_.Find(row4x4, column4x4);
  const bool skip = bp.skip && bp.is_inter;
  LoopFilterMask* const masks = post_filter_.masks();
  const int unit_id = DivideBy16(row4x4) * masks->num_64x64_blocks_per_row() +
                      DivideBy16(column4x4);
  const int row_limit = row4x4 + block_height4x4;
  const int column_limit = column4x4 + block_width4x4;
  const TransformSize current_block_uv_tx_size = GetTransformSize(
      frame_header_.segmentation.lossless[bp.segment_id], block_size, kPlaneU,
      kNumTransformSizes,  // This parameter is unused when plane != Y.
      SubsamplingX(kPlaneU), SubsamplingY(kPlaneU));

  for (int plane = kPlaneY; plane < PlaneCount(); ++plane) {
    // For U and V planes, do not build bit masks if level == 0.
    if (plane > kPlaneY && frame_header_.loop_filter.level[plane + 1] == 0) {
      continue;
    }
    // Build bit mask for vertical edges.
    const int subsampling_x = SubsamplingX(static_cast<Plane>(plane));
    const int subsampling_y = SubsamplingY(static_cast<Plane>(plane));
    const int plane_width =
        RightShiftWithRounding(frame_header_.width, subsampling_x);
    const int plane_height =
        RightShiftWithRounding(frame_header_.height, subsampling_y);
    const int vertical_step = 1 << subsampling_y;
    const int horizontal_step = 1 << subsampling_x;
    const int row_start = GetDeblockPosition(row4x4, subsampling_y);
    const int column_start = GetDeblockPosition(column4x4, subsampling_x);
    if (row_start >= row4x4 + block_height4x4 ||
        MultiplyBy4(row_start >> subsampling_y) >= plane_height ||
        column_start >= column4x4 + block_width4x4 ||
        MultiplyBy4(column_start >> subsampling_x) >= plane_width) {
      continue;
    }
    const BlockParameters& bp =
        *block_parameters_holder_.Find(row_start, column_start);
    const uint8_t vertical_level =
        bp.deblock_filter_level[plane][kLoopFilterTypeVertical];

    for (int row = row_start;
         row < row_limit && MultiplyBy4(row >> subsampling_y) < plane_height &&
         row < frame_header_.rows4x4;
         row += vertical_step) {
      for (int column = column_start;
           column < column_limit &&
           MultiplyBy4(column >> subsampling_x) < plane_width &&
           column < frame_header_.columns4x4;) {
        const TransformSize tx_size = (plane == kPlaneY)
                                          ? inter_transform_sizes_[row][column]
                                          : current_block_uv_tx_size;
        // (1). Don't filter frame boundary.
        // (2). For tile boundary, we don't know whether the previous tile is
        // available or not, thus we handle it after all tiles are decoded.
        const bool is_vertical_border =
            (column == column_start) && is_vertical_block_border;
        if (column == GetDeblockPosition(column4x4_start_, subsampling_x) ||
            (skip && !is_vertical_border)) {
          column += kNum4x4BlocksWide[tx_size] << subsampling_x;
          continue;
        }

        // bp_left is the parameter of the left prediction block which
        // is guaranteed to be inside the tile.
        const BlockParameters& bp_left =
            *block_parameters_holder_.Find(row, column - horizontal_step);
        const uint8_t left_level =
            is_vertical_border
                ? bp_left.deblock_filter_level[plane][kLoopFilterTypeVertical]
                : vertical_level;
        // We don't have to check if the left block is skipped or not,
        // because if the current transform block is on the edge of the coding
        // block, is_vertical_border is true; if it's not on the edge,
        // left skip is equal to skip.
        if (vertical_level != 0 || left_level != 0) {
          const TransformSize left_tx_size = GetTransformSize(
              frame_header_.segmentation.lossless[bp_left.segment_id],
              bp_left.size, static_cast<Plane>(plane),
              inter_transform_sizes_[row][column - horizontal_step],
              subsampling_x, subsampling_y);
          // 0: 4x4, 1: 8x8, 2: 16x16.
          const int transform_size_id =
              std::min({kTransformWidthLog2[tx_size] - 2,
                        kTransformWidthLog2[left_tx_size] - 2, 2});
          const int r = row & (kNum4x4InLoopFilterMaskUnit - 1);
          const int c = column & (kNum4x4InLoopFilterMaskUnit - 1);
          const int shift = LoopFilterMask::GetShift(r, c);
          const int index = LoopFilterMask::GetIndex(r);
          const auto mask = static_cast<uint64_t>(1) << shift;
          masks->SetLeft(mask, unit_id, plane, transform_size_id, index);
          const uint8_t current_level =
              (vertical_level == 0) ? left_level : vertical_level;
          masks->SetLevel(current_level, unit_id, plane,
                          kLoopFilterTypeVertical,
                          LoopFilterMask::GetLevelOffset(r, c));
        }
        column += kNum4x4BlocksWide[tx_size] << subsampling_x;
      }
    }

    // Build bit mask for horizontal edges.
    const uint8_t horizontal_level =
        bp.deblock_filter_level[plane][kLoopFilterTypeHorizontal];
    for (int column = column_start;
         column < column_limit &&
         MultiplyBy4(column >> subsampling_x) < plane_width &&
         column < frame_header_.columns4x4;
         column += horizontal_step) {
      for (int row = row_start;
           row < row_limit &&
           MultiplyBy4(row >> subsampling_y) < plane_height &&
           row < frame_header_.rows4x4;) {
        const TransformSize tx_size = (plane == kPlaneY)
                                          ? inter_transform_sizes_[row][column]
                                          : current_block_uv_tx_size;

        // (1). Don't filter frame boundary.
        // (2). For tile boundary, we don't know whether the previous tile is
        // available or not, thus we handle it after all tiles are decoded.
        const bool is_horizontal_border =
            (row == row_start) && is_horizontal_block_border;
        if (row == GetDeblockPosition(row4x4_start_, subsampling_y) ||
            (skip && !is_horizontal_border)) {
          row += kNum4x4BlocksHigh[tx_size] << subsampling_y;
          continue;
        }

        // bp_top is the parameter of the top prediction block which is
        // guaranteed to be inside the tile.
        const BlockParameters& bp_top =
            *block_parameters_holder_.Find(row - vertical_step, column);
        const uint8_t top_level =
            is_horizontal_border
                ? bp_top.deblock_filter_level[plane][kLoopFilterTypeHorizontal]
                : horizontal_level;
        // We don't have to check it the top block is skippped or not,
        // because if the current transform block is on the edge of the coding
        // block, is_horizontal_border is true; if it's not on the edge,
        // top skip is equal to skip.
        if (horizontal_level != 0 || top_level != 0) {
          const TransformSize top_tx_size = GetTransformSize(
              frame_header_.segmentation.lossless[bp_top.segment_id],
              bp_top.size, static_cast<Plane>(plane),
              inter_transform_sizes_[row - vertical_step][column],
              subsampling_x, subsampling_y);
          // 0: 4x4, 1: 8x8, 2: 16x16.
          const int transform_size_id =
              std::min({kTransformHeightLog2[tx_size] - 2,
                        kTransformHeightLog2[top_tx_size] - 2, 2});
          const int r = row & (kNum4x4InLoopFilterMaskUnit - 1);
          const int c = column & (kNum4x4InLoopFilterMaskUnit - 1);
          const int shift = LoopFilterMask::GetShift(r, c);
          const int index = LoopFilterMask::GetIndex(r);
          const auto mask = static_cast<uint64_t>(1) << shift;
          masks->SetTop(mask, unit_id, plane, transform_size_id, index);
          const uint8_t current_level =
              (horizontal_level == 0) ? top_level : horizontal_level;
          masks->SetLevel(current_level, unit_id, plane,
                          kLoopFilterTypeHorizontal,
                          LoopFilterMask::GetLevelOffset(r, c));
        }
        row += kNum4x4BlocksHigh[tx_size] << subsampling_y;
      }
    }
  }
}

void Tile::StoreMotionFieldMvsIntoCurrentFrame(const Block& block) {
  // The largest reference MV component that can be saved.
  constexpr int kRefMvsLimit = (1 << 12) - 1;
  const BlockParameters& bp = *block.bp;
  ReferenceFrameType reference_frame_to_store = kReferenceFrameNone;
  MotionVector mv_to_store = {};
  for (int i = 1; i >= 0; --i) {
    if (bp.reference_frame[i] > kReferenceFrameIntra &&
        std::abs(bp.mv[i].mv[MotionVector::kRow]) <= kRefMvsLimit &&
        std::abs(bp.mv[i].mv[MotionVector::kColumn]) <= kRefMvsLimit &&
        GetRelativeDistance(
            reference_order_hint_
                [frame_header_.reference_frame_index[bp.reference_frame[i] -
                                                     kReferenceFrameLast]],
            frame_header_.order_hint, sequence_header_.enable_order_hint,
            sequence_header_.order_hint_bits) < 0) {
      reference_frame_to_store = bp.reference_frame[i];
      mv_to_store = bp.mv[i];
      break;
    }
  }
  // Iterate over odd rows/columns beginning at the first odd row/column for the
  // block. It is done this way because motion field mvs are only needed at a
  // 8x8 granularity.
  const int row_start = block.row4x4 | 1;
  const int row_limit = std::min(block.row4x4 + kNum4x4BlocksHigh[block.size],
                                 frame_header_.rows4x4);
  const int column_start = block.column4x4 | 1;
  const int column_limit =
      std::min(block.column4x4 + kNum4x4BlocksWide[block.size],
               frame_header_.columns4x4);
  for (int row = row_start; row < row_limit; row += 2) {
    const int row_index = DivideBy2(row);
    ReferenceFrameType* const reference_frame_row_start =
        current_frame_.motion_field_reference_frame(row_index,
                                                    DivideBy2(column_start));
    static_assert(sizeof(reference_frame_to_store) == sizeof(int8_t), "");
    memset(reference_frame_row_start, reference_frame_to_store,
           DivideBy2(column_limit - column_start + 1));
    if (reference_frame_to_store <= kReferenceFrameIntra) continue;
    for (int column = column_start; column < column_limit; column += 2) {
      MotionVector* const mv =
          current_frame_.motion_field_mv(row_index, DivideBy2(column));
      *mv = mv_to_store;
    }
  }
}

}  // namespace libgav1
