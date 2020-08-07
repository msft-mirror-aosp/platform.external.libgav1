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
#include "src/post_filter.h"
#include "src/utils/blocking_counter.h"
#include "src/utils/compiler_attributes.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace {

constexpr int kStep64x64 = 16;  // =64/4.
constexpr int kCdefSkip = 8;

constexpr uint8_t kCdefUvDirection[2][2][8] = {
    {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 2, 2, 2, 3, 4, 6, 0}},
    {{7, 0, 2, 4, 5, 6, 6, 6}, {0, 1, 2, 3, 4, 5, 6, 7}}};

constexpr int kCdefBorderRows[2][4] = {{0, 1, 62, 63}, {0, 1, 30, 31}};

template <typename Pixel>
void CopyRowForCdef(const Pixel* src, int block_width, int unit_width,
                    bool is_frame_left, bool is_frame_right,
                    uint16_t* const dst) {
  if (sizeof(src[0]) == sizeof(dst[0])) {
    if (is_frame_left) {
      Memset(dst - kCdefBorder, kCdefLargeValue, kCdefBorder);
    } else {
      memcpy(dst - kCdefBorder, src - kCdefBorder,
             kCdefBorder * sizeof(dst[0]));
    }
    memcpy(dst, src, block_width * sizeof(dst[0]));
    if (is_frame_right) {
      Memset(dst + block_width, kCdefLargeValue,
             unit_width + kCdefBorder - block_width);
    } else {
      memcpy(dst + block_width, src + block_width,
             (unit_width + kCdefBorder - block_width) * sizeof(dst[0]));
    }
    return;
  }
  for (int x = -kCdefBorder; x < 0; ++x) {
    dst[x] = is_frame_left ? static_cast<uint16_t>(kCdefLargeValue) : src[x];
  }
  for (int x = 0; x < block_width; ++x) {
    dst[x] = src[x];
  }
  for (int x = block_width; x < unit_width + kCdefBorder; ++x) {
    dst[x] = is_frame_right ? static_cast<uint16_t>(kCdefLargeValue) : src[x];
  }
}

// For |height| rows, copy |width| pixels of size |pixel_size| from |src| to
// |dst|.
void CopyPixels(const uint8_t* src, int src_stride, uint8_t* dst,
                int dst_stride, int width, int height, size_t pixel_size) {
  int y = height;
  do {
    memcpy(dst, src, width * pixel_size);
    src += src_stride;
    dst += dst_stride;
  } while (--y != 0);
}

}  // namespace

void PostFilter::SetupCdefBorder(int row4x4) {
  assert(row4x4 >= 0);
  assert(DoCdef());
  int plane = kPlaneY;
  do {
    const ptrdiff_t src_stride = frame_buffer_.stride(plane);
    const ptrdiff_t dst_stride = cdef_border_.stride(plane);
    const int row_offset = DivideBy4(row4x4);
    const int num_pixels = SubsampledValue(
        MultiplyBy4(frame_header_.columns4x4), subsampling_x_[plane]);
    const int plane_height = SubsampledValue(MultiplyBy4(frame_header_.rows4x4),
                                             subsampling_y_[plane]);
    for (int i = 0; i < 4; ++i) {
      const int row = kCdefBorderRows[subsampling_y_[plane]][i];
      const int absolute_row =
          (MultiplyBy4(row4x4) >> subsampling_y_[plane]) + row;
      if (absolute_row >= plane_height) break;
      const uint8_t* src =
          GetSourceBuffer(static_cast<Plane>(plane), row4x4, 0) +
          row * src_stride;
      uint8_t* dst = cdef_border_.data(plane) + dst_stride * (row_offset + i);
      memcpy(dst, src, num_pixels * pixel_size_);
    }
  } while (++plane < planes_);
}

template <typename Pixel>
void PostFilter::PrepareCdefBlock(int block_width4x4, int block_height4x4,
                                  int row4x4, int column4x4,
                                  uint16_t* cdef_source, ptrdiff_t cdef_stride,
                                  const bool y_plane) {
  assert(y_plane || planes_ == kMaxPlanes);
  const int max_planes = y_plane ? 1 : kMaxPlanes;
  const int8_t subsampling_x = y_plane ? 0 : subsampling_x_[kPlaneU];
  const int8_t subsampling_y = y_plane ? 0 : subsampling_y_[kPlaneU];
  const int start_x = MultiplyBy4(column4x4) >> subsampling_x;
  const int start_y = MultiplyBy4(row4x4) >> subsampling_y;
  const int plane_width = SubsampledValue(width_, subsampling_x);
  const int plane_height = SubsampledValue(height_, subsampling_y);
  const int block_width = MultiplyBy4(block_width4x4) >> subsampling_x;
  const int block_height = MultiplyBy4(block_height4x4) >> subsampling_y;
  // unit_width, unit_height are the same as block_width, block_height unless
  // it reaches the frame boundary, where block_width < 64 or
  // block_height < 64. unit_width, unit_height guarantee we build blocks on
  // a multiple of 8.
  const int unit_width = Align(block_width, 8 >> subsampling_x);
  const int unit_height = Align(block_height, 8 >> subsampling_y);
  const bool is_frame_left = column4x4 == 0;
  const bool is_frame_right = start_x + block_width >= plane_width;
  const bool is_frame_top = row4x4 == 0;
  const bool is_frame_bottom = start_y + block_height >= plane_height;
  const int y_offset = is_frame_top ? 0 : kCdefBorder;
  const int cdef_border_row_offset = DivideBy4(row4x4) - (is_frame_top ? 0 : 2);

  for (int plane = y_plane ? kPlaneY : kPlaneU; plane < max_planes; ++plane) {
    uint16_t* cdef_src = cdef_source + static_cast<int>(plane == kPlaneV) *
                                           kCdefUnitSizeWithBorders *
                                           kCdefUnitSizeWithBorders;
    const int src_stride = frame_buffer_.stride(plane) / sizeof(Pixel);
    const Pixel* src_buffer =
        reinterpret_cast<const Pixel*>(source_buffer_[plane]) +
        (start_y - y_offset) * src_stride + start_x;
    const int cdef_border_stride = cdef_border_.stride(plane) / sizeof(Pixel);
    const Pixel* cdef_border =
        (thread_pool_ == nullptr)
            ? nullptr
            : reinterpret_cast<const Pixel*>(cdef_border_.data(plane)) +
                  cdef_border_row_offset * cdef_border_stride + start_x;

    // All the copying code will use negative indices for populating the left
    // border. So the starting point is set to kCdefBorder.
    cdef_src += kCdefBorder;

    // Copy the top 2 rows as follows;
    // If is_frame_top is true, both the rows are set to kCdefLargeValue.
    // Otherwise:
    //   If multi-threaded filtering is off, the rows are copied from
    //   |src_buffer|.
    //   Otherwise, the rows are copied from |cdef_border|.
    if (is_frame_top) {
      for (int y = 0; y < kCdefBorder; ++y) {
        Memset(cdef_src - kCdefBorder, kCdefLargeValue,
               unit_width + 2 * kCdefBorder);
        cdef_src += cdef_stride;
      }
    } else {
      const Pixel* top_border =
          (thread_pool_ == nullptr) ? src_buffer : cdef_border;
      const int top_border_stride =
          (thread_pool_ == nullptr) ? src_stride : cdef_border_stride;
      for (int y = 0; y < kCdefBorder; ++y) {
        CopyRowForCdef(top_border, block_width, unit_width, is_frame_left,
                       is_frame_right, cdef_src);
        top_border += top_border_stride;
        cdef_src += cdef_stride;
        // We need to increment |src_buffer| and |cdef_border| in this loop to
        // set them up for the subsequent loops below.
        src_buffer += src_stride;
        cdef_border += cdef_border_stride;
      }
    }

    // Copy the body as follows;
    // If multi-threaded filtering is off or if is_frame_bottom is true, all the
    // rows are copied from |src_buffer|.
    // Otherwise, the first |block_height|-kCdefBorder rows are copied from
    // |src_buffer| and the last kCdefBorder rows are coped from |cdef_border|.
    int y = block_height;
    const int y_threshold =
        (thread_pool_ == nullptr || is_frame_bottom) ? 0 : kCdefBorder;
    do {
      CopyRowForCdef(src_buffer, block_width, unit_width, is_frame_left,
                     is_frame_right, cdef_src);
      cdef_src += cdef_stride;
      src_buffer += src_stride;
    } while (--y != y_threshold);

    if (y > 0) {
      assert(y == kCdefBorder);
      // |cdef_border| now points to the top 2 rows of the current block. For
      // the next loop, we need it to point to the bottom 2 rows of the
      // current block. So increment it by 2 rows.
      cdef_border += MultiplyBy2(cdef_border_stride);
      for (int i = 0; i < kCdefBorder; ++i) {
        CopyRowForCdef(cdef_border, block_width, unit_width, is_frame_left,
                       is_frame_right, cdef_src);
        cdef_src += cdef_stride;
        cdef_border += cdef_border_stride;
      }
    }

    // Copy the bottom 2 rows as follows;
    // If is_frame_bottom is true, both the rows are set to kCdefLargeValue.
    // Otherwise:
    //   If multi-threaded filtering is off, the rows are copied from
    //   |src_buffer|.
    //   Otherwise, the rows are copied from |cdef_border|.
    y = 0;
    if (is_frame_bottom) {
      do {
        Memset(cdef_src - kCdefBorder, kCdefLargeValue,
               unit_width + 2 * kCdefBorder);
        cdef_src += cdef_stride;
      } while (++y < kCdefBorder + unit_height - block_height);
    } else {
      const Pixel* bottom_border =
          (thread_pool_ == nullptr) ? src_buffer : cdef_border;
      const int bottom_border_stride =
          (thread_pool_ == nullptr) ? src_stride : cdef_border_stride;
      do {
        CopyRowForCdef(bottom_border, block_width, unit_width, is_frame_left,
                       is_frame_right, cdef_src);
        bottom_border += bottom_border_stride;
        cdef_src += cdef_stride;
      } while (++y < kCdefBorder + unit_height - block_height);
    }
  }
}

void PostFilter::CopyBlockForCdefHelper(Plane plane,
                                        const uint8_t* src_buffer_row_base,
                                        uint8_t* cdef_buffer_row_base,
                                        int block_width4x4, int block_height4x4,
                                        int row4x4_start, int column4x4_start,
                                        bool is_frame_bottom) {
  if (thread_pool_ == nullptr || is_frame_bottom) {
    CopyPixels(src_buffer_row_base, frame_buffer_.stride(plane),
               cdef_buffer_row_base, frame_buffer_.stride(plane),
               MultiplyBy4(block_width4x4) >> subsampling_x_[plane],
               MultiplyBy4(block_height4x4) >> subsampling_y_[plane],
               pixel_size_);
    return;
  }
  // Copy everything except the last kCdefBorder rows from
  // |src_buffer_row_base|.
  const int copy_height =
      (MultiplyBy4(block_height4x4) >> subsampling_y_[plane]) - kCdefBorder;
  CopyPixels(src_buffer_row_base, frame_buffer_.stride(plane),
             cdef_buffer_row_base, frame_buffer_.stride(plane),
             MultiplyBy4(block_width4x4) >> subsampling_x_[plane], copy_height,
             pixel_size_);
  // Copy the last kCdefBorder rows from |cdef_border_|.
  const int start_x = MultiplyBy4(column4x4_start) >> subsampling_x_[plane];
  const int cdef_border_row_offset = DivideBy4(row4x4_start) + 2;
  const uint8_t* cdef_border_base =
      cdef_border_.data(plane) +
      cdef_border_row_offset * cdef_border_.stride(plane) +
      start_x * pixel_size_;
  uint8_t* cdef_dst =
      cdef_buffer_row_base + copy_height * frame_buffer_.stride(plane);
  CopyPixels(cdef_border_base, cdef_border_.stride(plane), cdef_dst,
             frame_buffer_.stride(plane),
             MultiplyBy4(block_width4x4) >> subsampling_x_[plane], kCdefBorder,
             pixel_size_);
}

void PostFilter::CopyBlockForCdefHelper(const uint8_t* src_buffer,
                                        int src_stride, uint8_t* cdef_buffer,
                                        int cdef_stride,
                                        const uint16_t* cdef_src,
                                        int block_width, int block_height) {
  if (thread_pool_ == nullptr) {
    CopyPixels(src_buffer, src_stride, cdef_buffer, cdef_stride, block_width,
               block_height, pixel_size_);
    return;
  }
#if LIBGAV1_MAX_BITDEPTH >= 10
  if (pixel_size_ == 2) {
    CopyPixels(reinterpret_cast<const uint8_t*>(cdef_src),
               kCdefUnitSizeWithBorders * pixel_size_, cdef_buffer, cdef_stride,
               block_width, block_height, pixel_size_);
    return;
  }
#endif
  uint8_t* dst = cdef_buffer;
  for (int y = 0; y < block_height; ++y) {
    for (int x = 0; x < block_width; ++x) {
      dst[x] = cdef_src[x];
    }
    dst += cdef_stride;
    cdef_src += kCdefUnitSizeWithBorders;
  }
}

template <typename Pixel>
void PostFilter::ApplyCdefForOneUnit(uint16_t* cdef_block, const int index,
                                     const int block_width4x4,
                                     const int block_height4x4,
                                     const int row4x4_start,
                                     const int column4x4_start) {
  // Cdef operates in 8x8 blocks (4x4 for chroma with subsampling).
  static constexpr int kStep = 8;
  static constexpr int kStep4x4 = 2;

  int cdef_buffer_row_base_stride[kMaxPlanes];
  uint8_t* cdef_buffer_row_base[kMaxPlanes];
  int src_buffer_row_base_stride[kMaxPlanes];
  const uint8_t* src_buffer_row_base[kMaxPlanes];
  const uint16_t* cdef_src_row_base[kMaxPlanes];
  int cdef_src_row_base_stride[kMaxPlanes];
  int column_step[kMaxPlanes];
  assert(planes_ >= 1);
  int plane = kPlaneY;
  do {
    cdef_buffer_row_base[plane] =
        GetCdefBuffer(static_cast<Plane>(plane), row4x4_start, column4x4_start);
    cdef_buffer_row_base_stride[plane] =
        frame_buffer_.stride(plane) * (kStep >> subsampling_y_[plane]);
    src_buffer_row_base[plane] = GetSourceBuffer(static_cast<Plane>(plane),
                                                 row4x4_start, column4x4_start);
    src_buffer_row_base_stride[plane] =
        frame_buffer_.stride(plane) * (kStep >> subsampling_y_[plane]);
    cdef_src_row_base[plane] =
        cdef_block +
        static_cast<int>(plane == kPlaneV) * kCdefUnitSizeWithBorders *
            kCdefUnitSizeWithBorders +
        kCdefBorder * kCdefUnitSizeWithBorders + kCdefBorder;
    cdef_src_row_base_stride[plane] =
        kCdefUnitSizeWithBorders * (kStep >> subsampling_y_[plane]);
    column_step[plane] = (kStep >> subsampling_x_[plane]) * sizeof(Pixel);
  } while (++plane < planes_);

  const bool is_frame_bottom =
      row4x4_start + kStep64x64 >= frame_header_.rows4x4;

  if (index == -1) {
    int plane = kPlaneY;
    do {
      CopyBlockForCdefHelper(
          static_cast<Plane>(plane), src_buffer_row_base[plane],
          cdef_buffer_row_base[plane], block_width4x4, block_height4x4,
          row4x4_start, column4x4_start, is_frame_bottom);
    } while (++plane < planes_);
    return;
  }

  PrepareCdefBlock<Pixel>(block_width4x4, block_height4x4, row4x4_start,
                          column4x4_start, cdef_block, kCdefUnitSizeWithBorders,
                          true);

  // Stored direction used during the u/v pass.  If bit 3 is set, then block is
  // a skip.
  int direction_y[8 * 8];
  int y_index = 0;

  const uint8_t y_primary_strength =
      frame_header_.cdef.y_primary_strength[index];
  const uint8_t y_secondary_strength =
      frame_header_.cdef.y_secondary_strength[index];
  // y_strength_index is 0 for both primary and secondary strengths being
  // non-zero, 1 for primary only, 2 for secondary only. This will be updated
  // with y_primary_strength after variance is applied.
  int y_strength_index = static_cast<int>(y_secondary_strength == 0);

  const bool compute_direction_and_variance =
      (y_primary_strength | frame_header_.cdef.uv_primary_strength[index]) != 0;
  BlockParameters* const* bp_row0_base =
      block_parameters_.Address(row4x4_start, column4x4_start);
  BlockParameters* const* bp_row1_base =
      bp_row0_base + block_parameters_.columns4x4();
  const int bp_stride = MultiplyBy2(block_parameters_.columns4x4());
  int row4x4 = row4x4_start;
  do {
    uint8_t* cdef_buffer_base = cdef_buffer_row_base[kPlaneY];
    const uint8_t* src_buffer_base = src_buffer_row_base[kPlaneY];
    const uint16_t* cdef_src_base = cdef_src_row_base[kPlaneY];
    BlockParameters* const* bp0 = bp_row0_base;
    BlockParameters* const* bp1 = bp_row1_base;
    int column4x4 = column4x4_start;
    do {
      const int block_width = kStep;
      const int block_height = kStep;
      const int cdef_stride = frame_buffer_.stride(kPlaneY);
      uint8_t* const cdef_buffer = cdef_buffer_base;
      const uint16_t* const cdef_src = cdef_src_base;
      const int src_stride = frame_buffer_.stride(kPlaneY);
      const uint8_t* const src_buffer = src_buffer_base;

      const bool skip = (*bp0)->skip && (*(bp0 + 1))->skip && (*bp1)->skip &&
                        (*(bp1 + 1))->skip;

      if (skip) {  // No cdef filtering.
        direction_y[y_index] = kCdefSkip;
        CopyBlockForCdefHelper(src_buffer, src_stride, cdef_buffer, cdef_stride,
                               cdef_src, block_width, block_height);
      } else {
        // Zero out residual skip flag.
        direction_y[y_index] = 0;

        int variance = 0;
        if (compute_direction_and_variance) {
          if (thread_pool_ == nullptr ||
              row4x4 + kStep4x4 < row4x4_start + block_height4x4) {
            dsp_.cdef_direction(src_buffer, src_stride, &direction_y[y_index],
                                &variance);
          } else if (sizeof(Pixel) == 2) {
            dsp_.cdef_direction(cdef_src, kCdefUnitSizeWithBorders * 2,
                                &direction_y[y_index], &variance);
          } else {
            // If we are in the last row4x4 for this unit, then the last two
            // input rows have to come from |cdef_border_|. Since we already
            // have |cdef_src| populated correctly, use that as the input
            // for the direction process.
            uint8_t direction_src[8][8];
            const uint16_t* cdef_src_line = cdef_src;
            for (auto& direction_src_line : direction_src) {
              for (int i = 0; i < 8; ++i) {
                direction_src_line[i] = cdef_src_line[i];
              }
              cdef_src_line += kCdefUnitSizeWithBorders;
            }
            dsp_.cdef_direction(direction_src, 8, &direction_y[y_index],
                                &variance);
          }
        }
        const int direction =
            (y_primary_strength == 0) ? 0 : direction_y[y_index];
        const int variance_strength =
            ((variance >> 6) != 0) ? std::min(FloorLog2(variance >> 6), 12) : 0;
        const uint8_t primary_strength =
            (variance != 0)
                ? (y_primary_strength * (4 + variance_strength) + 8) >> 4
                : 0;
        if ((primary_strength | y_secondary_strength) == 0) {
          CopyBlockForCdefHelper(src_buffer, src_stride, cdef_buffer,
                                 cdef_stride, cdef_src, block_width,
                                 block_height);
        } else {
          const int strength_index =
              y_strength_index | (static_cast<int>(primary_strength == 0) << 1);
          dsp_.cdef_filters[1][strength_index](
              cdef_src, kCdefUnitSizeWithBorders, block_height,
              primary_strength, y_secondary_strength,
              frame_header_.cdef.damping, direction, cdef_buffer, cdef_stride);
        }
      }
      cdef_buffer_base += column_step[kPlaneY];
      src_buffer_base += column_step[kPlaneY];
      cdef_src_base += column_step[kPlaneY] / sizeof(Pixel);

      bp0 += kStep4x4;
      bp1 += kStep4x4;
      column4x4 += kStep4x4;
      y_index++;
    } while (column4x4 < column4x4_start + block_width4x4);

    cdef_buffer_row_base[kPlaneY] += cdef_buffer_row_base_stride[kPlaneY];
    src_buffer_row_base[kPlaneY] += src_buffer_row_base_stride[kPlaneY];
    cdef_src_row_base[kPlaneY] += cdef_src_row_base_stride[kPlaneY];
    bp_row0_base += bp_stride;
    bp_row1_base += bp_stride;
    row4x4 += kStep4x4;
  } while (row4x4 < row4x4_start + block_height4x4);

  if (planes_ == kMaxPlanesMonochrome) {
    return;
  }

  const uint8_t uv_primary_strength =
      frame_header_.cdef.uv_primary_strength[index];
  const uint8_t uv_secondary_strength =
      frame_header_.cdef.uv_secondary_strength[index];

  if ((uv_primary_strength | uv_secondary_strength) == 0) {
    for (int plane = kPlaneU; plane <= kPlaneV; ++plane) {
      CopyBlockForCdefHelper(
          static_cast<Plane>(plane), src_buffer_row_base[plane],
          cdef_buffer_row_base[plane], block_width4x4, block_height4x4,
          row4x4_start, column4x4_start, is_frame_bottom);
    }
    return;
  }

  PrepareCdefBlock<Pixel>(block_width4x4, block_height4x4, row4x4_start,
                          column4x4_start, cdef_block, kCdefUnitSizeWithBorders,
                          false);

  // uv_strength_index is 0 for both primary and secondary strengths being
  // non-zero, 1 for primary only, 2 for secondary only.
  const int uv_strength_index =
      (static_cast<int>(uv_primary_strength == 0) << 1) |
      static_cast<int>(uv_secondary_strength == 0);
  for (int plane = kPlaneU; plane <= kPlaneV; ++plane) {
    const int8_t subsampling_x = subsampling_x_[plane];
    const int8_t subsampling_y = subsampling_y_[plane];
    const int block_width = kStep >> subsampling_x;
    const int block_height = kStep >> subsampling_y;
    int row4x4 = row4x4_start;

    y_index = 0;
    do {
      uint8_t* cdef_buffer_base = cdef_buffer_row_base[plane];
      const uint8_t* src_buffer_base = src_buffer_row_base[plane];
      const uint16_t* cdef_src_base = cdef_src_row_base[plane];
      int column4x4 = column4x4_start;
      do {
        const int cdef_stride = frame_buffer_.stride(plane);
        uint8_t* const cdef_buffer = cdef_buffer_base;
        const int src_stride = frame_buffer_.stride(plane);
        const uint8_t* const src_buffer = src_buffer_base;
        const uint16_t* const cdef_src = cdef_src_base;
        const bool skip = (direction_y[y_index] & kCdefSkip) != 0;
        int dual_cdef = 0;

        if (skip) {  // No cdef filtering.
          CopyBlockForCdefHelper(src_buffer, src_stride, cdef_buffer,
                                 cdef_stride, cdef_src, block_width,
                                 block_height);
        } else {
          // Make sure block pair is not out of bounds.
          if (column4x4 + (kStep4x4 * 2) <= column4x4_start + block_width4x4) {
            // Enable dual processing if subsampling_x is 1.
            dual_cdef = subsampling_x;
          }

          int direction = (uv_primary_strength == 0)
                              ? 0
                              : kCdefUvDirection[subsampling_x][subsampling_y]
                                                [direction_y[y_index]];

          if (dual_cdef != 0) {
            if (uv_primary_strength &&
                direction_y[y_index] != direction_y[y_index + 1]) {
              // Disable dual processing if the second block of the pair does
              // not have the same direction.
              dual_cdef = 0;
            }

            // Disable dual processing if the second block of the pair is a
            // skip.
            if (direction_y[y_index + 1] == kCdefSkip) {
              dual_cdef = 0;
            }
          }

          // Block width is 8 if either dual_cdef is true or subsampling_x == 0.
          const int width_index = dual_cdef | (subsampling_x ^ 1);
          dsp_.cdef_filters[width_index][uv_strength_index](
              cdef_src, kCdefUnitSizeWithBorders, block_height,
              uv_primary_strength, uv_secondary_strength,
              frame_header_.cdef.damping - 1, direction, cdef_buffer,
              cdef_stride);
        }
        // When dual_cdef is set, the above cdef_filter() will process 2 blocks,
        // so adjust the pointers and indexes for 2 blocks.
        cdef_buffer_base += column_step[plane] << dual_cdef;
        src_buffer_base += column_step[plane] << dual_cdef;
        cdef_src_base += (column_step[plane] / sizeof(Pixel)) << dual_cdef;
        column4x4 += kStep4x4 << dual_cdef;
        y_index += 1 << dual_cdef;
      } while (column4x4 < column4x4_start + block_width4x4);

      cdef_buffer_row_base[plane] += cdef_buffer_row_base_stride[plane];
      src_buffer_row_base[plane] += src_buffer_row_base_stride[plane];
      cdef_src_row_base[plane] += cdef_src_row_base_stride[plane];
      row4x4 += kStep4x4;
    } while (row4x4 < row4x4_start + block_height4x4);
  }
}

void PostFilter::ApplyCdefForOneSuperBlockRowHelper(uint16_t* cdef_block,
                                                    int row4x4,
                                                    int block_height4x4) {
  for (int column4x4 = 0; column4x4 < frame_header_.columns4x4;
       column4x4 += kStep64x64) {
    const int index = cdef_index_[DivideBy16(row4x4)][DivideBy16(column4x4)];
    const int block_width4x4 =
        std::min(kStep64x64, frame_header_.columns4x4 - column4x4);

#if LIBGAV1_MAX_BITDEPTH >= 10
    if (bitdepth_ >= 10) {
      ApplyCdefForOneUnit<uint16_t>(cdef_block, index, block_width4x4,
                                    block_height4x4, row4x4, column4x4);
      continue;
    }
#endif  // LIBGAV1_MAX_BITDEPTH >= 10
    ApplyCdefForOneUnit<uint8_t>(cdef_block, index, block_width4x4,
                                 block_height4x4, row4x4, column4x4);
  }
}

void PostFilter::ApplyCdefForOneSuperBlockRow(int row4x4_start, int sb4x4,
                                              bool is_last_row) {
  assert(row4x4_start >= 0);
  assert(DoCdef());
  for (int y = 0; y < sb4x4; y += kStep64x64) {
    const int row4x4 = row4x4_start + y;
    if (row4x4 >= frame_header_.rows4x4) return;

    // Apply cdef for the last 8 rows of the previous superblock row.
    // One exception: If the superblock size is 128x128 and is_last_row is true,
    // then we simply apply cdef for the entire superblock row without any lag.
    // In that case, apply cdef for the previous superblock row only during the
    // first iteration (y == 0).
    if (row4x4 > 0 && (!is_last_row || y == 0)) {
      assert(row4x4 >= 16);
      ApplyCdefForOneSuperBlockRowHelper(cdef_block_, row4x4 - 2, 2);
    }

    // Apply cdef for the current superblock row. If this is the last superblock
    // row we apply cdef for all the rows, otherwise we leave out the last 8
    // rows.
    const int block_height4x4 =
        std::min(kStep64x64, frame_header_.rows4x4 - row4x4);
    const int height4x4 = block_height4x4 - (is_last_row ? 0 : 2);
    if (height4x4 > 0) {
      ApplyCdefForOneSuperBlockRowHelper(cdef_block_, row4x4, height4x4);
    }
  }
}

void PostFilter::ApplyCdefWorker(std::atomic<int>* row4x4_atomic) {
  int row4x4;
  uint16_t cdef_block[kCdefUnitSizeWithBorders * kCdefUnitSizeWithBorders * 2];
  while ((row4x4 = row4x4_atomic->fetch_add(
              kStep64x64, std::memory_order_relaxed)) < frame_header_.rows4x4) {
    const int block_height4x4 =
        std::min(kStep64x64, frame_header_.rows4x4 - row4x4);
    ApplyCdefForOneSuperBlockRowHelper(cdef_block, row4x4, block_height4x4);
  }
}

void PostFilter::ApplyCdefThreaded() {
  std::atomic<int> row4x4(0);
  const int num_workers = thread_pool_->num_threads();
  BlockingCounter pending_workers(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    thread_pool_->Schedule([this, &row4x4, &pending_workers]() {
      ApplyCdefWorker(&row4x4);
      pending_workers.Decrement();
    });
  }
  // Have the current thread partake in applying CDEF.
  ApplyCdefWorker(&row4x4);
  // Wait for the threadpool jobs to finish.
  pending_workers.Wait();
}

}  // namespace libgav1
