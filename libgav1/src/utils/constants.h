#ifndef LIBGAV1_SRC_UTILS_CONSTANTS_H_
#define LIBGAV1_SRC_UTILS_CONSTANTS_H_

#include <cstdint>

namespace libgav1 {

// Returns the number of elements between begin (inclusive) and end (inclusive).
constexpr int EnumRangeLength(int begin, int end) { return end - begin + 1; }

enum {
  kCdfMaxProbability = 32768,
  kBlockWidthCount = 5,
  kMaxSegments = 8,
  kMinQuantizer = 0,
  kMinLossyQuantizer = 1,
  kMaxQuantizer = 255,
  kFrameLfCount = 4,
  kMaxLoopFilterValue = 63,
  kNum4x4In64x64 = 256,
  kNumTransformSizesLoopFilter = 3,  // 0: 4x4, 1: 8x8, 2: 16x16.
  kNumLoopFilterMasks = 4,
  kMaxAngleDelta = 3,
  kDirectionalIntraModes = 8,
  kMaxSuperBlockSizeLog2 = 7,
  kMinSuperBlockSizeLog2 = 6,
  kGlobalMotionReadControl = 3,
  kSuperResScaleNumerator = 8,
  kBooleanSymbolCount = 2,
  kRestorationTypeSymbolCount = 3,
  kSgrProjParamsBits = 4,
  kSgrProjPrecisionBits = 7,
  kRestorationBorder = 3,  // Horizontal and vertical border are both 3.
  kConvolveBorderLeftTop = 3,
  kConvolveBorderRightBottom = 4,
  kSubPixelTaps = 8,
  kWienerFilterBits = 7,
  kMaxPaletteSize = 8,
  kMinPaletteSize = 2,
  kMaxPaletteSquare = 64,
  kBorderPixels = 64,
  kWarpedModelPrecisionBits = 16,
  kMaxRefMvStackSize = 8,
  kExtraWeightForNearestMvs = 640,
  kMaxLeastSquaresSamples = 8,
  kMaxSuperBlockSizeInPixels = 128,
  kNum4x4InLoopFilterMaskUnit = 16,
  kRestorationUnitOffset = 8,
  // 2 pixel padding for 5x5 box sum on each side.
  kRestorationPadding = 4,
  // Loop restoration's processing unit size is fixed as 64x64.
  kRestorationProcessingUnitSize = 64,
  kRestorationProcessingUnitSizeWithBorders =
      kRestorationProcessingUnitSize + 2 * kRestorationBorder,
  // The max size of a box filter process output buffer.
  kMaxBoxFilterProcessOutputPixels = kRestorationProcessingUnitSize *
                                     kRestorationProcessingUnitSize,  // == 4096
  // The max size of a box filter process intermediate buffer.
  kBoxFilterProcessIntermediatePixels =
      (kRestorationProcessingUnitSizeWithBorders + kRestorationPadding) *
      (kRestorationProcessingUnitSizeWithBorders +
       kRestorationPadding),  // == 5476
  kSuperResFilterBits = 6,
  kSuperResFilterShifts = 1 << kSuperResFilterBits,
  kSuperResFilterTaps = 8,
  kSuperResScaleBits = 14,
  kSuperResExtraBits = kSuperResScaleBits - kSuperResFilterBits,
  kSuperResScaleMask = (1 << 14) - 1,
  // TODO(chengchen): consider merging these constants:
  // kFilterbits, kWienerFilterBits, and kSgrProjPrecisionBits, which are all 7,
  // They are designed to match AV1 convolution, which increases coeff
  // values up to 7 bits. We could consider to combine them and use kFilterBits
  // only.
  kFilterBits = 7,
  // Sub pixel is used in AV1 to represent a pixel location that is not at
  // integer position. Sub pixel is in 1/16 (1 << kSubPixelBits) unit of
  // integer pixel. Sub pixel values are interpolated using adjacent integer
  // pixel values. The interpolation is a filtering process.
  kSubPixelBits = 4,
  // Precision bits when computing inter prediction locations.
  kScaleSubPixelBits = 10,
  kWarpParamRoundingBits = 6,
  // Number of fractional bits of lookup in divisor lookup table.
  kDivisorLookupBits = 8,
  // Number of fractional bits of entries in divisor lookup table.
  kDivisorLookupPrecisionBits = 14,
  // Number of phases used in warped filtering.
  kWarpedPixelPrecisionShifts = 1 << 6,
  kQuantizedCoefficientBufferPadding = 4,
  // Maximum number of quantized coefficients that can be read from the
  // bitstream. This comes from the definition of segEob in section 5.11.39.
  // Size of the quantized coefficients buffer. This comes from the definition
  // of segEob in section 5.11.39 (with 4 bytes padded to each row and 4 rows
  // padded in the end to avoid boundary checks).
  kQuantizedCoefficientBufferSize = (32 + kQuantizedCoefficientBufferPadding) *
                                    (32 + kQuantizedCoefficientBufferPadding),
  kWedgeMaskMasterSize = 64,
  kMaxMaskBlockSize = kWedgeMaskMasterSize * kWedgeMaskMasterSize,
  kWedgeMaskSize = 9 * 2 * 16 * kWedgeMaskMasterSize * kWedgeMaskMasterSize,
  kMaxFrameDistance = 31,
  kReferenceFrameScalePrecision = 14,
  kNumWienerCoefficients = 3,
};  // anonymous enum

enum FrameType : uint8_t {
  kFrameKey,
  kFrameInter,
  kFrameIntraOnly,
  kFrameSwitch
};

enum Plane : uint8_t { kPlaneY, kPlaneU, kPlaneV };
enum : uint8_t { kMaxPlanesMonochrome = kPlaneY + 1, kMaxPlanes = kPlaneV + 1 };

// The plane types, called luma and chroma in the spec.
enum PlaneType : uint8_t { kPlaneTypeY, kPlaneTypeUV, kNumPlaneTypes };

enum ReferenceFrameType : int8_t {
  kReferenceFrameNone = -1,
  kReferenceFrameIntra,
  kReferenceFrameLast,
  kReferenceFrameLast2,
  kReferenceFrameLast3,
  kReferenceFrameGolden,
  kReferenceFrameBackward,
  kReferenceFrameAlternate2,
  kReferenceFrameAlternate,
  kNumReferenceFrameTypes,
  kNumInterReferenceFrameTypes =
      EnumRangeLength(kReferenceFrameLast, kReferenceFrameAlternate),
  kNumForwardReferenceTypes =
      EnumRangeLength(kReferenceFrameLast, kReferenceFrameGolden),
  kNumBackwardReferenceTypes =
      EnumRangeLength(kReferenceFrameBackward, kReferenceFrameAlternate)
};

enum {
  // Unidirectional compound reference pairs that are signaled explicitly:
  // {kReferenceFrameLast, kReferenceFrameLast2},
  // {kReferenceFrameLast, kReferenceFrameLast3},
  // {kReferenceFrameLast, kReferenceFrameGolden},
  // {kReferenceFrameBackward, kReferenceFrameAlternate}
  kExplicitUnidirectionalCompoundReferences = 4,
  // Other unidirectional compound reference pairs:
  // {kReferenceFrameLast2, kReferenceFrameLast3},
  // {kReferenceFrameLast2, kReferenceFrameGolden},
  // {kReferenceFrameLast3, kReferenceFrameGolden},
  // {kReferenceFrameBackward, kReferenceFrameAlternate2},
  // {kReferenceFrameAlternate2, kReferenceFrameAlternate}
  kUnidirectionalCompoundReferences =
      kExplicitUnidirectionalCompoundReferences + 5,
};  // anonymous enum

enum BlockSize : uint8_t {
  kBlock4x4,
  kBlock4x8,
  kBlock4x16,
  kBlock8x4,
  kBlock8x8,
  kBlock8x16,
  kBlock8x32,
  kBlock16x4,
  kBlock16x8,
  kBlock16x16,
  kBlock16x32,
  kBlock16x64,
  kBlock32x8,
  kBlock32x16,
  kBlock32x32,
  kBlock32x64,
  kBlock64x16,
  kBlock64x32,
  kBlock64x64,
  kBlock64x128,
  kBlock128x64,
  kBlock128x128,
  kMaxBlockSizes,
  kBlockInvalid
};

enum Partition : uint8_t {
  kPartitionNone,
  kPartitionHorizontal,
  kPartitionVertical,
  kPartitionSplit,
  kPartitionHorizontalWithTopSplit,
  kPartitionHorizontalWithBottomSplit,
  kPartitionVerticalWithLeftSplit,
  kPartitionVerticalWithRightSplit,
  kPartitionHorizontal4,
  kPartitionVertical4
};
enum : uint8_t { kMaxPartitionTypes = kPartitionVertical4 + 1 };

enum PredictionMode : uint8_t {
  // Intra prediction modes.
  kPredictionModeDc,
  kPredictionModeVertical,
  kPredictionModeHorizontal,
  kPredictionModeD45,
  kPredictionModeD135,
  kPredictionModeD113,
  kPredictionModeD157,
  kPredictionModeD203,
  kPredictionModeD67,
  kPredictionModeSmooth,
  kPredictionModeSmoothVertical,
  kPredictionModeSmoothHorizontal,
  kPredictionModePaeth,
  kPredictionModeChromaFromLuma,
  // Single inter prediction modes.
  kPredictionModeNearestMv,
  kPredictionModeNearMv,
  kPredictionModeGlobalMv,
  kPredictionModeNewMv,
  // Compound inter prediction modes.
  kPredictionModeNearestNearestMv,
  kPredictionModeNearNearMv,
  kPredictionModeNearestNewMv,
  kPredictionModeNewNearestMv,
  kPredictionModeNearNewMv,
  kPredictionModeNewNearMv,
  kPredictionModeGlobalGlobalMv,
  kPredictionModeNewNewMv,
  kNumPredictionModes,
  kNumCompoundInterPredictionModes =
      EnumRangeLength(kPredictionModeNearestNearestMv, kPredictionModeNewNewMv),
  kIntraPredictionModesY =
      EnumRangeLength(kPredictionModeDc, kPredictionModePaeth),
  kIntraPredictionModesUV =
      EnumRangeLength(kPredictionModeDc, kPredictionModeChromaFromLuma),
  kPredictionModeInvalid = 255
};

enum InterIntraMode : uint8_t {
  kInterIntraModeDc,
  kInterIntraModeVertical,
  kInterIntraModeHorizontal,
  kInterIntraModeSmooth,
  kNumInterIntraModes
};

enum MotionMode : uint8_t {
  kMotionModeSimple,
  kMotionModeObmc,  // Overlapped block motion compensation.
  kMotionModeLocalWarp,
  kNumMotionModes
};

enum TxMode : uint8_t {
  kTxModeOnly4x4,
  kTxModeLargest,
  kTxModeSelect,
  kNumTxModes
};

// These enums are named as kType1Type2 where Type1 is the transform type for
// the rows and Type2 is the transform type for the columns.
enum TransformType : uint8_t {
  kTransformTypeDctDct,
  kTransformTypeAdstDct,
  kTransformTypeDctAdst,
  kTransformTypeAdstAdst,
  kTransformTypeFlipadstDct,
  kTransformTypeDctFlipadst,
  kTransformTypeFlipadstFlipadst,
  kTransformTypeAdstFlipadst,
  kTransformTypeFlipadstAdst,
  kTransformTypeIdentityIdentity,
  kTransformTypeIdentityDct,
  kTransformTypeDctIdentity,
  kTransformTypeIdentityAdst,
  kTransformTypeAdstIdentity,
  kTransformTypeIdentityFlipadst,
  kTransformTypeFlipadstIdentity,
  kNumTransformTypes
};

// Allows checking whether a transform requires rows or columns to be flipped
// with a single comparison rather than a chain of ||s. This should result in
// fewer instructions overall.
enum : uint32_t {
  kTransformFlipColumnsMask = (1U << kTransformTypeFlipadstDct) |
                              (1U << kTransformTypeFlipadstAdst) |
                              (1U << kTransformTypeFlipadstIdentity) |
                              (1U << kTransformTypeFlipadstFlipadst),
  kTransformFlipRowsMask = (1U << kTransformTypeDctFlipadst) |
                           (1U << kTransformTypeAdstFlipadst) |
                           (1U << kTransformTypeIdentityFlipadst) |
                           (1U << kTransformTypeFlipadstFlipadst)
};

enum TransformSize : uint8_t {
  kTransformSize4x4,
  kTransformSize4x8,
  kTransformSize4x16,
  kTransformSize8x4,
  kTransformSize8x8,
  kTransformSize8x16,
  kTransformSize8x32,
  kTransformSize16x4,
  kTransformSize16x8,
  kTransformSize16x16,
  kTransformSize16x32,
  kTransformSize16x64,
  kTransformSize32x8,
  kTransformSize32x16,
  kTransformSize32x32,
  kTransformSize32x64,
  kTransformSize64x16,
  kTransformSize64x32,
  kTransformSize64x64,
  kNumTransformSizes
};

enum : uint32_t {
  // Mask of all transform sizes with either dimension equal to 64.
  kTransformSize64Mask =
      (1U << kTransformSize64x16) | (1U << kTransformSize64x32) |
      (1U << kTransformSize64x64) | (1U << kTransformSize16x64) |
      (1U << kTransformSize32x64),
  // Mask of all transform sizes with width equal to 16.
  kTransformWidth16Mask =
      (1U << kTransformSize16x4) | (1U << kTransformSize16x8) |
      (1U << kTransformSize16x16) | (1U << kTransformSize16x32) |
      (1U << kTransformSize16x64),
  // Mask of all transform sizes with height equal to 16.
  kTransformHeight16Mask =
      (1U << kTransformSize4x16) | (1U << kTransformSize8x16) |
      (1U << kTransformSize16x16) | (1U << kTransformSize32x16) |
      (1U << kTransformSize64x16)
};

enum TransformSet : uint8_t {
  // DCT Only (1).
  kTransformSetDctOnly,
  // 2D-DCT and 2D-ADST without flip (4) + Identity (1) + 1D Horizontal/Vertical
  // DCT (2) = Total (7).
  kTransformSetIntra1,
  // 2D-DCT and 2D-ADST without flip (4) + Identity (1) = Total (5).
  kTransformSetIntra2,
  // All transforms = Total (16).
  kTransformSetInter1,
  // 2D-DCT and 2D-ADST with flip (9) + Identity (1) + 1D Horizontal/Vertical
  // DCT (2) = Total (12).
  kTransformSetInter2,
  // DCT (1) + Identity (1) = Total (2).
  kTransformSetInter3,
  kNumTransformSets
};

enum TransformClass : uint8_t {
  kTransformClass2D,
  kTransformClassHorizontal,
  kTransformClassVertical,
};

enum FilterIntraPredictor : uint8_t {
  kFilterIntraPredictorDc,
  kFilterIntraPredictorVertical,
  kFilterIntraPredictorHorizontal,
  kFilterIntraPredictorD157,
  kFilterIntraPredictorPaeth,
  kNumFilterIntraPredictors
};

// In AV1 the name of the filter refers to the direction of filter application.
// Horizontal refers to the column edge and vertical the row edge.
enum LoopFilterType : uint8_t {
  kLoopFilterTypeVertical,
  kLoopFilterTypeHorizontal,
  kNumLoopFilterTypes
};

enum LoopRestorationType : uint8_t {
  kLoopRestorationTypeNone,
  kLoopRestorationTypeSwitchable,
  kLoopRestorationTypeWiener,
  kLoopRestorationTypeSgrProj,  // self guided projection filter.
  kNumLoopRestorationTypes
};

enum CompoundReferenceType : uint8_t {
  kCompoundReferenceUnidirectional,
  kCompoundReferenceBidirectional,
  kNumCompoundReferenceTypes
};

enum CompoundPredictionType : uint8_t {
  kCompoundPredictionTypeWedge,
  kCompoundPredictionTypeDiffWeighted,
  kCompoundPredictionTypeAverage,
  kCompoundPredictionTypeIntra,
  kCompoundPredictionTypeDistance,
  kNumCompoundPredictionTypes,
  // Number of compound prediction types that are explicitly signaled in the
  // bitstream (in the compound_type syntax element).
  kNumExplicitCompoundPredictionTypes = 2
};

enum InterpolationFilter : uint8_t {
  kInterpolationFilterEightTap,
  kInterpolationFilterEightTapSmooth,
  kInterpolationFilterEightTapSharp,
  kInterpolationFilterBilinear,
  kInterpolationFilterSwitchable,
  kNumInterpolationFilters,
  // Number of interpolation filters that can be explicitly signaled in the
  // compressed headers (when the uncompressed headers allow switchable
  // interpolation filters) of the bitstream.
  kNumExplicitInterpolationFilters = EnumRangeLength(
      kInterpolationFilterEightTap, kInterpolationFilterEightTapSharp)
};

enum MvJointType : uint8_t {
  kMvJointTypeZero,
  kMvJointTypeHorizontalNonZeroVerticalZero,
  kMvJointTypeHorizontalZeroVerticalNonZero,
  kMvJointTypeNonZero,
  kNumMvJointTypes
};

enum ObuType : int8_t {
  kObuInvalid = -1,
  kObuSequenceHeader = 1,
  kObuTemporalDelimiter = 2,
  kObuFrameHeader = 3,
  kObuTileGroup = 4,
  kObuMetadata = 5,
  kObuFrame = 6,
  kObuRedundantFrameHeader = 7,
  kObuTileList = 8,
  kObuPadding = 15,
};

extern const uint8_t k4x4WidthLog2[kMaxBlockSizes];

extern const uint8_t k4x4HeightLog2[kMaxBlockSizes];

extern const uint8_t kNum4x4BlocksWide[kMaxBlockSizes];

extern const uint8_t kNum4x4BlocksHigh[kMaxBlockSizes];

extern const uint8_t kBlockWidthPixels[kMaxBlockSizes];

extern const uint8_t kBlockHeightPixels[kMaxBlockSizes];

extern const BlockSize kSubSize[kMaxPartitionTypes][kMaxBlockSizes];

extern const BlockSize kPlaneResidualSize[kMaxBlockSizes][2][2];

extern const uint8_t kTransformWidth[kNumTransformSizes];

extern const uint8_t kTransformHeight[kNumTransformSizes];

extern const uint8_t kTransformWidthLog2[kNumTransformSizes];

extern const uint8_t kTransformHeightLog2[kNumTransformSizes];

extern const TransformSize kMaxTransformSizeRectangle[kMaxBlockSizes];

extern const int kMaxTransformDepth[kMaxBlockSizes];

extern const TransformSize kSplitTransformSize[kNumTransformSizes];

// Square transform of size min(w,h).
extern const TransformSize kTransformSizeSquareMin[kNumTransformSizes];

// Square transform of size max(w,h).
extern const TransformSize kTransformSizeSquareMax[kNumTransformSizes];

extern const TransformType kModeToTransformType[kIntraPredictionModesUV];

extern const uint8_t kNumTransformTypesInSet[kNumTransformSets];

extern const TransformType kInverseTransformTypeBySet[kNumTransformSets - 1]
                                                     [16];

// Replaces all occurrences of 64x* and *x64 with 32x* and *x32 respectively.
extern const TransformSize kAdjustedTransformSize[kNumTransformSizes];

extern const uint8_t kSgrProjParams[1 << kSgrProjParamsBits][4];

extern const int8_t kSgrProjMultiplierMin[2];

extern const int8_t kSgrProjMultiplierMax[2];

extern const int8_t kSgrProjDefaultMultiplier[2];

extern const int8_t kWienerDefaultFilter[3];

extern const int8_t kWienerTapsMin[3];

extern const int8_t kWienerTapsMax[3];

extern const int16_t kUpscaleFilter[kSuperResFilterShifts][kSuperResFilterTaps];

extern const int16_t kWarpedFilters[3 * kWarpedPixelPrecisionShifts + 1][8];

extern const int16_t kSubPixelFilters[6][16][8];

extern const int16_t kDirectionalIntraPredictorDerivative[44];

extern const uint8_t kPredictionModeDeltasLookup[kNumPredictionModes];

}  // namespace libgav1

#endif  // LIBGAV1_SRC_UTILS_CONSTANTS_H_
