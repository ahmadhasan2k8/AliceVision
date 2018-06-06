// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "aliceVision/depthMap/cuda/planeSweeping/device_utils.cuh"

#include "aliceVision/depthMap/cuda/deviceCommon/device_patch_es_glob.cuh"

#include <math_constants.h>

namespace aliceVision {
namespace depthMap {

// Global data handlers and parameters

#define DCT_DIMENSION 7

#define MAX_CUDA_DEVICES 10

extern texture<int4, 2, cudaReadModeElementType> volPixsTex;

extern texture<int2, 2, cudaReadModeElementType> pixsTex;

extern texture<float2, 2, cudaReadModeElementType> gradTex;

extern texture<float, 2, cudaReadModeElementType> depthsTex;

extern texture<float, 2, cudaReadModeElementType> depthsTex1;

extern texture<float4, 2, cudaReadModeElementType> normalsTex;

extern texture<float, 2, cudaReadModeElementType> sliceTex;

extern texture<float2, 2, cudaReadModeElementType> sliceTexFloat2;

extern texture<unsigned char, 2, cudaReadModeElementType> sliceTexUChar;

extern texture<uint2, 2, cudaReadModeElementType> sliceTexUInt2;

extern texture<unsigned int, 2, cudaReadModeElementType> sliceTexUInt;

extern texture<uchar4, 2, cudaReadModeElementType> rTexU4;

extern texture<uchar4, 2, cudaReadModeElementType> tTexU4;

extern texture<float4, 2, cudaReadModeElementType> f4Tex;

//////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ unsigned char computeSigmaOfL(int x, int y, int r);

__device__ unsigned char computeGradientSizeOfL(int x, int y);

__global__ void compute_varLofLABtoW_kernel(uchar4* labMap, int labMap_p, int width, int height, int wsh);

//////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void move3DPointByRcPixSize(float3& p, float rcPixSize);

__device__ void move3DPointByTcPixStep(float3& p, float tcPixStep);

__device__ float move3DPointByTcOrRcPixStep(float3& p, float pixStep, bool moveByTcOrRc);

__device__ void computePatch(patch& ptch, int depthid, int ndepths, int2& pix, int pixid, int t, bool doUsePixelsDepths,
                             bool useTcOrRcPixSize);

__global__ void slice_kernel(float* slice, int slice_p,
                             // float3* slicePts, int slicePts_p,
                             int ndepths, int slicesAtTime,
                             int width, int height, int wsh, int t, int npixs,
                             int maxDepth,
                             bool doUsePixelsDepths, bool useTcOrRcPixSize, const float gammaC, const float gammaP,
                             const float epipShift);

__device__ float3 computeDepthPoint_fine(float& pixSize, int depthid, int ndepths, int2& pix, int pixid, int t);

__global__ void slice_fine_kernel(float* slice, int slice_p,
                                  int ndepths, int slicesAtTime,
                                  int width, int height, int wsh, int t, int npixs,
                                  int maxDepth, const float gammaC, const float gammaP, const float epipShift);

__global__ void smoothDepthMap_kernel(
    float* dmap, int dmap_p,
    int width, int height, int wsh, const float gammaC, const float gammaP);

__global__ void filterDepthMap_kernel(
    float* dmap, int dmap_p,
    int width, int height, int wsh, const float gammaC, const float minCostThr);

__global__ void alignSourceDepthMapToTarget_kernel(
    float* dmap, int dmap_p,
    int width, int height, int wsh, const float gammaC, const float maxPixelSizeDist);

__global__ void computeNormalMap_kernel(
    float3* nmap, int nmap_p,
    int width, int height, int wsh, const float gammaC, const float gammaP);

__global__ void locmin_kernel(float* slice, int slice_p, int ndepths, int slicesAtTime,
                              int width, int height, int wsh, int t, int npixs,
                              int maxDepth,
                              bool doUsePixelsDepths, int kernelSizeHalf);

__global__ void getBest_kernel(float* slice, int slice_p,
                               // int* bestDptId, int bestDptId_p,
                               float* bestDpt, int bestDpt_p,
                               float* bestSim, int bestSim_p,
                               int slicesAtTime, int ndepths, int t, int npixs,
                               int wsh, int width, int height, bool doUsePixelsDepths, int nbest, bool useTcOrRcPixSize,
                               bool subPixel);

__global__ void getBest_fine_kernel(float* slice, int slice_p,
                                    // int* bestDptId, int bestDptId_p,
                                    float* bestDpt, int bestDpt_p,
                                    float* bestSim, int bestSim_p,
                                    int slicesAtTime, int ndepths, int t, int npixs,
                                    int wsh, int width, int height);

__device__ float2 computeMagRot(float l, float r, float u, float d);

__device__ float2 computeMagRotL(int x, int y);

__global__ void grad_kernel(float2* grad, int grad_p,
                            int2* pixs, int pixs_p,
                            int slicesAtTime, int ntimes,
                            int width, int height, int wsh, int npixs);

__global__ void getRefTexLAB_kernel(uchar4* texs, int texs_p, int width, int height);

__global__ void getTarTexLAB_kernel(uchar4* texs, int texs_p, int width, int height);

__global__ void reprojTarTexLAB_kernel(uchar4* texs, int texs_p, int width, int height, float fpPlaneDepth);

__global__ void reprojTarTexRgb_kernel(uchar4* texs, int texs_p, int width, int height, float fpPlaneDepth);

__global__ void copyUchar4Dim2uchar_kernel(int dim, uchar4* src, int src_p, unsigned char* tar, int tar_p, int width,
                                           int height);

__global__ void transpose_uchar4_kernel(uchar4* input, int input_p, uchar4* output, int output_p, int width, int height);

__global__ void transpose_float4_kernel(float4* input, int input_p, float4* output, int output_p, int width, int height);

__global__ void compAggrNccSim_kernel(
    float4* ostat1, int ostat1_p,
    float4* ostat2, int ostat2_p,
    uchar4* rImIn, int rImIn_p,
    uchar4* tImIn, int tImIn_p,
    int width, int height, int step, int orintation);

__global__ void compNccSimFromStats_kernel(
    float* odepth, int odepth_p,
    float* osim, int osim_p,
    float4* stat1, int stat1_p,
    float4* stat2, int stat2_p,
    int width, int height, int d, float depth);

__global__ void compWshNccSim_kernel(float* osim, int osim_p, int width, int height, int wsh, int step);

__global__ void aggrYKNCCSim_kernel(float* osim, int osim_p, int width, int height, int wsh, int step,
                                    const float gammaC, const float gammaP);

__global__ void updateBestDepth_kernel(
    float* osim, int osim_p,
    float* odpt, int odpt_p,
    float* isim, int isim_p,
    int width, int height, int step, float fpPlaneDepth, int d);

__global__ void downscale_bilateral_smooth_lab_kernel(
    cudaTextureObject_t gaussianTex,
    uchar4* texLab, int texLab_p,
    int width, int height, int scale, int radius, float gammaC);

__global__ void downscale_gauss_smooth_lab_kernel(
    cudaTextureObject_t gaussianTex,
    uchar4* texLab, int texLab_p,
    int width, int height, int scale, int radius);

__global__ void downscale_mean_smooth_lab_kernel(
    uchar4* texLab, int texLab_p,
    int width, int height, int scale);

__global__ void ptsStatForRcDepthMap_kernel(float2* out, int out_p,
                                            float3* pts, int pts_p,
                                            int npts, int width, int height,
                                            int maxNPixSize, int wsh, const float gammaC, const float gammaP);

__global__ void getSilhoueteMap_kernel(bool* out, int out_p, int step, int width, int height, const uchar4 maskColorLab);

__global__ void retexture_kernel(uchar4* out, int out_p, float4* retexturePixs, int retexturePixs_p, int width,
                                 int height, int npixs);

__global__ void retextureComputeNormalMap_kernel(
    uchar4* out, int out_p,
    float2* retexturePixs, int retexturePixs_p,
    float3* retexturePixsNorms, int retexturePixsNorms_p,
    int width, int height, int npixs);

__global__ void pushPull_Push_kernel(uchar4* out, int out_p, int width, int height);

__global__ void pushPull_Pull_kernel(uchar4* out, int out_p, int width, int height);

} // namespace depthMap
} // namespace aliceVision
