// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "aliceVision/depthMap/cuda/deviceCommon/device_patch_es_glob.h"
#include "aliceVision/depthMap/cuda/deviceCommon/device_simStat.h"

namespace aliceVision {
namespace depthMap {

__device__ void computeRotCSEpip(patch& ptch, const float3& p);

__device__ int angleBetwUnitV1andUnitV2(float3& V1, float3& V2);

__device__ bool checkPatch(patch& ptch, int angThr);

__device__ void computeHomography(float* _H, float3& _p, float3& _n);

__device__ float compNCC(float2& rpix, float2& tpix, int wsh);

__device__ float compNCCvarThr(float2& rpix, float2& tpix, int wsh, float varThr);

__device__ simStat compSimStat(float2& rpix, float2& tpix, int wsh);

__device__ float compNCCbyH(patch& ptch, int wsh);

__device__ float compNCCby3DptsYK(patch& ptch, int wsh, int width, int height, const float _gammaC, const float _gammaP,
                                  const float epipShift);

__device__ float compNCCby3Dpts(patch& ptch, int wsh, int width, int height);

__device__ float compNCCby3DptsEpipOpt(patch& ptch, int width, int height);

__device__ void getPixelFor3DPointRC(float2& out, float3& X);

__device__ void getPixelFor3DPointTC(float2& out, float3& X);

__device__ float frontoParellePlaneRCDepthFor3DPoint(const float3& p);

__device__ float frontoParellePlaneTCDepthFor3DPoint(const float3& p);

__device__ float3 get3DPointForPixelAndFrontoParellePlaneRC(float2& pix, float fpPlaneDepth);

__device__ float3 get3DPointForPixelAndFrontoParellePlaneRC(int2& pixi, float fpPlaneDepth);

__device__ float3 get3DPointForPixelAndDepthFromRC(const float2& pix, float depth);

__device__ float3 get3DPointForPixelAndDepthFromTC(const float2& pix, float depth);

__device__ float3 get3DPointForPixelAndDepthFromRC(const int2& pixi, float depth);

__device__ float3 triangulateMatchRef(float2& refpix, float2& tarpix);

__device__ float computeRcPixSize(const float3& p);

__device__ float computePixSize(const float3& p);

__device__ float computeTcPixSize(const float3& p);

__device__ float refineDepthSubPixel(const float3& depths, const float3& sims);

} // namespace depthMap
} // namespace aliceVision

