// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

namespace aliceVision {
namespace depthMap {

// patch exhaustive search

struct patch
{
	float3 p; //< 3d point
	float3 n; //< normal
	float3 x; //< 
	float3 y; //< 
    float d;  //< 
};

__device__ void rotPointAroundVect(float3& out, float3& X, float3& vect, int angle);

__device__ void rotatePatch(patch& ptch, int rx, int ry);

__device__ void movePatch(patch& ptch, int pt);

__device__ void computeRotCS(float3& xax, float3& yax, float3& n);

} // namespace depthMap
} // namespace aliceVision
