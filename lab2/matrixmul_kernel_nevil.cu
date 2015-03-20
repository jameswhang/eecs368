/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  
  int tile_width = 32;
  __shared__ float Mds[32][32];
  __shared__ float Nds[32][32];

  // Identify the row and column of Pd element to compute
  int Row = blockIdx.y * tile_width + threadIdx.y;
  int Col = blockIdx.x * tile_width + threadIdx.x;
  // int Width = P.width;

  int MRows = M.height;
  int MCols = M.width;
  int NRows = N.height;
  int NCols = N.width;
  int PRows = P.height;
  int PCols = P.width;

  __syncthreads();

  float pValue = 0;
  // Loop over Md and Nd tiles to compute Pd element.
   
  for (int m = 0; m < (tile_width + MCols - 1)/tile_width; m++) {

    if (m * tile_width + threadIdx.x < MCols && Row < MRows) {
	Mds[threadIdx.y][threadIdx.x] = M.elements[Row*MCols + m * tile_width + threadIdx.x];
    } else {
	Mds[threadIdx.y][threadIdx.x] = 0.0;
    }
    
    if (m * tile_width + threadIdx.y < NRows && Col < NCols) {
    	Nds[threadIdx.y][threadIdx.x] = N.elements[(m * tile_width + threadIdx.y) * NCols + Col];
    } else {
	Nds[threadIdx.y][threadIdx.x] = 0.0;
    }

    // Synchronize
    __syncthreads();
    for (int k = 0; k < tile_width; ++k) {
      pValue += Mds[threadIdx.y][k] * Nds[k][threadIdx.y];
    }
    __syncthreads();
  }

  if (Row < PRows && Col < PCols) {
	P.elements[((blockIdx.y * blockDim.y + threadIdx.y) * PCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = pValue;
  }
  //P.elements[Row*Width+Col] = pValue;
  
  /* 
  for (int k = 0; k < Width; ++k) {
    pValue += M.elements[Row*Width+k] * N.elements[k*Width+Col];
  }
  P.elements[Row*Width + Col] = pValue; 
  */
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
