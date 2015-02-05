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
__global__ void MatrixMulKernel(Matrix Md, Matrix Nd, Matrix Pd)
{
    const int TILE_WIDTH = 32;

    __shared__ float Mshared[TILE_WIDTH][TILE_WIDTH + 1];   // Tile size of 32x32 
    __shared__ float Nshared[TILE_WIDTH][TILE_WIDTH + 1];

    int Row = TILE_WIDTH*blockIdx.y + threadIdx.y;
    int Col = TILE_WIDTH*blockIdx.x + threadIdx.x;
    float Pvalue = 0.0;
    //Mshared[threadIdx.y][threadIdx.x] = 0.0;
    //Nshared[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (Md.width - 1)/TILE_WIDTH + 1; ++k)
    {
        if ((Row < Md.height) && (threadIdx.x + (k*TILE_WIDTH)) < Md.width)
        {
            Mshared[threadIdx.y][threadIdx.x] = Md.elements[(Row*Md.width) + threadIdx.x + (k*TILE_WIDTH)];
        }
        else
        {
            Mshared[threadIdx.y][threadIdx.x] = 0.0;
        }            
        if ( Col < Nd.width && (threadIdx.y + k*TILE_WIDTH) < Nd.height)
       {
            Nshared[threadIdx.y][threadIdx.x] = Nd.elements[(threadIdx.y + k*TILE_WIDTH)*Nd.width + Col];
        }
        else
        {
            Nshared[threadIdx.y][threadIdx.x] = 0.0;
        }            
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            Pvalue += Mshared[threadIdx.y][j] * Nshared[j][threadIdx.x];
	     __syncthreads();
        }
    }
    if (Row < Pd.height && Col < Pd.width)
    {
        Pd.elements[Row*Pd.width + Col] = (float)Pvalue;
    }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
