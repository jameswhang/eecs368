#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cutil.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

#define SQRT_2    1.4142135623730950488
#define SPREAD_BOTTOM   (2)
#define SPREAD_TOP      (6)

#define NEXT(init_, spread_)\
    (init_ + (int)((drand48() - 0.5) * (drand48() - 0.5) * 4.0 * SQRT_2 * SQRT_2 * spread_));

#define CLAMP(value_, min_, max_)\
    if (value_ < 0)\
        value_ = (min_);\
    else if (value_ > (max_))\
        value_ = (max_);

// Generate another bin for the histogram.  The bins are created as a random walk ...
static uint32_t next_bin(uint32_t pix)
{
    const uint16_t bottom = pix & ((1<<HISTO_LOG)-1);
    const uint16_t top   = (uint16_t)(pix >> HISTO_LOG);

    int new_bottom = NEXT(bottom, SPREAD_BOTTOM)
    CLAMP(new_bottom, 0, HISTO_WIDTH-1)

    int new_top = NEXT(top, SPREAD_TOP)
    CLAMP(new_top, 0, HISTO_HEIGHT-1)

    const uint32_t result = (new_bottom | (new_top << HISTO_LOG)); 

    return result; 
}

// Return a 2D array of histogram bin-ids.  This function generates
// bin-ids with correlation characteristics similar to some actual images.
// The key point here is that the pixels (and thus the bin-ids) are *NOT*
// randomly distributed ... a given pixel tends to be similar to the
// pixels near it.
//
//


static uint32_t **generate_histogram_bins()
{
    uint32_t **input = (uint32_t**)alloc_2d(INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));
    
    input[0][0] = HISTO_WIDTH/2 | ((HISTO_HEIGHT/2) << HISTO_LOG);
    for (int i = 1; i < INPUT_WIDTH; ++i)
        input[0][i] =  next_bin(input[0][i - 1]);
    for (int j = 1; j < INPUT_HEIGHT; ++j)
    {
        input[j][0] =  next_bin(input[j - 1][0]);
        for (int i = 1; i < INPUT_WIDTH; ++i)
            input[j][i] =  next_bin(input[j][i - 1]);
    }	
	
    return input;
}

int main(int argc, char* argv[])
{
    /* Case of 0 arguments: Default seed is used */
    if (argc < 2){
	srand48(0);
    }
    /* Case of 1 argument: Seed is specified as first command line argument */ 
    else {
	int seed = atoi(argv[1]);
	srand48(seed);
    }

    uint8_t *gold_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // Use kernel_bins for your final result
    uint8_t *kernel_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // A 2D array of histogram bin-ids.  One can think of each of these bins-ids as
    // being associated with a pixel in a 2D image.
    
   	uint32_t **input = generate_histogram_bins();

    int PADDED_WIDTH = ((INPUT_WIDTH + 128) & 0xFFFFFF80);
    	
	int startCPUTime = clock();
    TIME_IT("ref_2dhisto", 1000, ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins);)
    int endCPUTime = clock();
    int CPUTime = endCPUTime - startCPUTime;
	
	uint32_t *d_Input = (uint32_t *)allocateDevice(PADDED_WIDTH*INPUT_HEIGHT*sizeof(uint32_t));
	uint32_t *d_bins = (uint32_t *)allocateDevice(HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
	uint8_t *d_byteBins = (uint8_t *)allocateDevice(HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint8_t));
	copyToDevice(d_Input, &(input[0][0]), PADDED_WIDTH*INPUT_HEIGHT*sizeof(uint32_t));
	
	unsigned int hTimer;		
	cutCreateTimer(&hTimer);
	cutStartTimer(hTimer);
	TIME_IT("opt_2dhisto", 1000, opt_2dhisto(d_Input, INPUT_HEIGHT, INPUT_HEIGHT, d_bins, d_byteBins);)
    	cutStopTimer(hTimer);
    	
	float GPUTime = cutGetTimerValue(hTimer);
    	printf("GPU Time using cut timer: %fms\n", GPUTime);
    	printf("x%04f speed up\n", (float)CPUTime/(GPUTime * 1000));	
    	/* Include your teardown code below (temporary variables, function calls, etc.)*/
        copyToHost(kernel_bins, d_byteBins,  HISTO_WIDTH*HISTO_HEIGHT*sizeof(uint8_t));
	freeGPU(d_bins);
	freeGPU(d_byteBins);
	freeGPU(d_Input);


    /* End of teardown code */
    int passed=1;
    for (int i =  0; i < HISTO_HEIGHT*HISTO_WIDTH; i++){
        if (gold_bins[i] != kernel_bins[i]){
            passed = 0;
            break;
        }
    }
    (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");

    free(gold_bins);
    free(kernel_bins);
}
