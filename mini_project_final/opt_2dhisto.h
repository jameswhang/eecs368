#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t*, size_t, size_t, uint32_t*, uint8_t*);
void* allocateDevice(size_t);
void copyToDevice(void*, void*, size_t);
void copyToHost(void*, void*, size_t);
void freeGPU(void*);


/* Include below the function headers of any other functions that you implement */


#endif
