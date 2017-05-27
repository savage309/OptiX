#define kernel extern "C" __global__
#define device __device__
#define shared __shared__
#define syncThreads __syncthreads

#if __cplusplus > 199711L
#define HAS_CPP_11
#endif

#define BLOCK_SIZE 32

extern "C"
__global__
void generatePrimaryRay() {
    
}

extern "C"
__global__
void exception() {
    
}
