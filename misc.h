#ifndef MISC_H
#define MISC_H

#define gpuErrchk(op,ans) { gpuAssert(op, ans, __FILE__, __LINE__); }
#define THREADS_PER_BLOCK 4
#define THREADS_PER_WARP 32
#define LOG 0

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef int lock_t;
typedef bool flag_t;

typedef void* host_ptr;
typedef void* dev_ptr;

int getRandom( int low, int hi);
void finish( const char*, int);
void gpuAssert(const char*, cudaError_t, const char*, int);
int s_cudaFree( dev_ptr);
void* s_cudaMemcpy( void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
dev_ptr s_cudaMalloc( size_t);
dev_ptr s_cudaMalloc( dev_ptr*, size_t);
dev_ptr s_hostToDevice( dev_ptr, const host_ptr, size_t);
dev_ptr s_allocToDevice( const host_ptr, size_t);
host_ptr s_deviceToHost( host_ptr, const dev_ptr, size_t);
host_ptr s_allocToHost( const dev_ptr, size_t);

inline __host__ __device__ int _max(int a, int b) { return a > b ? a : b; }
inline __host__ __device__ int _min(int a, int b) { return a < b ? a : b; }

__device__ void reduce_max(int* data, int count, int* res);

#endif //MISC_H
