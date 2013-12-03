#include "misc.h"

#include <stdio.h>
#include <stdlib.h> 
#include <assert.h>

 /**************************************************************************************************
**************************************************************************************************/
int getRandom( int low, int hi)
{
	assert( hi >= low);
	return( rand() % (hi-low+1) + low);
}

/**************************************************************************************************
**************************************************************************************************/
void finish( const char* msg, int res)
{
	if( msg) printf("%s\n", msg);
	exit( res);
}

/**************************************************************************************************
**************************************************************************************************/
dev_ptr s_cudaMalloc( size_t size)
{
	void* ptr = 0;
	gpuErrchk( "cudaMalloc()", cudaMalloc( &ptr, size));
	return ptr;
}

/**************************************************************************************************
**************************************************************************************************/
dev_ptr s_cudaMalloc( dev_ptr* ptr, size_t size)
{
	gpuErrchk( "cudaMalloc()", cudaMalloc( ptr, size));
	return *ptr;
}

/**************************************************************************************************
**************************************************************************************************/
int s_cudaFree( dev_ptr devPtr)
{
	gpuErrchk( "cudaFree()", cudaFree( devPtr));
	return 0;
}

/**************************************************************************************************
**************************************************************************************************/
void gpuAssert(const char* operation, cudaError_t err, const char *file, int line)
{
	if( err != cudaSuccess)
	{
		printf("%s %d %s failed: [%d] %s\n", file, line, operation, err, cudaGetErrorString( err));
		exit( -1);
	}
}

/**************************************************************************************************
* safely copies <count> bytes from <src> onto <dst>
* returns <dst>
**************************************************************************************************/
void* s_cudaMemcpy( void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	gpuErrchk( "cudaMemcpy()", cudaMemcpy( dst, src, count, kind));
	return dst;
}

/**************************************************************************************************
* copies <size> bytes from <hostPtr> onto <devPtr>
* returns <devPtr>
**************************************************************************************************/
dev_ptr s_hostToDevice( dev_ptr devPtr, const host_ptr hostPtr, size_t size)
{
	s_cudaMemcpy( devPtr, hostPtr, size, cudaMemcpyHostToDevice);
	return devPtr;
}

/**************************************************************************************************
* copies <size> bytes from <devPtr> onto <hostPtr>
* returns <hostPtr>
**************************************************************************************************/
host_ptr s_deviceToHost( host_ptr hostPtr, const dev_ptr devPtr, size_t size)
{
	s_cudaMemcpy( hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
	return hostPtr;
}

/**************************************************************************************************
* allocates <size> bytes in the device and copies the content in <hostPtr> onto it
* returned pointer must be deallocated using cudaFree or s_cudaFree
**************************************************************************************************/
dev_ptr s_allocToDevice( const host_ptr hostPtr, size_t size)
{
	dev_ptr ret = 0;
	
	if( size > 0)
	{
		ret = s_cudaMalloc( size);
		s_hostToDevice( ret, hostPtr, size);
	}

	return ret;
}

/**************************************************************************************************
* allocates <size> bytes in the host and copies the content in <devPtr> onto it
* returned pointer must be deallocated using free
**************************************************************************************************/
host_ptr s_allocToHost( const dev_ptr devPtr, size_t size)
{
	host_ptr ret = 0;

	if( size > 0)
	{
		ret = malloc( size);
		s_deviceToHost( ret, devPtr, size);
	}
	
	return ret;
}

/**************************************************************************************************
**************************************************************************************************/
__device__ void reduce_max(int* data, int count, int* res)
{
    int tid = threadIdx.x;
    bool odd = count % 2;
    for( int s=count/2; s>0; s>>=1)
    {
        if( tid < s)
        {
            data[tid] = _max(data[tid],data[tid+s]);

            if( tid == s-1 && odd)
                data[tid] = _max(data[tid],data[tid+s+1]);
        }

        odd = s % 2;

        __syncthreads();
    }

    if( !tid) res[blockIdx.x] = data[0];
}





