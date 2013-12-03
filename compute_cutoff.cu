#include "compute_cutoff.h"
#include "misc.h"
#include "common_def.h"

__device__ void compute_cutoff(shared_thread_s* shared, int count, int* cutoff_res, int* timestamp_res)
{
    int tid = threadIdx.x;
    bool odd = count % 2;
    for( int s=count/2; s>0; s>>=1)
    {
        if( tid < s)
        {
            shared[tid].next_cutoff = _min(shared[tid].next_cutoff,shared[tid+s].next_cutoff);
            shared[tid].timestamp = _max(shared[tid].timestamp,shared[tid+s].timestamp);

            if( tid == s-1 && odd)
            {
                shared[tid].next_cutoff = _min(shared[tid].next_cutoff,shared[tid+s+1].next_cutoff);
                shared[tid].timestamp = _max(shared[tid].timestamp,shared[tid+s+1].timestamp);
            }
        }

        odd = s % 2;

        __syncthreads();
    }

    if( !tid)
    {
        cutoff_res[blockIdx.x] = shared[0].next_cutoff;
        timestamp_res[blockIdx.x] = shared[0].timestamp;
        
    }
}
