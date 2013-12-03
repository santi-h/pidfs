#include "kernel.h"
#include "common_def.h"
#include "State.h"
#include "ida.h"
#include "compute_cutoff.h"
#include <climits>

__global__ void kernel( global_s* in)
{
    extern __shared__ shared_block_s s[];

    // PREPARE SHARED MEMORY
    int tid = threadIdx.x;
    global_s& global = *in;
    shared_block_s& shared = *s;
    shared_thread_s& thread = shared.threads[tid];

    if( !tid)
    {
        shared.solution_lock = 0;
        shared.goal = global.goal;
        shared.cutoff = global.initial_cutoff;
    }

    int iteration = 0;

    __syncthreads();

    while( !shared.solution_lock)
    {
        thread.next_cutoff = INT_MAX;
        thread.stack = global.stacks[tid];
        thread.timestamp = shared.timestamp;
        
        if( !tid) printf("calling with cutoff %d, timestamp %d\n", shared.cutoff, shared.timestamp);

        ida( shared, shared.cutoff, global.solution, iteration);

        __syncthreads();

        if( !shared.solution_lock)
            compute_cutoff( shared.threads, blockDim.x, &shared.cutoff, &shared.timestamp);

        iteration++;
    }
    __syncthreads();
    compute_cutoff( shared.threads, blockDim.x, &shared.cutoff, &shared.timestamp);
    if( !tid) global.timestamps = shared.timestamp;
}
