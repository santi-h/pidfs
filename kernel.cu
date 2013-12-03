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
        shared.solution_found = 0;
        
        shared.goal = global.goal;
        shared.cutoff = global.initial_cutoff;
    }

    int iteration = 0;

    __syncthreads();

    thread.next_cutoff = INT_MAX;
    thread.stack = global.stacks[tid];
    thread.timestamp = shared.timestamp;

    ida( shared, shared.cutoff, global.solution, iteration, global.stacks[tid]);
}
