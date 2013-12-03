#ifndef BFS_H
#define BFS_H

#include "State.h"
#include "CudaArray.h"
#include "common_def.h"

struct bfs_ret_s
{
    CudaArray<stack_t> stacks;
    CudaArray<action_t> solution;
    cost_t next_cutoff;
};

bfs_ret_s bfs( const State&, const State&, int);

#endif
