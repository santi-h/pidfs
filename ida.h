#ifndef IDA_H
#define IDA_H

#include "common_def.h"
#include "misc.h"
#include "State.h"

class Heuristic;


__device__ void ida( shared_block_s&, cost_t, CudaArray<action_t>&, int, stack_t&);
__device__ bool _explored(const stack_t&, const State&);

#endif IDA_H
