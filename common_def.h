#ifndef COMMON_DEF_H
#define COMMON_DEF_H

#include "misc.h"
#include "State.h"
#include "CudaArray.h"

struct stack_elem_s;
struct global_s;
struct shared_thread_s;
struct shared_block_s;
typedef CudaArray<stack_elem_s> stack_t;

struct stack_elem_s
{
    action_t action;
    State state;
    CudaArray<action_t> possible;
    cost_t g;

	CUDA_CALLABLE_MEMBER ~stack_elem_s(){}

    CUDA_CALLABLE_MEMBER stack_elem_s(action_t p1, const State& p2, const CudaArray<action_t> p3, cost_t p4) :
        action(p1),
        state(p2),
        possible(p3),
        g(p4)
    {}

	CUDA_CALLABLE_MEMBER stack_elem_s(const stack_elem_s& o) :
		action(o.action),
		state(o.state),
		possible(o.possible),
		g(o.g)
	{}

    CUDA_CALLABLE_MEMBER stack_elem_s() :
        action( no_op()),
        state(),
        possible(),
        g(0)
    {}
};

struct global_s
{
    CudaArray<action_t> solution;
    CudaArray<stack_t> stacks;
    State goal;
    cost_t initial_cutoff;
    int timestamps;
    int pushes;
};

struct shared_thread_s
{
    int timestamp;
    cost_t next_cutoff;
    stack_t stack;
};

struct shared_block_s
{
    lock_t solution_lock;
    bool solution_found;
    State goal;
    cost_t cutoff;
    int timestamp;
    shared_thread_s threads[THREADS_PER_BLOCK];
};

#endif
