#ifndef STATE_H
#define STATE_H

#include "CudaArray.h"
#include "misc.h"

typedef int cost_t;
enum action_t;
__host__ __device__ action_t no_op();

class State
{
friend class Heuristic;
friend class StatePresenter;

public:
    static const int SIDE = 4;
    static const int BUFFER = SIDE*SIDE;
    static const int BLANK = SIDE*SIDE;

private:
    typedef int linear_idx;
    typedef int grid_idx;
    CUDA_CALLABLE_MEMBER static grid_idx row(linear_idx);
    CUDA_CALLABLE_MEMBER static grid_idx col(linear_idx);
    CUDA_CALLABLE_MEMBER static linear_idx idx(grid_idx, grid_idx);

private:
    char board[BUFFER+1];
    linear_idx blankIdx;

public:
    CUDA_CALLABLE_MEMBER State();
    CUDA_CALLABLE_MEMBER State( const char[]);
    CUDA_CALLABLE_MEMBER cost_t perform( action_t);
    CUDA_CALLABLE_MEMBER CudaArray<action_t> actions() const;
    CUDA_CALLABLE_MEMBER bool operator==( const State&) const;
    CUDA_CALLABLE_MEMBER cost_t step( action_t) const;
    CUDA_CALLABLE_MEMBER const char* serialize() const;
    CUDA_CALLABLE_MEMBER char* serialize( char*) const;
};

inline const char* State::serialize() const { return board; }

#endif
