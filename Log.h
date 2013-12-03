#ifndef LOG_H
#define LOG_H

class State;
enum action_t;
template< class T> class CudaArray;

#define BRANCHING_FACTOR 4
enum event_e
{
    PUSH,
    POP,
    DEL_EXPLORED,
    DEL_CUTOFF,
    GOAL
};

struct log_s
{
    int timestamp;
    int tid;
    int iteration;
    int cutoff;
    event_e event;
};

struct del_log_s
{
    log_s log;
    action_t action;
};

struct push_log_s
{
    log_s log;
    action_t action;
    action_t actions[BRANCHING_FACTOR];
    char state[1];
};

__device__ bool logGoal( int timestamp, int iteration, int cutoff);
__device__ bool logPush( int timestamp, int iteration, int cutoff, CudaArray<action_t>& actions, const State& state, action_t action);
__device__ bool logPop( int timestamp, int iteration, int cutoff, CudaArray<action_t>& actions, const State& state, action_t action);
__device__ bool logDelExplored( int timestamp, int iteration, int cutoff, action_t action);
__device__ bool logDelCutoff( int timestamp, int iteration, int cutoff, action_t action);

#endif
