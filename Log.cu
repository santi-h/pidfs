#include "Log.h"
#include "State.h"
#include "CudaArray.h"
#include "devicelog.h"

__device__ bool _logPushPop(int timestamp, 
                            int iteration, 
                            int cutoff, 
                            CudaArray<action_t>& actions, 
                            const State& state,
                            action_t action,
                            event_e event)
{
    const char* str = state.serialize();
    int str_len = 0;
    while( str[str_len]) str_len++;
    size_t bytes = sizeof(push_log_s) + str_len;
    
    push_log_s* out = (push_log_s*)malloc(bytes);
    out->log.timestamp = timestamp;
    out->log.tid = threadIdx.x;
    out->log.iteration = iteration;
    out->log.cutoff = cutoff;
    out->log.event = event;
    int i;
    for( i=0; i<actions.getSize(); i++)
        out->actions[i] = actions[i];
    if( i<BRANCHING_FACTOR) out->actions[i] = no_op();

    out->state[0] = 0;
    for( i=0; i<str_len; i++) out->state[i] = str[i];
    out->state[str_len] = 0;
    out->action = action;

    log(out,bytes);
    free( out);
    
    return 1;
}

__device__ bool _logDel( int timestamp, int iteration, int cutoff, action_t action, event_e event)
{
    del_log_s out;
    out.log.timestamp = timestamp;
    out.log.tid = threadIdx.x;
    out.log.iteration = iteration;
    out.log.cutoff = cutoff;
    out.log.event = event;
    out.action = action;
    
    log(&out,sizeof(del_log_s));
    return 1;
}

__device__ bool logGoal( int timestamp, int iteration, int cutoff)
{
    log_s out;
    out.timestamp = timestamp;
    out.tid = threadIdx.x;
    out.iteration = iteration;
    out.cutoff = cutoff;
    out.event = GOAL;
    
    log(&out,sizeof(log_s));
    return 1;
}

__device__ bool logDelExplored( int timestamp, int iteration, int cutoff, action_t action)
{
    return _logDel( timestamp, iteration, cutoff, action, DEL_EXPLORED);
}

__device__ bool logDelCutoff( int timestamp, int iteration, int cutoff, action_t action)
{
    return _logDel( timestamp, iteration, cutoff, action, DEL_CUTOFF);
}

__device__ bool logPop( int timestamp, int iteration, int cutoff)
{
    log_s out;
    out.timestamp = timestamp;
    out.tid = threadIdx.x;
    out.iteration = iteration;
    out.cutoff = cutoff;
    out.event = POP;
    
    log( &out, sizeof(log_s));
    return 1;
}

__device__ bool logPush(    int timestamp, 
                            int iteration, 
                            int cutoff, 
                            CudaArray<action_t>& actions, 
                            const State& state,
                            action_t action)
{
    return _logPushPop( timestamp, iteration, cutoff, actions, state, action, PUSH);
}

__device__ bool logPop(     int timestamp, 
                            int iteration, 
                            int cutoff, 
                            CudaArray<action_t>& actions, 
                            const State& state,
                            action_t action)
{
    return _logPushPop( timestamp, iteration, cutoff, actions, state, action, POP);
}

