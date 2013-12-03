#include "ida.h"
#include "Log.h"

/*
 * struct ida_input
 * {
 *     CudaArray<State>* stack; // stack to use
 *     int* bar; // shared work delimiter of the stack
 *     lock_t* bar_lock; // lock to reduce bar
 *     const State* goal; // goal state
 *     const Heuristic* h; // heuristic fucntion
 *     CudaArray<action_t>* solution; // to store the solution
 *     lock_t* solution_lock; // lock to store solution
 *     cost_t cutoff; // this iteration's cutoff
 *     cost_t* next_cutoff; // to put the next cutoff
 * };
 */
 __device__ void ida( shared_block_s& shared, cost_t cutoff, CudaArray<action_t>& solution, int iteration)
 {
    int tid = threadIdx.x;
    shared_thread_s& thread = shared.threads[tid];
    stack_t& stack = thread.stack;
    cost_t next_cutoff = thread.next_cutoff;
    int stack_size = stack.getSize();

    while( stack_size && !shared.solution_lock)
    {
        stack_elem_s& top = stack.peek();
        if( top.possible.getSize())
        {
            action_t child_action = top.possible.pop();
            State child = top.state;
            cost_t child_g = top.g + child.perform( child_action);
            if( !_explored(stack, child))       //// EXPENSIVE
            {
                if( child_g > cutoff)
                {
                    if( child_g < next_cutoff) next_cutoff = child_g;
                    if( LOG) logDelCutoff( thread.timestamp, iteration, cutoff, child_action);
                }
                else
                {
                    if( child == shared.goal)
                    {
                        if( atomicExch(&shared.solution_lock, 1) == 0)
                        {
                            printf("goal found by tid=%d\n", tid);
                            for( int i=0; i<stack_size; i++) solution.push(stack[i].action);
                            solution.push(child_action);
                        }
                        if( LOG) logGoal( thread.timestamp, iteration, cutoff);
                    }
                    else
                    {
                        // PUSH
                        CudaArray<action_t> actions = child.actions();
                        stack.push( stack_elem_s(child_action, child, actions, child_g));
                        stack_size++;
                        if( LOG) logPush( thread.timestamp, iteration, cutoff, actions, child, child_action);
                    }
                }
            }
            else
                if( LOG) logDelExplored( thread.timestamp, iteration, cutoff, child_action);
        }
        else
        {
            // POP
            stack.pop();
            stack_size--;
            if( LOG) logPop( thread.timestamp, iteration, cutoff, top.possible, top.state, top.action);
        }

        thread.timestamp++;
    }
    
    thread.next_cutoff = next_cutoff;
 }
 
__device__ bool _explored(const CudaArray<stack_elem_s>& stack, const State& state)
{
    /*
	for( int i=stack.getSize()-1, c=0; i>=0 && c < 3; i--, c++)
		if( stack[i].state == state) return 1;

	return 0;
    //*/
	//*
    int limit = stack.getSize();
    for( int i=0; i<limit; i++) if( stack[i].state == state) return 1;
    return 0;
	//*/
}
