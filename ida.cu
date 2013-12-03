#include "ida.h"

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
 __device__ void ida( shared_block_s& shared, cost_t cutoff, CudaArray<action_t>& solution, int iteration, stack_t& initialStack)
{
     int tid = threadIdx.x;
    shared_thread_s& thread = shared.threads[tid];
    cost_t next_cutoff = thread.next_cutoff;
    bool exitIda = 0;
    stack_t stack = initialStack;
    while( !exitIda)
    {
        if( stack.getSize() <= 0)
        {
            //... increase iteration
            cutoff = next_cutoff;
            //printf("tid %d cutoff %d\n", tid, cutoff);
            stack = initialStack;
            next_cutoff = INT_MAX;
        }

        if( stack.getSize() > 0)
        {
            if( shared.solution_found)
            {//... a solution was found by another thread
                if( solution.getSize()-1 <= cutoff)
                {//... the solution is at least as optimal as the solution we could find
                    exitIda = 1;
                    //printf("tid %d not doing cutoff %d\n", tid, cutoff);
					//printf("setting exit to true, tid=%d exitIda=%d\n", tid, exitIda);
                }
            }

            if( !exitIda)
            {//... no solution was found yet, do IDA
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
                        }
                        else
                        {
                            if( child == shared.goal)
                            {
                                while( atomicExch(&shared.solution_lock, 1) != 0);
                                //... critical section
                                if( !shared.solution_found || cutoff < solution.getSize()-1)
                                {
                                    printf("goal found by tid=%d\n", tid);
                                    int stack_size = stack.getSize();
                                    for( int i=0; i<stack_size; i++) solution.push(stack[i].action);
                                    solution.push(child_action);
                                }
                                shared.solution_found = 1;
                                atomicExch(&shared.solution_lock, 0); // release lock
                                exitIda = 1;
                            }
                            else
                                stack.push( stack_elem_s(child_action, child, child.actions(), child_g));
                        }
                    }
                }
                else
                    stack.pop();
            }
        }
    }
 }
 
__device__ bool _explored(const CudaArray<stack_elem_s>& stack, const State& state)
{
    int limit = stack.getSize();
    for( int i=0; i<limit; i++) if( stack[i].state == state) return 1;
    return 0;
}
