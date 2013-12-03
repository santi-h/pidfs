#include "bfs.h"
#include "StateHash.h"
#include "StateEqualTest.h"
#include <unordered_set>
#include <queue>
#include <vector>
#include <stack>

using namespace std;

static unordered_set<State*, StateHash, StateEqualTest> explored;

struct Node
{
    Node* parent;
    stack_elem_s elem;
    Node( Node* p1, const stack_elem_s& p2) : parent(p1), elem(p2) {}
    Node() : parent(0), elem() {}
};

class StateCompare
{
public:
    bool operator()(const Node* n1, const Node* n2)
    {
        return (n1->elem.g) > (n2->elem.g);
    }
};

CudaArray<action_t> checkGoal( CudaArray<stack_elem_s>& stk, const State& goal)
{
    CudaArray<action_t> ret;

    if( stk.peek().state == goal)
    {
        CudaArray<stack_elem_s> aux;
        for( ; stk.getSize(); stk.pop()) aux.push( stk.peek());
        for( ; aux.getSize(); aux.pop()) ret.push( aux.peek().action);
    }

    return ret;
}

// warning: this might return a solution that is not optimal
CudaArray<action_t> expand( CudaArray<stack_elem_s>& stk, const State& goal)
{
    CudaArray<action_t> ret;

    while( stk.peek().possible.getSize() == 1 && ret.getSize() <= 0)
    {
        action_t action = stk.peek().possible.pop();
        State state = stk.peek().state;
        cost_t g = stk.peek().g + state.perform( action);
        stk.push( stack_elem_s( action, state, state.actions(), g));
        ret = checkGoal( stk, goal);
    }

    return ret;
}

bfs_ret_s bfs( const State& start, const State& goal, int limit)
{
    bfs_ret_s ret;
    queue<CudaArray<stack_elem_s>> q;
    CudaArray<stack_elem_s> stack0;
    stack0.push( stack_elem_s(no_op(),start,start.actions(),start.step(no_op())));
    q.push( stack0);
    while( ret.solution.getSize() <= 0 && q.size() < limit)
    {
        CudaArray<stack_elem_s>& highStack = q.front();

        // assuming elem.possible.getSize() > 1
        CudaArray<stack_elem_s> lowStack = highStack;
        highStack.peek().possible.empty();
        while( highStack.peek().possible.getSize() < lowStack.peek().possible.getSize())
            highStack.peek().possible.push(lowStack.peek().possible.pop());
        
        CudaArray<action_t> ret1 = expand( highStack, goal);
        CudaArray<action_t> ret2 = expand( lowStack, goal);
        
        ret.solution = ret1.getSize() < ret2.getSize() ? ret1 : ret2;

        q.push( highStack);
        q.push( lowStack);
        q.pop();
    }

    if( ret.solution.getSize() <= 0)
    {
        ret.next_cutoff = q.front().peek().g;
        for( ; q.size(); q.pop()) ret.stacks.push( q.front());
    }
    
    return ret;
}




