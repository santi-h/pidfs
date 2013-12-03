#include "bfs.h"
#include "StateHash.h"
#include "StateEqualTest.h"
#include <unordered_set>
#include <queue>
#include <deque>
#include <vector>
#include <stack>
#include <climits>

using namespace std;

static unordered_set<State*, StateHash, StateEqualTest> explored;

/**************************************************************************************************
**************************************************************************************************/
struct Node
{
    Node* parent;
    stack_elem_s elem;
    Node( Node* p1, const stack_elem_s& p2) : parent(p1), elem(p2) {}
    Node() : parent(0), elem() {}
};

/**************************************************************************************************
**************************************************************************************************/
class StateCompare
{
public:
    bool operator()(const Node* n1, const Node* n2)
    {
        return (n1->elem.g) > (n2->elem.g);
    }
};

/**************************************************************************************************
**************************************************************************************************/
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

/**************************************************************************************************
* warning: this might return a solution that is not optimal
**************************************************************************************************/
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

/**************************************************************************************************
**************************************************************************************************/
CudaArray<action_t> populate_explored( const State& state)
{
    CudaArray<action_t> ret;
    CudaArray<action_t> aux = state.actions( );
    while( aux.getSize() > 0)
    {
        action_t action = aux.pop();
        State* newState = new State( state);
        newState->perform( action);
        if( explored.count( newState) > 0)
            delete newState;
        else
        {
            explored.insert( newState);
            ret.push( action);
        }
    }
    return ret;
}

/**************************************************************************************************
**************************************************************************************************/
int level( deque<stack_t>& fringe, int limit)
{
    int levels = 0;
    int potential = 0;
    for( auto it = fringe.begin(); it != fringe.end(); it++)
        potential += it->peek().possible.getSize();

    while( potential < limit)
    {
        potential = 0;
        levels++;
        // GO UP ONE LEVEL
        for( size_t nstacks = fringe.size(); nstacks > 0; nstacks--)
        {   // for each stack in fringe
            stack_t& stk = fringe.front();

            while( stk.peek().possible.getSize() > 0)
            {   // for each action in the top element of this stack
                action_t action = stk.peek().possible.pop();
                State state = stk.peek().state;
                cost_t g = stk.peek().g + state.perform( action);
                CudaArray<action_t> possible = populate_explored( state);

                if( possible.getSize() > 0)
                {
                    stack_t newStack = stk;
                    newStack.peek().possible.empty();
                    newStack.push( stack_elem_s(action, state, possible, g));
                    potential += possible.getSize();
                    fringe.push_back( newStack);
                }
            }

            fringe.pop_front();
        }
    }
    
    return levels;
}

/**************************************************************************************************
**************************************************************************************************/
struct basket_s
{
    CudaArray<action_t> balls;
    int type;
    basket_s( const CudaArray<action_t> p1=CudaArray<action_t>(), int p2=0) : balls(p1),type(p2){}
};
class BasketCompare
{
public:
    bool operator()( const basket_s& a, const basket_s& b) const
    {
        return( a.balls.getSize() < b.balls.getSize());
    }
};
CudaArray<stack_t> distribute( const deque<stack_t>& stacks, int nthreads)
{
    priority_queue<basket_s, vector<basket_s>, BasketCompare> baskets;
    for( int i=0; i<stacks.size(); i++)
        baskets.push( basket_s(stacks[i].peek().possible, i));
    
    while( baskets.size() < nthreads)
    {
        // GET BASKETS WITH THE MOST NUMBER OF BALLS, PLACE THEM IN bytype
        vector<basket_s>* bytype = new vector<basket_s>[stacks.size()]();
        //printf("aux0, %d baskets\n", baskets.size());
        int maxballs = baskets.top().balls.getSize();
        int maxtype = -1;
        size_t maxbaskets = 0;
        vector<int> popped;
        for( ; baskets.size() > 0 && baskets.top().balls.getSize() >= maxballs; baskets.pop())
        {
            int type = baskets.top().type;
            bytype[ type].push_back( baskets.top());
            if( bytype[ type].size() == 1) popped.push_back( type);
            if( bytype[ type].size() > maxbaskets)  
            {
                maxtype = type;
                maxbaskets = bytype[ type].size();
            }
        }
        
        // PLACE BACK BASKETS THAT WE ARE NOT SPLITTING
        //printf("aux1, %d baskets, %d popped\n", baskets.size(), popped.size());
        int idx = 0;
        while( popped.size() > 1) if( popped[idx] != maxtype)
        {
            vector< basket_s>& basketspopped = bytype[ popped[idx]];
            for( ; basketspopped.size() > 0; basketspopped.pop_back())
            {
                //printf(" placing back a basket with %d balls\n", basketspopped.back().balls.getSize());
                baskets.push( basketspopped.back());
            }
            popped.erase( popped.begin() + idx);
        }
        else
            idx++;

        //... bytype[ maxtype] contains the baskets with the most balls

        // GET ALL BALLS FROM highbaskets AND EMPTY BASKETS IN IT
        //printf("aux2, %d baskets, %d popped\n", baskets.size(), popped.size());
        vector<basket_s>& highbaskets = bytype[maxtype];
        CudaArray<action_t> balls;
        for( auto it=highbaskets.begin(); it != highbaskets.end(); it++)
        {
            //printf("pushing %d balls\n", it->balls.getSize());
            balls.push( it->balls);
            it->balls.empty();
        }

        // ADD NEW BASKET TO highbaskets AND DISTRIBUTE BALLS IN ROUND ROBBIN
        //printf("aux3 %d balls\n", balls.getSize());
        highbaskets.push_back(basket_s(CudaArray<action_t>(), maxtype));
        for( idx = 0; balls.getSize() > 0; idx = (idx+1)%highbaskets.size())
            highbaskets[idx].balls.push( balls.pop());

        // PLACE BASKETS IN priority_queue
        //printf("aux4\n");
        for( idx=0; idx<highbaskets.size(); idx++)
        {
            //printf(" auxa idx=%d, size=%d\n", idx, highbaskets.size());
            basket_s& b = highbaskets[idx];
            //printf(" auxi\n");
            baskets.push( b);
            //printf(" auxb\n");
        }

        //printf("aux5\n");
        delete[] bytype;
        //printf("aux6\n");
    }
    
    CudaArray<stack_t> ret;
    
    for( ; baskets.size() > 0; baskets.pop())
    {
        basket_s& basket = baskets.top();
        stack_t stk = stacks[basket.type];
        stk.peek().possible = basket.balls;
        ret.push( stk);
    }
    
    return ret;
}

/**************************************************************************************************
**************************************************************************************************/
bfs_ret_s bfs( const State& start, const State& goal, int limit)
{
    bfs_ret_s ret;
    explored.clear();
    explored.insert( new State( start));
    CudaArray<action_t> possible = populate_explored( start);
    stack_t stk;
    stk.push( stack_elem_s( no_op(), start, possible, start.step( no_op())));
    deque<stack_t> fringe;
    fringe.push_front( stk);
    ret.next_cutoff = level( fringe, limit);
    ret.stacks = distribute(fringe, limit);

    return ret;
}




