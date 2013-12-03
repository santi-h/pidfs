#include "State.h"

enum action_t
{
    UP = -State::SIDE,
    DOWN = State::SIDE,
    NOP = 0,
    LEFT = -1,
    RIGHT = 1
};

__host__ __device__ action_t no_op()
{
    return NOP;
}

State::State()
{
    for( char i=0; i<BUFFER-1; i++) board[i] = i+1;
    board[BUFFER-1] = BLANK;
    board[BUFFER] = '\0';
    blankIdx = BUFFER-1;
}

State::State( const char b[])
{
    for( char i=0; i<BUFFER; i++)
    {
        board[i] = b[i];
        if( b[i] == BLANK) blankIdx = i;
    }
    board[BUFFER] = '\0';
}

cost_t State::perform( action_t action)
{
    board[blankIdx] = board[blankIdx+action];
    board[blankIdx+action] = BLANK;
    blankIdx = blankIdx+action;

    return 1;
}

CudaArray<action_t> State::actions() const
{
    CudaArray<action_t> ret;

    if( col(blankIdx) == col(blankIdx+UP)) ret.push(UP);
    if( col(blankIdx) == col(blankIdx+DOWN)) ret.push(DOWN);
    if( row(blankIdx) == row(blankIdx+LEFT)) ret.push(LEFT);
    if( row(blankIdx) == row(blankIdx+RIGHT)) ret.push(RIGHT);

    return ret;
}

bool State::operator==( const State& o) const
{
    for( int i=0; i<BUFFER; i++) if( board[i] != o.board[i]) return 0;
    return 1;
}

State::grid_idx State::row(linear_idx idx)
{
    return (idx<0 || idx>=BUFFER) ? -1 : idx/SIDE;
}

State::grid_idx State::col(linear_idx idx)
{
    return (idx<0 || idx>=BUFFER) ? -1 : idx%SIDE;
}

State::linear_idx State::idx(grid_idx r, grid_idx c)
{
    if( r < 0 || r >= SIDE) return -1;
    if( c < 0 || c >= SIDE) return -1;
    return r*SIDE+c;
}

cost_t State::step( action_t a) const
{
    return (a != NOP);
}

char* State::serialize( char* dst) const
{
    for( int i=0; i<BUFFER; i++)
        dst[i] = board[i] == BLANK ? 'a' : 'a'+board[i];

    dst[BUFFER] = 0;
    return dst;
}

