#include "StatePresenter.h"
#include "State.h"

using namespace std;

string StatePresenter::toString( const State& state)
{
    const int meat_size = 4;
    const int line_size = State::SIDE * (meat_size + 1) + 2;
    const int n_lines = State::SIDE*2 + 1;
    const int buf_size = line_size * n_lines + 1;

    char* cret = new char[buf_size];
    char* buf = cret;
    buf[buf_size-1] = 0;

    for( int i=0; i<State::SIDE; i++)
    {
        buf += drawLine(buf, 4);
        buf += drawCells(buf, 4, &state.board[State::SIDE*i]);
    }
    buf += drawLine(buf, 4);

    string ret = cret;
    delete cret;
    return ret;
}

int StatePresenter::drawLine( char* buf, int meatCount,  char bread, char meat)
{
    for( int i=0; i<State::SIDE; i++)
    {
        *buf++ = bread;
        for( int j=0; j<meatCount; j++) *buf++ = meat;
    }
    *buf++ = bread;
    *buf++ = '\n';
    
    return meatCount*State::SIDE+State::SIDE+2;
}

int StatePresenter::drawCells( char* buf, int meatCount, const char* cells, char bread)
{
    char aux[10];

    for( int i=0; i<State::SIDE; i++)
    {
        *buf++ = bread;
        int printed = 0;
        if( cells[i] == State::BLANK)
        {
            aux[0] = ' ';
            aux[1] = '\0';
            printed = 1;
        }
        else
            printed = sprintf_s(aux, sizeof(aux),"%d", cells[i]);

        int spaces = meatCount - printed;
        while(spaces-- > 0) *buf++ = ' ';
        strncpy_s(buf, spaces, aux, spaces);
        buf += printed;
    }
    
    *buf++ = bread;
    *buf++ = '\n';

    return meatCount*State::SIDE+State::SIDE+2;
}


