#include "start_states.h"
#include "State.h"

using namespace std;

vector<State> start_states()
{
    vector<State> ret;
    
    char board0[] = {    1,  2,  7,  4,
                         5, 10,  3,  8,
                         6, 14, 13, 12,
                         9, 11, 15, State::BLANK};

    char board1[] = {   14, 13, State::BLANK, 4,
                         1,  7,  3,  8,
                         6,  2,  5, 10,
                         9, 12, 15, 11};

    char board2[] = {    6,  8, State::BLANK, 3,
                        10,  9,  7,  2,
                         5, 13, 12, 15,
                        14,  4, 11,  1};

    char board3[] = {    1,  2,  7,  3,
                         5,  6,  4,  8,
                         9, 10, 11, 12,
                        13, 14, 15, State::BLANK}; 

    char board4[] = {    2,  3,  4,  8,
                         1,  6,  7, 12,
                        11, 14, 13, 15,
                         5,  9, State::BLANK, 10};


    char board5[] = {    2,  5,  1,  4,
                         9, 10,  3,  8,
                        State::BLANK, 6, 7, 15,
                        14, 13, 12, 11};

    char board6[] = {    5,  1,  2,  8,
                         9, 10,  4, 15,
                        14,  3, 11,  7,
                        13, 12,  6, State::BLANK};

    char board7[] = {    9,  5,  2,  3,
                         1,  6,  4,  8,
                        13,  7, 12, 15,
                        11, 10, 14, State::BLANK};

    char board8[] = {    1,  2,  7,  3,
                         5,  6,  11, 4,
                         9, 15,  State::BLANK, 8,
                         13, 10, 14, 12}; // depth 10

    char board9[] = {    1,  2,  3,  4,
                         5,  7,  State::BLANK, 8,
                         9,  6, 10, 11,
                        13, 14, 15, 12}; // depth 5
                        
    char board10[] = {    1,  2,  3,  4,
                         6,  10,  7, 8,
                        13,   9, 15, 11,
                         5,  14, State::BLANK, 12}; // depth 15

    char board11[] = {    1,  2,  3,  4,
                         10,  9,  6,  State::BLANK,
                          5, 11,  8,  7,
                         13, 14, 15, 12}; // depth 12
                         
    char board12[] = {    2,  5,  3,  4,
                          9,  1,  7,  8,
                          State::BLANK, 6,  11,  12,
                         13, 10, 14, 15}; // depth 10

    ret.push_back( State(board0));
    ret.push_back( State(board1));
    ret.push_back( State(board2));
    ret.push_back( State(board3));
    ret.push_back( State(board4));
    ret.push_back( State(board5));
    ret.push_back( State(board6));
    ret.push_back( State(board7));
    ret.push_back( State(board8));
    ret.push_back( State(board9));
    ret.push_back( State(board10));
    ret.push_back( State(board11));
    ret.push_back( State(board12));

    return ret;
}
