#ifndef STATEEQUALTEST_H
#define STATEEQUALTEST_H

#include "State.h"

class StateEqualTest
{
public:
    bool operator()( const State*, const State*) const;
};

inline bool StateEqualTest::operator()( const State* s1, const State* s2) const
{
    return *s1 == *s2;
}

#endif
