#ifndef STATEHASH_H
#define STATEHASH_H

#include <functional>
#include <string>
#include "State.h"

class StateHash
{
public:
    size_t operator()( const State*) const;
};

inline size_t StateHash::operator()( const State* state) const
{
    return std::hash<std::string>()( std::string(state->serialize()));
}

#endif
