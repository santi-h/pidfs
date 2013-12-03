#ifndef STATEPRESENTER_H
#define STATEPRESENTER_H

#include <string>

class State;

class StatePresenter
{
private:
    static int drawLine(char*,int,char='+',char='-');
    static int drawCells(char*,int,const char*,char='|');

public:
    static std::string toString( const State&);
};

#endif
