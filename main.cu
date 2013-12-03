#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>
#include <queue>
#include <climits>
#include <unordered_set>
#include <functional>

#include "State.h"
#include "StatePresenter.h"
#include "kernel.h"
#include "common_def.h"
#include "start_states.h"
#include "bfs.h"
#include "StateHash.h"
#include "StateEqualTest.h"
#include "DeviceMemory.h"
#include "Log.h"
#include "hostlog.h"

using namespace std;

/**************************************************************************************************
* returns 1 if eof reached, otherwise it returns 0 and rewinds
**************************************************************************************************/
int read_and_rewind( void* dst, size_t bytes, FILE* file)
{
    fpos_t pos;
    fgetpos( file, &pos);
    size_t bytesRead = fread ( dst, bytes, 1, file);
    if( feof( file)) return EOF;
    fsetpos( file, &pos);
    return 0;
}

/**************************************************************************************************
**************************************************************************************************/
void createLogFile( int timestamps, CudaArray<stack_t>& stacks)
{
    FILE* logs[THREADS_PER_BLOCK];
    FILE* file = fopen("log.csv", "w");
    fprintf( file, "%d\n",THREADS_PER_BLOCK);
    fprintf( file, "%d\n", timestamps);
    fprintf( file, "%d\n", BRANCHING_FACTOR);

    char hshstr[State::BUFFER+1];
    char buff[128];
    for( int i=0; i<THREADS_PER_BLOCK; i++)
    {
        sprintf(buff, "%d", i);
        char filename[80] = "log";
        strcat(strcat( filename, buff), ".txt");
        logs[i] = fopen(filename, "rb");
        
        stack_t& stack = stacks[i];
        for( int j=0; j<stack.getSize(); j++)
        {
            stack_elem_s& entry = stack[j];
            CudaArray<action_t> possible = entry.possible;
            fprintf(file, "%d,%s,%d", entry.action, entry.state.serialize(hshstr), possible.getSize());
            while( possible.getSize()>0)
                fprintf(file, ",%d", possible.pop());
            
            if( j<stack.getSize()-1) fprintf(file,",");
        }
        fprintf(file, "\n");

    }
    
    bool loop = 1;
    int timestamp = 0;
    while( loop)
    {
        loop = 0;
        for( int i=0; i<THREADS_PER_BLOCK; i++) if( !feof(logs[i]))
        {
            log_s current_log;
            if( read_and_rewind( &current_log, sizeof(log_s), logs[i]) != EOF)
            {
                int bytes = 0;
                if( current_log.timestamp == timestamp)
                {
                    if( current_log.event == PUSH || current_log.event == POP)
                    {
                        fpos_t pos; fgetpos( logs[i], &pos);
                        push_log_s log;
                        fread ( &log, sizeof(push_log_s), 1, logs[i]);
                        int str_len = 0;
                        hshstr[str_len++] = log.state[0]== State::BLANK ? 'a' : 'a' +log.state[0];
                        hshstr[str_len++] = log.state[1]== State::BLANK ? 'a' : 'a' +log.state[1];
                        hshstr[str_len++] = log.state[2]== State::BLANK ? 'a' : 'a' +log.state[2];
                        hshstr[str_len++] = log.state[3]== State::BLANK ? 'a' : 'a' +log.state[3];
                        char ch;
                        while( (ch = fgetc(logs[i])) > 0) {
                            hshstr[str_len++] = ch == State::BLANK ? 'a' : 'a' + ch;
                        }
                        fsetpos( logs[i], &pos);
                        hshstr[str_len] = 0;
                        bytes = sizeof(push_log_s) + str_len;
                        loop = 1;
                        fprintf(file, "%d,%d,%d,%d,%s,%d,%s",
                            current_log.iteration,
                            current_log.cutoff,
                            current_log.tid,
                            current_log.timestamp,
                            current_log.event == PUSH ? "PUSH" : "POP",
                            log.action,
                            hshstr);

                        for( int j=0; j<BRANCHING_FACTOR && log.actions[j] != no_op(); j++)
                            fprintf(file, ",%d", log.actions[j]);
                        fprintf(file, "\n");
                    }
                    else if( current_log.event == DEL_EXPLORED)
                    {
                        del_log_s log;
                        read_and_rewind( &log, sizeof(del_log_s), logs[i]);
                        bytes = sizeof( del_log_s);
                        loop = 1;
                        fprintf(file, "%d,%d,%d,%d,DEL_EXPLORED,%d\n",
                            current_log.iteration,
                            current_log.cutoff,
                            current_log.tid,
                            current_log.timestamp,
                            log.action);
                    }
                    else if( current_log.event == DEL_CUTOFF)
                    {
                        del_log_s log;
                        read_and_rewind( &log, sizeof(del_log_s), logs[i]);
                        bytes = sizeof( del_log_s);
                        loop = 1;
                        fprintf(file, "%d,%d,%d,%d,DEL_CUTOFF,%d\n",
                            current_log.iteration,
                            current_log.cutoff,
                            current_log.tid,
                            current_log.timestamp,
                            log.action);
                    }
                    else if( current_log.event == GOAL)
                    {
                        bytes = sizeof( log_s);
                        loop = 1;
                        fprintf(file, "%d,%d,%d,%d,GOAL\n",
                            current_log.iteration,
                            current_log.cutoff,
                            current_log.tid,
                            current_log.timestamp);
                    }
                    else
                    {
                        printf( "UNKNOWN!\n");
                    }
                }
                while( bytes-- > 0) fgetc( logs[i]);
            }
        }
        timestamp++;
        
    }
    
    
    for( int i=0; i<THREADS_PER_BLOCK; i++)
    {
        assert( fgetc(logs[i]) == EOF);
        fclose( logs[i]);
    }
    fclose( file);
}

/**************************************************************************************************
**************************************************************************************************/
void sendout( CudaArray<stack_t>& fringe)
{
    stack_t* fringe_cpy = (stack_t*)malloc(fringe.getSize() * sizeof(stack_t));
    memcpy( fringe_cpy, fringe.data(), fringe.getSize()*sizeof(stack_t));
    for( int i=0; i<fringe.getSize(); i++)
    {
        stack_t& stack = fringe_cpy[i];
        stack_elem_s* stack_cpy = (stack_elem_s*)malloc(stack.getSize()*sizeof(stack_elem_s));
        memcpy(stack_cpy, stack.data(), stack.getSize()*sizeof(stack_elem_s));

        for( int j=0; j<stack.getSize(); j++)
            stack_cpy[j].possible.sendout_safe();

        stack_elem_s* d_stack_cpy = (stack_elem_s*)s_allocToDevice( stack_cpy, stack.getSize()*sizeof(stack_elem_s));
        free( stack_cpy);
        stack.replace_safe( d_stack_cpy, stack.getSize());
    }
    stack_t* d_fringe_cpy = (stack_t*)s_allocToDevice(fringe_cpy, fringe.getSize()*sizeof(stack_t));
    free( fringe_cpy);
    fringe.replace(d_fringe_cpy, fringe.getSize());
}

/**************************************************************************************************
**************************************************************************************************/
void bringin( CudaArray<stack_t>& stacks)
{
    stacks.bringin();
    for( int i=0; i<stacks.getSize(); i++)
    {
        stacks[i].bringin();
        for( int j=0; j<stacks[i].getSize(); j++)
            stacks[i][j].possible.bringin();
    }
}

/**************************************************************************************************
**************************************************************************************************/
int main(void)
{
    vector<State> starts = start_states();

    // set start state
    State start = starts[11];
    string str = StatePresenter::toString( start);
    printf("%s\n", str.c_str());
    str = StatePresenter::toString( State());
    printf("%s\n", str.c_str());

    bfs_ret_s res = bfs( start, State(), THREADS_PER_BLOCK);

    printf("res.stacks=%d res.solution=%d res.next_cutoff=%d\n", res.stacks.getSize(), res.solution.getSize(), res.next_cutoff);
    if( res.stacks.getSize())
    {
        // SET UP global_s
        global_s h_global;
        h_global.stacks = res.stacks;
        sendout( h_global.stacks);
        h_global.solution.setBuffer(85);
        h_global.solution.sendout();
        h_global.initial_cutoff = res.next_cutoff;
        h_global.timestamps = 0;
        h_global.pushes = 0;
        
        // MOVE INPUTS TO GLOBAL MEMORY
        DeviceMemory<global_s> d_global = h_global;

        // PERFORM KERNEL CALL
        clock_t starttime = clock();
        if( LOG) cudaStartLog(1, THREADS_PER_BLOCK, "SEPARATE");
        kernel<<<1,THREADS_PER_BLOCK, sizeof(shared_block_s)>>>( d_global);
        gpuErrchk( "cudaDeviceSynchronize()", cudaDeviceSynchronize()); 
        if( LOG) cudaStopLog();
        clock_t finish = clock();
        clock_t clocks = finish-starttime;
        double ms = (((double)clocks)/CLOCKS_PER_SEC)*1000;

        // BRING IN RETURN VALUES
        d_global.set( h_global);
        h_global.solution.bringin();
        printf("[%.2fms]solution found: %d\n", ms, h_global.solution.getSize());
        bringin( h_global.stacks);
        
        // CREATE LOG FILE
        if( LOG) createLogFile( h_global.timestamps, h_global.stacks);
    }
    
    int input = 0;
    printf("enter a number to exit... ");
    scanf("%d", &input);
    
    return 0;
}

