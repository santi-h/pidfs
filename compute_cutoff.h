#ifndef COMPUTE_CUTOFF_H
#define COMPUTE_CUTOFF_H

struct shared_thread_s;

__device__ void compute_cutoff(shared_thread_s*, int, int*, int*);

#endif
