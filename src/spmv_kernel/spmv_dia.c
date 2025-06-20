#include "spmv.h"

void spmv_dia(int *offset, double *data, int N, int nd, int stride, double *x, double*y)
{
  int k, istart, iend, index;

  for(int i = 0; i < nd; i++){
    k = offset[i];
    index = 0;
    istart = (0 < -k) ? index = N-stride, -k : 0;   
    iend = (N-1 < N-1-k) ? N-1 : N-1-k;
    for(int n = istart; n <= iend; n++){
      y[n] += (data[(size_t)i*stride+n-index] * x[n+k]);
    } 
  }
}
