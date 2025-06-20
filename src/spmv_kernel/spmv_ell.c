#include "spmv.h"

void spmv_ell(int *indices, double *data, int N, int nc, double *x,
                   double *y) {
  for (int j = 0; j < nc; j++) {
    for (int i = 0; i < N; i++) {
      y[i] += data[j * N + i] * x[indices[j * N + i]];
    }
  }
}
