#include "spmv_omp.h"

void spmv_csr_omp(int *row_ptr, int *colind, double *val, int N, double *x,
                  double *y) {
#pragma omp parallel for default(shared)
#pragma ivdep
  for (int i = 0; i < N; i++) {
    double temp = 0;
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      temp += val[j] * x[colind[j]];
    }
    y[i] += temp;
  }
}

void spmv_csr_omp_ldist(int *row_ptr, int *colind, double *val, int N,
                        double *x, double *y, long long int *ldist) {
#pragma omp parallel for default(shared)
#pragma ivdep
  for (int i = 0; i < N; i++) {

    int tid = omp_get_thread_num();
    double temp = 0;
    int counter = 0;
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      temp += val[j] * x[colind[j]];
      counter++;
    }
    y[i] += temp;
    ldist[tid] += counter;
  }
}
