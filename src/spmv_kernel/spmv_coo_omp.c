#include "spmv_omp.h"

void spmv_coo_omp(int *rowind, int *colind, double *val, int nz, int N,
                  double *x, double *y, double **y_local) {

  int num_threads = omp_get_max_threads();

#pragma omp parallel for
  for (int i = 0; i < nz; i++) {
    int tid = omp_get_thread_num();
    y_local[tid][rowind[i]] += val[i] * x[colind[i]];
  }

  for (int i = 0; i < N; i++) {
    for (int t = 0; t < num_threads; t++) {
      y[i] += y_local[t][i];
    }
  }
}

void spmv_coo_omp_ldist(int *rowind, int *colind, double *val, int nz, int N,
                        double *x, double *y, double **y_local,
                        long long int *ldist) {

  int num_threads = omp_get_max_threads();

#pragma omp parallel for
  for (int i = 0; i < nz; i++) {
    int tid = omp_get_thread_num();
    y_local[tid][rowind[i]] += val[i] * x[colind[i]];
    ldist[tid]++;
  }

  for (int i = 0; i < N; i++) {
    for (int t = 0; t < num_threads; t++) {
      y[i] += y_local[t][i];
    }
  }
}
