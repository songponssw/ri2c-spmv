#include "spmv_omp.h"

void spmv_ell_omp(int *indices, double *data, int N, int nc, double *x,
                  double *y) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int rows_per_worker = (N + num_threads - 1) / num_threads;

    int start_row = tid * rows_per_worker;
    int end_row = (tid + 1) * rows_per_worker - 1;
    end_row = (end_row >= N) ? N - 1 : end_row;

    for (int j = 0; j < nc; j++) {
#pragma omp simd
      for (int i = start_row; i <= end_row; i++) {
        y[i] += data[j * N + i] * x[indices[j * N + i]];
      }
    }
  }
}

void spmv_ell_omp_ldist(int *indices, double *data, int N, int nc, double *x,
                          double *y, long long int *ldist) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int rows_per_worker = (N + num_threads - 1) / num_threads;

    int start_row = tid * rows_per_worker;
    int end_row = (tid + 1) * rows_per_worker - 1;
    end_row = (end_row >= N) ? N - 1 : end_row;

    for (int j = 0; j < nc; j++) {
#pragma omp simd
      for (int i = start_row; i <= end_row; i++) {
        y[i] += data[j * N + i] * x[indices[j * N + i]];
        ldist[tid]++;
      }
    }
  }
}
