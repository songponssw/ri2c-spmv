#include "spmv_omp.h"

void spmv_dia_omp(int *offset, double *data, int N, int nd, int stride,
                  double *x, double *y) {
#pragma omp parallel
  for (int i = 0; i < nd; i++) {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int rows_per_worker = (N + num_threads - 1) / num_threads;

    int start_row = tid * rows_per_worker;
    int end_row = (tid + 1) * rows_per_worker - 1;
    end_row = (end_row >= N) ? N - 1 : end_row;

    int k = offset[i];
    int index = (k < 0) ? N - stride : 0;
    int istart = (0 < -k) ? -k : 0;
    int iend = (N - 1 < N - 1 - k) ? N - 1 : N - 1 - k;

    istart = (istart > start_row) ? istart : start_row;
    iend = (iend < end_row) ? iend : end_row;

#pragma omp simd
    for (int n = istart; n <= iend; n++) {
      y[n] += data[(size_t)i * stride + n - index] * x[n + k];
    }
  }
}

void spmv_dia_omp_ldist(int *offset, double *data, int N, int nd, int stride,
                           double *x, double *y, long long int *ldist) {
#pragma omp parallel
  for (int i = 0; i < nd; i++) {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int rows_per_worker = (N + num_threads - 1) / num_threads;

    int start_row = tid * rows_per_worker;
    int end_row = (tid + 1) * rows_per_worker - 1;
    end_row = (end_row >= N) ? N - 1 : end_row;

    int k = offset[i];
    int index = (k < 0) ? N - stride : 0; // Initialize `index` unconditionally
    int istart = (0 < -k) ? -k : 0;
    int iend = (N - 1 < N - 1 - k) ? N - 1 : N - 1 - k;

    istart = (istart > start_row) ? istart : start_row;
    iend = (iend < end_row) ? iend : end_row;

#pragma omp simd
    for (int n = istart; n <= iend; n++) {
      y[n] += data[(size_t)i * stride + n - index] * x[n + k];
      ldist[tid]++;
    }
  }
}
