// #include "config.h"
#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

void spmv_coo_omp(int *row, int *col, double *coo_val, int nz, int N, double *x,
                  double *y, double **y_local);
void spmv_csr_omp(int *row_ptr, int *colind, double *val, int N, double *x,
                  double *y);
void spmv_dia_omp(int *offset, double *data, int N, int nd, int stride,
                  double *x, double *y);
void spmv_ell_omp(int *indices, double *data, int N, int nd, double *x,
                  double *y);
void spmv_coo_omp_ldist(int *rowind, int *colind, double *val, int nz, int N,
                        double *x, double *y, double **y_local,
                        long long int *ldist);
void spmv_csr_omp_ldist(int *row_ptr, int *colind, double *val, int N,
                        double *x, double *y, long long int *ldist);
void spmv_dia_omp_ldist(int *offset, double *data, int N, int nd, int stride,
                        double *x, double *y, long long int *ldist);
void spmv_ell_omp_ldist(int *indices, double *data, int N, int nd, double *x,
                        double *y, long long int *ldist);
