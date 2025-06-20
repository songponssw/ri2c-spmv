#include <stddef.h>

void spmv_coo(int *row, int *col, double *coo_val, int nz, int N, double *x, double *y);
void spmv_csr(int *row_ptr, int *colind, double*val, int N, double *x, double *y);
void spmv_dia(int *offset, double *data, int N, int nd, int stride, double *x, double *y);
void spmv_ell(int *indices, double *data, int N, int nd, double *x, double *y);
