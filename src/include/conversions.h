#include "config.h"
// #include "spmv_coo.h"
#include "utils.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void coo_csr(int nz, int N, int *row, int *col, double *coo_val, int **row_ptr,
             int **colind, double **csr_val);
void csr_dia(int *row_ptr, int *colind, MYTYPE *val, int **offset,
             MYTYPE **data, int N, int *nd, int *stride, int nnz);
void csr_ell(int *row_ptr, int *colind, double *val, int **indices,
             double **data, int N, int *nc, int nnz);
void csr_dia2dim(int *row_ptr, int *colind, MYTYPE *val, int **offset,
                 MYTYPE ***data, int N, int *nd, int *stride, int nnz);
void csr_dia_feature(int *row_ptr, int *colind, double *val, int **offset,
                     double **data, int N, int *nd, int *stride, int nnz, int **offset_count);
