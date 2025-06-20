// #include "spmv_coo.h"
// #include "config.h"
#include "spmv.h"

void spmv_coo(int *rowind, int *colind, double *val, int nz, int N, double *x,
              double *y) {
  int i;
  for (i = 0; i < nz; i++) {
    y[rowind[i]] += val[i] * x[colind[i]];
  }
}

void spmv_coo_x0_fix(int *rowind, int *colind, double *val, int nz, int N,
                     double *x, double *y) {
  int i;
  for (i = 0; i < nz; i++) {
    y[rowind[i]] += val[i] * x[0];
  }
}

void spmv_coo_col0_fix(int *rowind, int *colind, double *val, int nz, int N,
                     double *x, double *y) {
  int i;
  for (i = 0; i < nz; i++) {
    y[rowind[i]] += val[i] * x[colind[0]];
  }
}


void spmv_coo_ai(int *rowind, int *colind, double *val, int nz, int N,
                 double *x, double *y) {
  int i;
  for (i = 0; i < nz; i++) {
    y[rowind[i]] += val[i] * val[i];
  }
}
