#include "spmv.h"
// #include "config.h"

void spmv_csr(int *row_ptr, int *colind, double *val, int N, double *x,
              double *y) {
  int i, j;
  double temp;
  for (i = 0; i < N; i++) {
    temp = 0;
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      temp += val[j] * x[colind[j]];
    }
    y[i] += temp;
  }
}

// Print address of x,y: Show that address order
// void spmv_csr_addr(int *row_ptr, int *colind, double *val, int N, double *x,
//                    double *y) {
//   int i, j;
//   double temp;
//   double *prev_addr = &x[colind[0]];
//   int num_cl = 0;
//   for (i = 0; i < N; i++) {
//     temp = 0;
//     printf("Matrix A \t Dense vector X\n");
//     for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
//       temp += val[j] * x[colind[j]];
//       ptrdiff_t diff = &x[colind[j]] - prev_addr;
//
//       printf("val[%d] * x[%d] \t\n", j, colind[j]);
//       printf("%p x %p \t diff: %td\n\n", &val[j], &x[colind[j]], diff);
//
//       // Same cache line or not
//       if (abs(diff) > 8) {
//         num_cl += 1;
//       }
//       prev_addr = &x[colind[j]];
//     }
//     printf("num_cl: %d\n", num_cl);
//     printf("--------\n");
//     num_cl = 0;
//     y[i] += temp;
//   }
// }
//
// void spmv_csr_x0_fix(int *row_ptr, int *colind, double *val, int N, double
// *x,
//                      double *y) {
//   int i, j;
//   double temp;
//   for (i = 0; i < N; i++) {
//     temp = 0;
//     for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
//       temp += val[j] * x[0];
//     }
//     y[i] += temp;
//   }
// }
//
// void spmv_csr_col0_fix(int *row_ptr, int *colind, double *val, int N, double
// *x,
//                      double *y) {
//   int i, j;
//   double temp;
//   for (i = 0; i < N; i++) {
//     temp = 0;
//     for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
//       temp += val[j] * x[colind[0]];
//     }
//     y[i] += temp;
//   }
// }
//
//
// void spmv_csr_ai(int *row_ptr, int *colind, double *val, int N, double *x,
//                  double *y) {
//   int i, j;
//   double temp;
//   for (i = 0; i < N; i++) {
//     temp = 0;
//     for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
//       temp += val[j] * val[j];
//     }
//     y[i] += temp;
//   }
// }
