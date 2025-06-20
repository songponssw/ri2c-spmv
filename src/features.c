#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "mmio.h"
#include "utils.h"
#include <omp.h>

struct MyFeature {
  int n;
  int nnz;

  int min_nnz_row;
  int max_nnz_row;
  float avg_nnz_row;
  float nnz_row_sd;
  float nnz_row_var;
  long long int empty_row;

  int num_cols;
  unsigned int ell_elem;
  double ell_ratio;
  int stride;

  int num_diags;
  unsigned int diag_elem;
  double dia_ratio;

  int is_symmetry;
};

void printMyFeature(const struct MyFeature *s) {
  printf("%d, %d,", s->n, s->nnz);
  printf("%d, %d, %f, %f, %f, %lld,", s->min_nnz_row, s->max_nnz_row, s->avg_nnz_row,
         s->nnz_row_sd, s->nnz_row_var, s->empty_row);
  printf("%d, %u, %lf,", s->num_cols, s->ell_elem, s->ell_ratio);
  printf("%d, %u, %lf, %d,", s->num_diags, s->diag_elem, s->dia_ratio, s->stride);
  printf("%d", s->is_symmetry);
  printf("\n");
}

int is_symmetric(int* row, int* col, int nnz) {
    int symmetric = 1; // Sym 

    #pragma omp parallel for shared(symmetric)
    for (int i = 0; i < nnz; i++) {
        if (!symmetric) continue; // quick exit for other threads

        int r = row[i];
        int c = col[i];
        int found = 0;

        // Each thread executes this inner loop independently
        for (int j = 0; j < nnz; j++) {
            if (row[j] == c && col[j] == r) {
                found = 1;
                break;
            }
        }

        if (!found) {
            #pragma omp atomic write
            if (symmetric == 1) {
              symmetric = 0; // Not sym
            }
        }
    }

    return symmetric;
}


///
/// UTILS 
///
long long sum_i(int *arr, int n) {
  int i;
  long long sum = 0;
  for (i = 0; i < n; i++)
    sum += arr[i];
  return sum;
}

float mean_i(int *arr, int n) {
  long long sum = sum_i(arr, n);
  return sum / (float)n;
}

float sd_i(int *arr, int n, float mean_i) {
  float var = 0.0;
  int i;
  for (i = 0; i < n; i++) {
    var += (mean_i - arr[i]) * (mean_i - arr[i]);
  }
  var = var / n;
  return sqrt(var);
}

float var_i(int *arr, int n, float mean_i) {
  float var = 0.0;
  int i;
  for (i = 0; i < n; i++) {
    var += (mean_i - arr[i]) * (mean_i - arr[i]);
  }
  var = var / n;
  return var;
}

//
//
// Main
//
//

int main(int argc, char *argv[]) {
  FILE *f;
  int M, N, nz, inner, inner_max, outer_max;
  int i, *row, *col;
  double *coo_val;

  // Variable for CSR and other formats
  int *row_ptr, *colind;
  double *val;

  /*
   *
   * Read Input matrix
   *
   */
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [mm filename]\n", argv[0]);
    exit(1);
  } else {
    if ((f = fopen(argv[1], "r")) == NULL) {
      fprintf(stderr, "File Not Found\n");
      exit(1);
    }
  }
  get_matrix_market(f, &M, &N, &nz, &row, &col, &coo_val);

  int *nnz_row, max_nnz_row, min_nnz_row;
  long long int empty_row = 0;

  nnz_row = calloc(N, sizeof(int));
  min_nnz_row = N;
  max_nnz_row = 0;

  for (int i = 0; i < nz; i++) {
    nnz_row[row[i]]++;
  }

  // Empty Row
  for (int i=0 ; i<N; i++) {
    if(nnz_row[i] == 0){
      empty_row++;
    }
  }

  for (int i = 0; i < N; i++) {
    if (nnz_row[i] > max_nnz_row)
      max_nnz_row = nnz_row[i];
    if (nnz_row[i] < min_nnz_row)
      min_nnz_row = nnz_row[i];
  }

  int num_cols = 0;
  unsigned int ell_elem = 0;
  double ell_ratio = 0;

  num_cols = max_nnz_row;
  ell_elem = 2 * max_nnz_row * N;
  ell_ratio = (double)ell_elem / nz;

  int num_diags = 0;
  unsigned int diag_elem = 0;
  double dia_ratio = 0;

  int *ind, diag_no;
  ind = (int *)calloc(2 * N - 1, sizeof(int));
  if (ind == NULL) {
    fprintf(stderr, "couldn't allocate ind using calloc\n");
    exit(1);
  }

  // Count num_diags
  for (i = 0; i < nz; i++) {
    if (!ind[N + col[i] - row[i] - 1]++)
      num_diags++;
  }

  // Count diag_elem
  int min = abs(diag_no);
  int stride = 0;
  diag_no = -((2 * N - 1) / 2);
  for (i = 0; i < 2 * N - 1; i++) {
    if (ind[i]) {
      diag_elem += (long long)(N - abs(diag_no));
      if (min > abs(diag_no))
        min = abs(diag_no);  
    }
    diag_no++;
  }
  stride = N - min;


  // Count Symmetry
  // Convert to binary matrix 
	int is_symmetry = 0;
	is_symmetry = is_symmetric(row, col, nz);



  struct MyFeature features;
  features.n = N;
  features.nnz = nz;
  features.avg_nnz_row = (double)sum_i(nnz_row, N) / features.n;
  features.nnz_row_sd = sd_i(nnz_row, N, mean_i(nnz_row, N));
  features.nnz_row_var = var_i(nnz_row, N, mean_i(nnz_row, N));
  features.min_nnz_row = min_nnz_row;
  features.max_nnz_row = max_nnz_row;
  features.empty_row = empty_row;

  features.num_cols = features.max_nnz_row;
  features.ell_elem = features.num_cols * N;
  features.ell_ratio = (double)features.ell_elem / features.nnz;

  features.diag_elem = diag_elem;
  features.num_diags = num_diags;
  features.dia_ratio = (double)features.diag_elem / features.nnz;
  features.stride = stride;
  features.is_symmetry = is_symmetry;
  

  printMyFeature(&features);

  return 0;
}
