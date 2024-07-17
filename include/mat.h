#pragma once

#include <stddef.h>

#define MAT2D_GET(mat, i, j) mat.elems[(i) * mat.cols + (j)]

typedef struct mat {
  size_t *dims;
  size_t dim_count;
  double *elems;
} mat_t;

typedef struct {
  size_t cols;
  size_t rows;
  double *elems;
} Mat2D;

void mul_Mat2D(const Mat2D *m1, const Mat2D *m2, Mat2D *out);
void destroy_Mat2D(Mat2D *m);
Mat2D new_Mat2D(const size_t rows, const size_t cols);
void random_init_Mat2D(Mat2D *m, const double min, const double max);
void zero_init_Mat2D(Mat2D *m);

// m += s
void add_scalar_Mat2D(Mat2D *m, const double s);
void sum_Mat2D(Mat2D *m1, const Mat2D *m2);
void print_Mat2D(const Mat2D *m, const char *end);
