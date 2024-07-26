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

void add_scalar_Mat2D(Mat2D *m, const double s);
void sum_Mat2D(Mat2D *m1, const Mat2D *m2);
void print_Mat2D(const Mat2D *m, const char *end);
Mat2D transpose_Mat2D(const Mat2D *m);

void add_column_scalar(Mat2D *col, const double s);
void Mat2D_col_mul(const Mat2D *mat, const Mat2D *vec, Mat2D *out);
void vec_Mat2D_mul(const Mat2D *vec, const Mat2D *mat, Mat2D *out);

void convolution2D(const Mat2D *input, const Mat2D *kernel, int stride, int padding, Mat2D *out);
void max_pooling2D(const Mat2D *input, Mat2D *out, size_t pool_size);
void avg_pooling2D(const Mat2D *input, Mat2D *out, size_t pool_size);
