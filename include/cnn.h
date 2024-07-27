#pragma once

#include "mat.h"
#include <stddef.h>
#include <stdint.h>

enum layer_kind {
  _INPUT,
  DENSE,
  CONV2D,
  MAX_POOL,
  AVG_POOL,
  FLATTEN,
};

typedef enum {
  SIGMOID,
  SOFTMAX,
  RELU,
  TANH,
} ActFun;

typedef struct {
  Mat2D ws;
  double bias;
  Mat2D a;
} DenseLayer;

typedef struct {
  const Mat2D *input;
  size_t height;
  size_t width;
  size_t channels;
} InputLayer;

typedef struct {
  size_t kernel_count;
  size_t channels;
  int padding;
  int stride;
  Mat2D *kernels;
  Mat2D *a;
} Conv2dLayer;

typedef struct {
  Mat2D *a;
  size_t channels;
  size_t pool_size;
} PoolingLayer;

typedef struct {
  Mat2D a;
} FlattenLayer;

typedef struct {
  enum layer_kind kind;
  ActFun act;
  union {
    DenseLayer dl;
    InputLayer il;
    Conv2dLayer cl;
    PoolingLayer pl;
    FlattenLayer fl;
  };
} layer_t;

typedef struct {
  size_t layer_count;
  size_t capacity;
  layer_t *layers;
} nn_t;

nn_t new_nn(size_t height, size_t width, size_t channels);
void nn_add_dense_layer(nn_t *nn, size_t input_size, size_t output_size, ActFun act);
void nn_forward(nn_t *nn, const Mat2D *input);
void nn_destroy(nn_t *nn);
void nn_init_random(nn_t *nn, const double min, const double max);
void nn_init_zero(nn_t *nn);
nn_t nn_backprop(const nn_t *nn, const Mat2D *y);
Mat2D nn_layer_output(const layer_t *l);
Mat2D nn_output(const nn_t *nn);
void nn_fit(nn_t *nn, const Mat2D *train_data, const Mat2D *labels, size_t batch_size, double lr);
