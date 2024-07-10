#pragma once

#include "mat.h"
#include <stddef.h>
#include <stdint.h>

enum layer_kind {
  DENSE,
  _INPUT,
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
  Mat2D a;  // layer activations shape(1, ws.cols)
} DenseLayer;

typedef struct {
  const Mat2D *input;
} InputLayer;

typedef struct {
  enum layer_kind kind;
  ActFun act;
  union {
    DenseLayer dl;
    InputLayer il;
  };
} layer_t;

typedef struct {
  size_t layer_count;
  size_t capacity;
  layer_t *layers;
} nn_t;

nn_t new_nn();
void add_dense_layer(nn_t *nn, size_t input_size, size_t output_size, ActFun act);
void nn_forward(nn_t *nn, const Mat2D *input);
void nn_destroy(nn_t *nn);
void nn_init_random(nn_t *nn, const double min, const double max);
void nn_init_zero(nn_t *nn);
nn_t nn_backprop(const nn_t *nn, const Mat2D *y);
Mat2D nn_layer_output(const layer_t *l);
Mat2D nn_output(const nn_t *nn);
