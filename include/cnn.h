#pragma once

#include "mat.h"
#include <stddef.h>
#include <stdint.h>

enum layer_kind {
  DENSE,
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
  Mat2D a;  // same size as weight coulmns
  ActFun act;
} DenseLayer;

typedef struct {
  enum layer_kind kind;
  union {
    DenseLayer dl;
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
