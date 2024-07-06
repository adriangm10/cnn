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
  double *a;  // same size as weight coulmns
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
  layer_t *layers;
} nn_t;
