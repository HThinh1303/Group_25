#include "./softmax.h"

void Softmax::forward(const Matrix& bottom) {
  // a = exp(z) / \sum{ exp(z) }
  top.array() = (bottom.rowwise() - bottom.colwise().maxCoeff()).array().exp();
  RowVector z_exp_sum = top.colwise().sum();  // \sum{ exp(z) }
  top.array().rowwise() /= z_exp_sum;
}

