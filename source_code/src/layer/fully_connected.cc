#include "./fully_connected.h"

void FullyConnected::init() {
  weight.resize(dim_in, dim_out);
  bias.resize(dim_out);
  grad_weight.resize(dim_in, dim_out);
  grad_bias.resize(dim_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

void FullyConnected::forward(const Matrix& bottom) {
  // z = w' * x + b
  const int n_sample = bottom.cols();
  top.resize(dim_out, n_sample);
  top = weight.transpose() * bottom;
  top.colwise() += bias;
}



