#include "./relu.h"

void ReLU::forward(const Matrix& bottom) {
  // a = z*(z>0)
  top = bottom.cwiseMax(0.0);
}

