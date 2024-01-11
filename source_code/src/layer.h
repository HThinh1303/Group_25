#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <Eigen/Core>
#include <vector>
#include "./utils.h"

class Layer {
 protected:
  Matrix top;  // layer output
  //   Matrix grad_bottom;  // gradient w.r.t input

 public:
  virtual ~Layer() {}

  virtual void forward(const Matrix& bottom) = 0;
  //  virtual void update(Optimizer& opt) {}
  virtual const Matrix& output() { return top; }
  //   virtual const Matrix& back_gradient() { return grad_bottom; }
  virtual int output_dim() { return -1; }
};

#endif  // SRC_LAYER_H_
