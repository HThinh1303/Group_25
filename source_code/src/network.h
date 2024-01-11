#ifndef SRC_NETWORK_H_
#define SRC_NETWORK_H_

#include <stdlib.h>
#include <vector>
#include "./layer.h"
#include "./utils.h"

class Network {
 private:
  std::vector<Layer*> layers;  // layer pointers

 public:
  ~Network() {
    for (int i = 0; i < layers.size(); i ++) {
      delete layers[i];
    }
  }

  void add_layer(Layer* layer) { layers.push_back(layer); }
  void forward(const Matrix& input);
  const Matrix& output() { return layers.back()->output(); }
};

#endif  // SRC_NETWORK_H_