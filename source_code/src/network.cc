#include "./network.h"

void Network::forward(const Matrix& input) {
  if (layers.empty())
    return;
  layers[0]->forward(input);
  for (int i = 1; i < layers.size(); i++) {
    layers[i]->forward(layers[i-1]->output());
  }
}







