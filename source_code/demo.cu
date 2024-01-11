/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/mnist.h"
#include "src/network.h"

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


int main() {
  // data
  MNIST dataset("data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn;
  // Old version
  // Layer* conv1 = new Conv(1, 28, 28, 4, 5, 5, 2, 2, 2);
  // Layer* pool1 = new MaxPooling(4, 14, 14, 2, 2, 2);
  // Layer* conv2 = new Conv(4, 7, 7, 16, 5, 5, 1, 2, 2);
  // Layer* pool2 = new MaxPooling(16, 7, 7, 2, 2, 2);
  // Layer* fc3 = new FullyConnected(pool2->output_dim(), 32);
  // Layer* fc4 = new FullyConnected(32, 10);
  // Layer* relu1 = new ReLU;
  // Layer* relu2 = new ReLU;
  // Layer* relu3 = new ReLU;
  // Layer* softmax = new Softmax;
  // LeNet-5
  Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5);
  Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5);
  Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
  Layer* fc4 = new FullyConnected(120, 84);
  Layer* fc5 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* softmax = new Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(relu4);
  dnn.add_layer(fc5);
  dnn.add_layer(softmax);
  const int n_epoch = 2;
  const int batch_size = 128;
  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    std::cout << "Epoch: " << epoch << std::endl;
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      std::cout << "Idx: " << start_idx << " " <<std::endl;
      //   int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                    std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                    std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
          // if (false && ith_batch % 10 == 1) {
          //   std::cout << ith_batch << "-th grad: " << std::endl;
          //   dnn.check_gradient(x_batch, target_batch, 10);
          // }
      GpuTimer timer; 
      timer.Start();
      dnn.forward(x_batch);
      timer.Stop();
      printf("Time for entire forward-pass: %.3f ms\n", timer.Elapsed());
    }
    // test
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  return 0;
}

