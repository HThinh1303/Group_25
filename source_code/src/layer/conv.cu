#include "conv.h"
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHANNEL_IN 1
#define CHANNEL_OUT 6
#define HEIGHT_KERNEL 5
#define WIDTH_KERNEL 5
__constant__ float d_weight[CHANNEL_IN * CHANNEL_OUT * HEIGHT_KERNEL * WIDTH_KERNEL];

#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(-1); \
    }

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

void Conv::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  //std::cout << weight.colwise().sum() << std::endl;
  //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::im2col(const Vector& image, Matrix& data_col) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // im2col
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c ++) {
    Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          data_col(i, c * hw_kernel + j) = 0;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
        }
      }
    }
  }
}
//Normal convolution layer
__global__ void conv_kernel(const float* input, const float* weight, const float* bias, float* output, int height_in, int width_in, int height_out, int width_out, int channel_in, int height_kernel, int width_kernel, int stride, int pad_h, int pad_w) {
    int hw_kernel = height_kernel * width_kernel;  // Kích thước kernel
    int hw_out = height_out * width_out;  // Kích thước đầu ra

    int index = blockIdx.x * blockDim.x + threadIdx.x;  // Vị trí hiện tại trên GPU

    int c = index / hw_out;  // Kênh đầu vào
    int i = index % hw_out;  // Vị trí trong đầu ra

    int step_h = i / width_out;
    int step_w = i % width_out;
    int start_idx = step_h * width_in * stride + step_w * stride;  // Vị trí bắt đầu cửa sổ conv

    if (c < channel_in && i < hw_out) {
        for (int j = 0; j < hw_kernel; j++) {
            int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // Cột sau khi padding
            int cur_row = start_idx / width_in + j / width_kernel - pad_h;

            if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) {
                output[index] += 0;
            } else {
                int pick_idx = cur_row * width_in + cur_col;
                output[index] += input[c * height_in * width_in + pick_idx] * weight[c * hw_kernel + j];
            }
        }

        output[index] += bias[c];
    }
}
const int blockSize = 256;

void Conv::conv_d(const Matrix& input, Matrix& output) {
    int height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    int width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    int dim_out = height_out * width_out * channel_out;

    Matrix result(dim_out, 1);

    float* d_input;  // Đầu vào trên GPU
    float* d_weight;  // Trọng số trên GPU
    float* d_bias;  // Bias trên GPU
    float* d_output;  // Kết quả trên GPU

    // Cấp phát bộ nhớ trên GPU
    cudaMalloc((void**)&d_input, input.size() * sizeof(float));
    cudaMalloc((void**)&d_weight, weight.size() * sizeof(float));
    cudaMalloc((void**)&d_bias, bias.size() * sizeof(float));
    cudaMalloc((void**)&d_output, result.size() * sizeof(float));

    // Sao chép dữ liệu từ CPU sang GPU
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Số lượng block và số lượng thread trên mỗi block
    int numBlocks = (dim_out + blockSize - 1) / blockSize;
    int threadsPerBlock = blockSize;

    // Thực hiện phép tính convolution trên GPU
    conv_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output,height_in, width_in, height_out, width_out, channel_in, height_kernel, width_kernel, stride, pad_h, pad_w);

    // Sao chép kết quả từ GPU sang CPU
    cudaMemcpy(result.data(), d_output, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Giải phóng bộ nhớ trên GPU
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    output = Eigen::Map<Vector>(result.data(), result.size());
}

__global__ void conv_kernel1(const float* input, const float* weight, const float* bias,
                            float* output, int height_in, int width_in, int height_kernel, int width_kernel,
                            int pad_h, int pad_w, int stride, int channel_in, int channel_out,
                            int height_out, int width_out) {
    extern __shared__ float shared_mem[];

    int tile_size = blockDim.x + width_kernel - 1;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    int output_row = blockIdx.y * blockDim.y + tile_row;
    int output_col = blockIdx.x * blockDim.x + tile_col;

    int channel_out_idx = blockIdx.z;

    for (int c = 0; c < channel_in; c++) {
        int input_row = output_row * stride - pad_h;
        int input_col = output_col * stride - pad_w;

        if (input_row >= 0 && input_row < height_in && input_col >= 0 && input_col < width_in) {
            shared_mem[tile_row * tile_size + tile_col] = input[c * height_in * width_in + input_row * width_in + input_col];
        } else {
            shared_mem[tile_row * tile_size + tile_col] = 0.0f;
        }
        __syncthreads();

        float result = 0.0f;

        if (tile_row < height_kernel && tile_col < width_kernel && output_row < height_out && output_col < width_out) {
            for (int i = 0; i < height_kernel; i++) {
                for (int j = 0; j < width_kernel; j++) {
                    result += shared_mem[(tile_row + i) * tile_size + (tile_col + j)] * weight[(c * channel_out + channel_out_idx) * height_kernel * width_kernel + i * width_kernel + j];
                }
            }
        }
        __syncthreads();

        if (output_row < height_out && output_col < width_out) {
            output[(channel_out_idx * height_out * width_out + output_row * width_out + output_col)] = result;
        }
    }

    if (output_row < height_out && output_col < width_out) {
        output[(channel_out_idx * height_out * width_out + output_row * width_out + output_col)] += bias[channel_out_idx];
    }
}

void Conv::conv_d1(const Matrix& input, Matrix& output) {
    int height_in = input.rows() / (width_in * channel_in);
    int height_out = (height_in + 2 * pad_h - height_kernel) / stride + 1;

    Matrix result(height_out * width_out * channel_out, 1);

    float* d_input, * d_weight, * d_bias, * d_output;

    cudaMalloc((void**)&d_input, sizeof(float) * input.size());
    cudaMalloc((void**)&d_weight, sizeof(float) * weight.size());
    cudaMalloc((void**)&d_bias, sizeof(float) * bias.size());
    cudaMalloc((void**)&d_output, sizeof(float) * result.size());

    cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight.data(), sizeof(float) * weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((width_out + block_dim.x - 1) / block_dim.x, (height_out + block_dim.y - 1) / block_dim.y, channel_out);

    int shared_mem_size = (block_dim.y + height_kernel - 1) * (block_dim.x + width_kernel - 1) * sizeof(float);

    conv_kernel1<<<grid_dim, block_dim, shared_mem_size>>>(d_input, d_weight, d_bias, d_output, height_in, width_in,
                                                            height_kernel, width_kernel, pad_h, pad_w, stride,
                                                            channel_in, channel_out, height_out, width_out);

    cudaMemcpy(result.data(), d_output, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    output = Eigen::Map<Vector>(result.data(), result.size());
}

__global__ void conv_kernel2(const float* input, const float* bias,
                            float* output, int height_in, int width_in, int height_kernel, int width_kernel,
                            int pad_h, int pad_w, int stride, int channel_in, int channel_out,
                            int height_out, int width_out) {
    extern __shared__ float shared_mem[];

    int tile_size = blockDim.x + width_kernel - 1;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    int output_row = blockIdx.y * blockDim.y + tile_row;
    int output_col = blockIdx.x * blockDim.x + tile_col;

    int channel_out_idx = blockIdx.z;

    for (int c = 0; c < channel_in; c++) {
        int input_row = output_row * stride - pad_h;
        int input_col = output_col * stride - pad_w;

        if (input_row >= 0 && input_row < height_in && input_col >= 0 && input_col < width_in) {
            shared_mem[tile_row * tile_size + tile_col] = input[c * height_in * width_in + input_row * width_in + input_col];
        } else {
            shared_mem[tile_row * tile_size + tile_col] = 0.0f;
        }
        __syncthreads();

        float result = 0.0f;

        if (tile_row < height_kernel && tile_col < width_kernel && output_row < height_out && output_col < width_out) {
            for (int i = 0; i < height_kernel; i++) {
                for (int j = 0; j < width_kernel; j++) {
                    result += shared_mem[(tile_row + i) * tile_size + (tile_col + j)] * d_weight[(c * channel_out + channel_out_idx) * height_kernel * width_kernel + i * width_kernel + j];
                }
            }
        }
        __syncthreads();

        if (output_row < height_out && output_col < width_out) {
            output[(channel_out_idx * height_out * width_out + output_row * width_out + output_col)] = result;
        }
    }

    if (output_row < height_out && output_col < width_out) {
        output[(channel_out_idx * height_out * width_out + output_row * width_out + output_col)] += bias[channel_out_idx];
    }
}

void Conv::conv_d2(const Matrix& input, Matrix& output) {
    int height_in = input.rows() / (width_in * channel_in);
    int height_out = (height_in + 2 * pad_h - height_kernel) / stride + 1;

    Matrix result(height_out * width_out * channel_out, 1);

    float* d_input, * d_bias, * d_output;

    cudaMalloc((void**)&d_input, sizeof(float) * input.size());
    cudaMalloc((void**)&d_bias, sizeof(float) * bias.size());
    cudaMalloc((void**)&d_output, sizeof(float) * result.size());

    cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((width_out + block_dim.x - 1) / block_dim.x, (height_out + block_dim.y - 1) / block_dim.y, channel_out);

    int shared_mem_size = (block_dim.y + height_kernel - 1) * (block_dim.x + width_kernel - 1) * sizeof(float);

    conv_kernel2<<<grid_dim, block_dim, shared_mem_size>>>(d_input, d_bias, d_output, height_in, width_in,
                                                            height_kernel, width_kernel, pad_h, pad_w, stride,
                                                            channel_in, channel_out, height_out, width_out);

    cudaMemcpy(result.data(), d_output, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);

    output = Eigen::Map<Vector>(result.data(), result.size());
}

void Conv::conv_h(const Matrix& input, Matrix& output)
{
    Matrix data_col;
    im2col(input, data_col);
    Matrix result = data_col * weight;
    result.rowwise() += bias.transpose();
    
    output = Eigen::Map<Vector>(result.data(), result.size());
}

//Normal convolution layer
__global__ void im2col_kernel(const float* image, float* data_col, int height_in, 
int width_in, int height_kernel, int width_kernel, int height_out, int width_out, int channel_in, int stride, int pad_h, int pad_w) {
    // Calculate the current thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the number of threads per block
    int block_size = blockDim.x * gridDim.x;

    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;

    // Iterate over the data_col matrix
    for (int i = idx; i < hw_out * channel_in; i += block_size) {
        int c = i / hw_out;
        int j = i % hw_out;

        int step_h = j / width_out;
        int step_w = j % width_out;
        int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window

        for (int k = 0; k < hw_kernel; k++) {
            int cur_col = start_idx % width_in + k % width_kernel - pad_w;  // col after padding
            int cur_row = start_idx / width_in + k / width_kernel - pad_h;

            if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) {
                data_col[i * hw_kernel + k] = 0;
            }
            else {
                int pick_idx = cur_row * width_in + cur_col;
                data_col[i * hw_kernel + k] = image[c * hw_in + pick_idx];  // pick which pixel
            }
        }
    }
}


void Conv::conv_d3(const Matrix& input, Matrix& output)
{
    // Khai báo và khởi tạo dữ liệu trên GPU
    float* d_image, *d_data_col;
    float* imageData = (float *)(input).transpose().data();

    cudaMalloc((void**)&d_image, height_in * width_in * sizeof(float));
    cudaMalloc((void**)&d_data_col, height_in * width_in * channel_out * sizeof(float));
    cudaMemcpy(d_image, imageData, height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(512);
    dim3 gridDim(1);

    float* dataCol = (float *)malloc(height_kernel * width_kernel * channel_in * height_out * width_out * sizeof(float));

    
    im2col_kernel<<<gridDim, blockDim>>>(d_image, d_data_col, height_in, width_in,
                                                    height_kernel, width_kernel, height_out, width_out,
                                                    channel_in, stride, pad_h, pad_w);
    cudaMemcpy(dataCol, d_data_col, height_in * width_in * channel_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_data_col);

    Matrix data_col = Eigen::Map<Matrix>(dataCol, height_out * width_out, height_kernel * width_kernel * channel_in);
    Matrix result = data_col * weight;
    result.rowwise() += bias.transpose();
    
    output = Eigen::Map<Vector>(result.data(), result.size());
}



void Conv::forward(const Matrix& bottom) {
  GpuTimer timer;
  timer.Start(); 
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i ++) {
    // im2col
    Matrix result;
    // Host: Sequential
    // conv_h(bottom.col(i), result);

    // Device : Solution 1: Thực hiện song song hóa hàm im2col để sinh ra ma trận data_col (Ma trận chưa được tích chập trọng số và bias)
    // conv_d3(bottom.col(i), result);

    // Device 3: Solution 2: Thực hiện tích chập giúp đưa ra output đã tích chập với trọng số và bias (Song song)
    conv_d(bottom.col(i), result);

    // Device 1: Tiled shared memory convolution
    // conv_d1(bottom.col(i), result);
    
    // Device 2: Weight matrix (kernel values) in constant memory
    // conv_d2(bottom.col(i), result);
    top.col(i) = result;
  }
  timer.Stop();
  printf("Time Convolution: %.3f ms\n", timer.Elapsed());
}
