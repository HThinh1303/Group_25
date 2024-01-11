CXX = gcc  # Compiler cho file .cc
NVCC = nvcc  # Compiler cho file .cu

# Thêm các cờ để chỉ định thư mục bao gồm
CXXFLAGS = -Ithird_party/eigen
NVCCFLAGS = -Ithird_party/eigen

# Danh sách các file .cu và .cc bạn muốn biên dịch
CU_FILES = src/layer/conv.cu demo.cu
CC_FILES = src/network.cc src/mnist.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/softmax.cc

# Chuyển danh sách các file thành các file đối tượng
CU_OBJECTS = $(CU_FILES:.cu=.o)
CC_OBJECTS = $(CC_FILES:.cc=.o)

# Thêm các file header vào mảng tương ứng
CU_HEADERS = src/layer/conv.h 
CC_HEADERS = src/network.h src/mnist.h src/layer/fully_connected.h src/layer/max_pooling.h src/layer/relu.h src/layer/softmax.h

# Tên chương trình bạn muốn xây dựng
TARGET = demo

# Quy tắc để biên dịch file .cc
%.o: %.cc $(CC_HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

# Quy tắc để biên dịch file .cu
%.o: %.cu $(CU_HEADERS)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

# Quy tắc để tạo chương trình
$(TARGET): $(CC_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(CC_OBJECTS) $(CU_OBJECTS) -o $(TARGET) $(NVCCFLAGS)

# Quy tắc để xóa các file được tạo
clean:
	rm -f $(CC_OBJECTS) $(CU_OBJECTS) $(TARGET)