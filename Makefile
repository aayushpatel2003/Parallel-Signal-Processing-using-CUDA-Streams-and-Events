# Makefile for building the CUDA Streams project

TARGET = streams.exe
NVCC = nvcc
NVCC_FLAGS = -I./ --std=c++17

SRC = streams.cu
HDR = streams.h

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) output.txt