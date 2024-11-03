TARGET := a.out
SRCS := $(shell ls *.cu)
OBJS := $(SRCS:.cu=.o)
DEPS := $(SRCS:.cu=.d)
NVCCFLAGS := -std=c++11 -Xcompiler="-Wall -Wextra" -I../../../

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	nvcc $(NVCCFLAGS) $^ -o $@

%.o: %.cu
	nvcc $(NVCCFLAGS) $< -c -o $@

%.d: %.cu
	nvcc -MM $< $(NVCCFLAGS) > $@

.PHONY: clean
clean:
	rm -r $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)

