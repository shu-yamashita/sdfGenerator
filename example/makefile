TARGET := a.out
SRCS := $(shell ls *.cu)
OBJS := $(SRCS:.cu=.o)
DEPS := $(SRCS:.cu=.d)
NVCCFLAGS := -I../

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	nvcc $^ -o $@ 

%.o: %.cu
	nvcc $< -c -o $@ $(NVCCFLAGS)

%.d: %.cu
	nvcc -MM $< $(NVCCFLAGS) > $@

.PHONY: clean
clean:
	rm -r $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)

