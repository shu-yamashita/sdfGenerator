TARGET := a.out
SRCS := $(shell ls *.cu)
OBJS := $(SRCS:.cu=.o)
DEPS := $(SRCS:.cu=.d)

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	nvcc $^ -o $@ 

%.o: %.cu
	nvcc $< -c -o $@ -I./

%.d: %.cu
	nvcc -MM $< -I./ > $@

.PHONY: clean
clean:
	rm -r $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)

