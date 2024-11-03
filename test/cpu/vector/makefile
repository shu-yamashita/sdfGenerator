TARGET := a.out
SRCS := $(shell ls *.cpp)
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)
CXXFLAGS := -std=c++11 -Wall -Wextra -I../../../

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	g++ $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	g++ $(CXXFLAGS) $< -c -o $@

%.d: %.cpp
	g++ -MM $< $(CXXFLAGS) > $@

.PHONY: clean
clean:
	rm -r $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)

