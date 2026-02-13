# ==============================
# Compiler
# ==============================
CXX = g++
CXXFLAGS = -O2 -pthread -std=c++17

# ==============================
# Files
# ==============================
SRC = test_mnist.cpp tensor.cpp ops.cpp autograd.cpp cnn.cpp nn.cpp loss.cpp optim.cpp
OBJ = $(SRC:.cpp=.o)

TARGET = mnist

# ==============================
# Build
# ==============================
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ==============================
# Run
# ==============================
run: $(TARGET)
	./$(TARGET)

# ==============================
# Clean
# ==============================
clean:
	rm -f *.o $(TARGET)
