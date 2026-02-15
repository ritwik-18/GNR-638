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
# Compiler
# ==============================
CXX = g++
CXXFLAGS = -O2 -pthread -std=c++17

# ==============================
# Core Source Files
# ==============================
CORE = tensor.cpp ops.cpp autograd.cpp cnn.cpp nn.cpp loss.cpp optim.cpp

# ==============================
# Targets
# ==============================
all: test_gradcheck test_nn test_cnn test_mnist

test_gradcheck: test_gradcheck.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -o test_gradcheck test_gradcheck.cpp $(CORE)

test_nn: test_nn.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -o test_nn test_nn.cpp $(CORE)

test_cnn: test_cnn.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -o test_cnn test_cnn.cpp $(CORE)

test_mnist: test_mnist.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -o test_mnist test_mnist.cpp $(CORE)

clean:
	rm -f test_gradcheck test_nn test_cnn test_mnist *.o
