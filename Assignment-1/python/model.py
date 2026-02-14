import deep_framework as df
import pickle

class MyCNN(df.Module):
    def __init__(self):
        super().__init__()
        # Architecture Constraints: Conv, Act, Pool, FC [cite: 42-46]
        # Input: 1x32x32
        
        # Layer 1: Conv2D
        # Input 1 channel -> Output 6 channels, 5x5 kernel
        self.c1 = df.Conv2D(1, 6, 5, 5) 
        self.r1 = df.ReLU()
        self.p1 = df.MaxPool2D(2, 2)     
        
        # Calculation for FC layer input:
        # 32x32 input -> Conv(5x5, no pad) -> 28x28
        # 28x28 -> MaxPool(2x2) -> 14x14
        # Flatten size = 6 channels * 14 * 14 = 1176
        self.fc1 = df.Linear(1176, 10)   # 10 Output Classes
        
        self.layers = [self.c1, self.fc1] # Track layers with params

    def forward(self, x):
        x = self.c1.forward(x)
        x = self.r1.forward(x)
        x = self.p1.forward(x)
        x = df.flatten(x)
        x = self.fc1.forward(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

    # Save weights logic for evaluation [cite: 96]
    def save_weights(self, path):
        weights_data = []
        for p in self.parameters():
            # Copy C++ vector data to Python list
            weights_data.append(list(p.data)) 
        
        with open(path, 'wb') as f:
            pickle.dump(weights_data, f)
        print(f"Model weights saved to {path}")

    # Load weights logic for evaluation
    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights_data = pickle.load(f)
        
        params = self.parameters()
        if len(params) != len(weights_data):
            raise ValueError("Mismatch in number of parameters!")
            
        for i, p in enumerate(params):
            # Load Python list back into C++ vector
            p.data = weights_data[i]
        print(f"Model weights loaded from {path}")