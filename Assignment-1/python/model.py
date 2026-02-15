import deep_framework as df

class MyCNN:
    def __init__(self, num_classes):

        # Conv Layer 1: 3 → 16
        self.c1 = df.Conv2D(3, 16, 5, 5, stride=1, padding=0)
        self.r1 = df.ReLU()
        self.p1 = df.MaxPool2D(2, 2)

        # Conv Layer 2: 16 → 32
        self.c2 = df.Conv2D(16, 32, 3, 3, stride=1, padding=0)
        self.r2 = df.ReLU()
        self.p2 = df.MaxPool2D(2, 2)

        # After shape calculation:
        # 32x32
        # → Conv5 → 28x28
        # → Pool → 14x14
        # → Conv3 → 12x12
        # → Pool → 6x6
        # Final = 32 channels × 6 × 6 = 1152

        self.fc1 = df.Linear(1152, num_classes)

    def forward(self, x):

        x = self.c1.forward(x)
        x = self.r1.forward(x)
        x = self.p1.forward(x)

        x = self.c2.forward(x)
        x = self.r2.forward(x)
        x = self.p2.forward(x)

        batch_size = x.shape[0]
        x = df.reshape(x, df.IntVector([batch_size, 1152]))

        x = self.fc1.forward(x)
        return x

    def parameters(self):
        return (
            self.c1.parameters() +
            self.c2.parameters() +
            self.fc1.parameters()
        )

    def save_weights(self, path):
        import pickle
        params = self.parameters()
        weights = [list(p.data) for p in params]
        with open(path, "wb") as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        import pickle
        with open(path, "rb") as f:
            weights = pickle.load(f)

        params = self.parameters()
        for i, p in enumerate(params):
            p.data = df.FloatVector(weights[i])
