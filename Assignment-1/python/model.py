import deep_framework as df
import pickle


class MyCNN:
    def __init__(self, num_classes):

        # -------- Block 1 --------
        self.c1 = df.Conv2D(1, 8, 3, 3)
        self.r1 = df.ReLU()

        self.c2 = df.Conv2D(8, 8, 3, 3)
        self.r2 = df.ReLU()

        self.p1 = df.MaxPool2D(2, 2)

        # -------- Block 2 --------
        self.c3 = df.Conv2D(8, 16, 3, 3)
        self.r3 = df.ReLU()

        self.p2 = df.MaxPool2D(2, 2)

        # Final size:
        # 32 → 30 → 28 → pool → 14
        # 14 → 12 → pool → 6
        # 16 * 6 * 6 = 576

        self.fc1 = df.Linear(576, 64)
        self.r4 = df.ReLU()

        self.fc2 = df.Linear(64, num_classes)

        self.layers = [
            self.c1, self.c2, self.c3,
            self.fc1, self.fc2
        ]


    def forward(self, x):

        # Block 1
        x = self.c1.forward(x)
        x = self.r1.forward(x)

        x = self.c2.forward(x)
        x = self.r2.forward(x)

        x = self.p1.forward(x)

        # Block 2
        x = self.c3.forward(x)
        x = self.r3.forward(x)

        x = self.p2.forward(x)

        # Flatten
        batch_size = x.shape[0]
        x = df.reshape(x, df.IntVector([batch_size, 576]))

        # FC
        x = self.fc1.forward(x)
        x = self.r4.forward(x)
        x = self.fc2.forward(x)

        return x


    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params


    def save_weights(self, path):
        weights_data = []
        for p in self.parameters():
            weights_data.append(list(p.data))

        with open(path, 'wb') as f:
            pickle.dump(weights_data, f)

        print(f"Model weights saved to {path}")


    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights_data = pickle.load(f)

        params = self.parameters()

        if len(params) != len(weights_data):
            raise ValueError("Mismatch in number of parameters!")

        for i, p in enumerate(params):
            p.data = df.FloatVector(weights_data[i])

        print(f"Model weights loaded from {path}")
