# Neural Network from First Principles
import numpy as np

# Input dataset: 3 samples, each with 4 features
X = np.array([
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])

class DenseLayer:
    def __init__(self, input_size: int, neuron_count: int) -> None:
        # Random initialization of weights, small values
        self.weights = 0.01 * np.random.randn(input_size, neuron_count)
        self.biases = np.zeros((1, neuron_count))
        # print("Weights:", self.weights)
        # print("Biases:", self.biases)

    def forward_pass(self, data: np.ndarray) -> None:
        # Compute the output of the layer
        self.output = np.dot(data, self.weights) + self.biases

# Building the model: two fully connected layers
first_layer = DenseLayer(input_size=4, neuron_count=5)
second_layer = DenseLayer(input_size=5, neuron_count=3)

# Forward propagation through the network
first_layer.forward_pass(X)
second_layer.forward_pass(first_layer.output)

# Display the results
print("First Layer Output:")
print(first_layer.output)

print("\nSecond Layer Output:")
print(second_layer.output)
