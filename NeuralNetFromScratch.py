# Define the weights and bias
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],  # weights1
    [0.5, -0.91, 0.26, -0.5],  # weights2
    [-0.26, -0.27, 0.17, 0.87],  # weights3
]
biases = [2, 3, 0.5]  # bias1, bias2, bias3

# Calculate the weighted sum
weighted_sum = [
    sum(i * w for i, w in zip(inputs, weight)) + bias
    for weight, bias in zip(weights, biases)
]
print("Weighted sum:", weighted_sum)
