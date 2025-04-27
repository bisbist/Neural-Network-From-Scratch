inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

# Calculate the weighted sum
weighted_sum = sum(i * w for i, w in zip(inputs, weights)) + bias
print("Weighted sum:", weighted_sum)