import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2./layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def leaky_relu(self, x, alpha=0.01): return np.where(x > 0, x, alpha * x)
    def leaky_relu_derivative(self, x, alpha=0.01): return np.where(x > 0, 1, alpha)
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.activations, self.z_values = [X], []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = self.leaky_relu(z) if self.activation == 'leaky_relu' else self.relu(z)
            self.activations.append(a)
        return self.activations[-1]

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        return np.sum(log_probs) / m

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        gradients = []

        dZ = self.activations[-1] - y
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients.insert(0, (dW, db))

        for i in range(len(self.weights)-1, 0, -1):
            dA = np.dot(dZ, self.weights[i].T)
            dA *= self.leaky_relu_derivative(self.z_values[i-1]) if self.activation == 'leaky_relu' else self.relu_derivative(self.z_values[i-1])
            dW = np.dot(self.activations[i-1].T, dA) / m
            db = np.sum(dA, axis=0, keepdims=True) / m
            gradients.insert(0, (dW, db))
            dZ = dA

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def train(self, X, y, epochs, learning_rate, batch_size, loss_plot_fn=None):
        losses = []
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            loss = self.cross_entropy_loss(self.forward(X), y)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        if loss_plot_fn:
            loss_plot_fn(losses)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
