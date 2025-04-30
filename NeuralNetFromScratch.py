import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate multi-class data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)
y_onehot = np.eye(3)[y]  # One-hot encoding

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Normalize
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # The initialization for ReLU
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2./layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer: Softmax
                a = self.softmax(z)
            else:
                # Hidden layers: ReLU/LeakyReLU
                if self.activation == 'leaky_relu':
                    a = self.leaky_relu(z)
                else:
                    a = self.relu(z)
            
            self.activations.append(a)
        return self.activations[-1]
    
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        return np.sum(log_probs) / m
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        gradients = []
        
        # Output layer gradient
        dZ = self.activations[-1] - y
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients.insert(0, (dW, db))
        
        # Hidden layers
        for i in range(len(self.weights)-1, 0, -1):
            if self.activation == 'leaky_relu':
                dA = np.dot(dZ, self.weights[i].T) * self.leaky_relu_derivative(self.z_values[i-1])
            else:
                dA = np.dot(dZ, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
            
            dW = np.dot(self.activations[i-1].T, dA) / m
            db = np.sum(dA, axis=0, keepdims=True) / m
            gradients.insert(0, (dW, db))
            dZ = dA
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]
    
    def train(self, X, y, epochs, learning_rate, batch_size=32):
        losses = []
        for epoch in range(epochs):
            # Mini-batch
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            # Compute loss
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y_pred, y)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        # Plot loss
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training Loss')
        plt.show()
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Initialize network (input: 20, hidden: 64 -> 32, output: 3)
nn = NeuralNetwork(layer_sizes=[20, 64, 32, 3], activation='leaky_relu')
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=500)

# Evaluate
train_preds = nn.predict(X_train)
test_preds = nn.predict(X_test)
print(f"Train Accuracy: {np.mean(train_preds == np.argmax(y_train, axis=1)) * 100:.2f}%")
print(f"Test Accuracy: {np.mean(test_preds == np.argmax(y_test, axis=1)) * 100:.2f}%")