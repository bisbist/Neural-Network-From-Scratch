from config import *
from utils.data_utils import generate_data, normalize
from utils.plot_utils import plot_loss, plot_confusion_matrix
from model.neural_net import NeuralNetwork
import numpy as np

X_train, X_test, y_train, y_test = generate_data()
X_train, X_test = normalize(X_train, X_test)

nn = NeuralNetwork(layer_sizes=[INPUT_SIZE] + HIDDEN_LAYERS + [OUTPUT_CLASSES], activation=ACTIVATION)
nn.train(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, loss_plot_fn=plot_loss)

train_preds = nn.predict(X_train)
test_preds = nn.predict(X_test)

plot_confusion_matrix(y_test, test_preds, labels=[f"Class {i}" for i in range(OUTPUT_CLASSES)])
print(f"Train Accuracy: {np.mean(train_preds == np.argmax(y_train, axis=1)) * 100:.2f}%")
print(f"Test Accuracy: {np.mean(test_preds == np.argmax(y_test, axis=1)) * 100:.2f}%")

# Predict on new data
new_X = np.random.randn(100, INPUT_SIZE)
new_X_norm = (new_X - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
print("Predictions on new data:", nn.predict(new_X_norm))
