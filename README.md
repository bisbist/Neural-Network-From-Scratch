# 🧠 Neural Network from Scratch with Advanced Features

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository features a multi-class neural network implemented entirely from scratch using NumPy, without relying on high-level libraries like TensorFlow or PyTorch. Designed for clarity and educational value, this project demonstrates a fully functional deep learning pipeline—covering data preprocessing, network design, training, evaluation, and prediction.

# 🚀 Features

- ## 🔢 Supports Multi-Class Classification

    Built using synthetic classification data with 3 classes and 20 input features.
- ## 🧱 Custom Neural Network Implementation
    * Modular and extensible class-based design
    * Configurable architecture with arbitrary layer sizes
    * Activation: ReLU or Leaky ReLU in hidden layers, Softmax in the output layer
    * He initialization for stable learning in deep nets
- ## 📉 Cross-Entropy Loss

    Implements standard cross-entropy loss tailored for multi-class classification tasks.
- ## 🔁 Mini-Batch Gradient Descent

    Efficient training with mini-batch updates for faster convergence.
- ## 📊 Visualization and Evaluation
    * Live training loss plot
    * Confusion matrix heatmap for model evaluation
    * Test and training accuracy evaluation
- ## Generalization and Inference

    Seamless support for new input predictions using trained weights and stored normalization statistics.

# 📌 Requirements

- Python 3.8+
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Install dependencies via pi:
```
pip install numpy scikit-learn matplotlib seaborn
```
# 🧪 How It Works

## 1. Data Generation and Preprocessing
Generates synthetic data using ```make_classification()``` with 3 classes.
Performs standard normalization using training set statistics.
## 2. Model Architecture
Example configuration:
```NeuralNetwork(layer_sizes=[20, 64, 32, 3], activation='leaky_relu')```
- Input layer: 20 features
- Hidden layers: 64 → 32
- Output layer: 3 neurons with softmax

## 3. Training
- Trains with mini-batch gradient descent
- Epoch-wise loss plotted to monitor convergence
## 4. Evaluation
- Outputs accuracy on both training and test datasets
- Confusion matrix visualizes model performance per class

# 📈 Sample Output

🔹 Training Metrics

- Training Accuracy: ~85–90%

- Test Accuracy: ~77–85%

🔹 Loss Curve: Visualizes the convergence of the network during training
![Fig 1: Loss Curve](./Loss%20Curve.png)

🔹 Confusion Matrix: Evaluates class-wise prediction performance on the test set
![Fig 2: Confusion Matrix](./Confusion%20Matrix.png)

# 🔍 Sample Prediction Output

```
Predicted classes for new data: [2 1 0 1 2 0 1 2 0 1 ...]
```

# 🧩 Future Enhancements

- Add support for Dropout and Batch Normalization
- Integrate momentum/Adam optimizers
- Experiment with different activation functions


🧠 Learn More

- [Understanding Softmax and Cross Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
- [Why Normalize Data Before Training?](https://vivek-murali.medium.com/is-data-normalization-always-necessary-before-training-machine-learning-models-15b10b17e436)
- [He Initialization Explained](https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/)