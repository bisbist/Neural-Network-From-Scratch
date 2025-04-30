# Neural Network from Scratch with Advanced Features

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **from-scratch implementation** of a deep neural network with:
- Customizable architecture (multiple hidden layers)
- ReLU/LeakyReLU activations
- Softmax output with cross-entropy loss
- Mini-batch gradient descent

## Key Features
| Feature | Implementation Detail | Research Relevance |
|---------|-----------------------|--------------------|
| **Multi-layer Architecture** | Configurable `layer_sizes` (e.g., [20,64,32,3]) | Shows understanding of deep learning fundamentals |
| **Advanced Activations** | ReLU/LeakyReLU with He initialization | Addresses vanishing gradient problem |
| **Proper Loss Function** | Cross-entropy loss + Softmax output | Correct handling of multi-class classification |
| **Efficient Training** | Mini-batch gradient descent | Scalability awareness |

## Results
Achieved **89% test accuracy** on synthetic 3-class data: