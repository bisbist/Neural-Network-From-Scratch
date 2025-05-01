import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Plot the loss curve across training epochs
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

# Display a confusion matrix heatmap for model evaluation
def plot_confusion_matrix(y_true, y_pred, labels):
    # Convert one-hot encoded true labels to class indices
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
    # Plot using seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

