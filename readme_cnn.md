# CIFAR-10 CNN Training (Keras 3)

This notebook demonstrates how to build, train, and evaluate a compact convolutional neural network (CNN) on the CIFAR-10 image dataset using Keras 3 with TensorFlow as backend.

The workflow includes:
- Loading and normalizing the CIFAR-10 dataset  
- Splitting into training, validation, and test sets  
- A small but solid CNN with data augmentation and regularization  
- Training with early stopping and learning-rate scheduling  
- Evaluation on the held-out test set

---

### ▶️ Run in Google Colab

Click the badge below to open the notebook directly in Google Colab.  
Once it opens, select **Runtime → Change runtime type → GPU (T4)** for faster training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maciejjkowara/keras_training/blob/main/cifar10_cnn_three_cells.ipynb)

---

### Requirements

The notebook installs or relies on:

```bash
keras >= 3.0
tensorflow >= 2.15
scikit-learn
numpy
