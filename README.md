# Fashion-MNIST Classification 

## Overview
This project focuses on classifying clothing images from the **Fashion-MNIST dataset** using an Artificial Neural Network (ANN) built with PyTorch. The model learns to identify different categories of clothing from grayscale images.

The project explores neural network architecture, model training, evaluation, and regularization techniques to improve generalization.

## Dataset Source
Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist

## Dataset
The project uses the complete Fashion-MNIST dataset.

- **Total images:** 70,000
- **Training images:** 60,000
- **Testing images:** 10,000
- **Image size:** 28 × 28 pixels
- **Image type:** Grayscale
- **Number of classes:** 10

Each image is flattened into a 784-dimensional feature vector before being passed to the ANN.

### Classes
The dataset contains the following categories:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Technologies Used
- Python
- PyTorch
- Matplotlib
- Google Colab
- CUDA GPU acceleration

## Model Architecture
The project uses a fully connected Artificial Neural Network with three hidden layers.

| Layer | Configuration |
|---|---|
| Input layer | 784 features |
| Hidden layer 1 | 128 neurons |
| Hidden layer 2 | 128 neurons |
| Hidden layer 3 | 64 neurons |
| Output layer | 10 neurons |

### Techniques Used
- **ReLU activation:** Introduces non-linearity.
- **Batch Normalization:** Normalizes intermediate activations during training.
- **Dropout (p = 0.3):** Helps reduce overfitting by randomly dropping neurons during training.
- **Cross-Entropy Loss:** Measures classification error.
- **SGD Optimizer:** Updates model parameters during training.
- **L2 Regularization:** Uses `weight_decay = 0.0001` to penalize large weights.
- **GPU training:** Uses CUDA when available to accelerate computation.

## Data Preprocessing
1. Load the Fashion-MNIST training and testing datasets.
2. Convert each 28 × 28 image into a flattened vector of 784 features.
3. Scale pixel values from the range 0–255 to 0–1.
4. Create data loaders for batch-based processing.
5. Train the model using the training data and evaluate it on the appropriate evaluation dataset.

## Training
The model is trained using PyTorch for **100 epochs** with the SGD optimizer and Cross-Entropy Loss. Regularization techniques are included to help control overfitting and improve performance on unseen images.

## Evaluation
The model's performance is evaluated using classification accuracy and loss.

## Conclusion

**Despite implementing regularization techniques such as Dropout, Batch Normalization, and L2 regularization, the model still shows signs of overfitting. The significant gap between training and test accuracy indicates that the model performs better on the training data than on unseen data. Further improvements could include hyperparameter tuning, early stopping, and experimenting with a Convolutional Neural Network (CNN) to improve generalization.**


# Fashion-MNIST ANN Model Optimization Using Optuna

To improve the performance of the baseline ANN model, I implemented hyperparameter optimization using **Optuna** with PyTorch. The objective was to identify the best combination of hyperparameters to improve classification accuracy on the Fashion-MNIST dataset.

### Hyperparameter Tuning
- **Number of Hidden Layers:** 1–5
- **Neurons per Hidden Layer:** 8–128
- **Number of Epochs:** 10–50
- **Learning Rate:** \(10^{-5}\) to \(10^{-1}\)
- **Dropout Rate:** 0.1–0.5
- **Batch Size:** 16, 32, 64, 128
- **Optimizer:** Adam, SGD, RMSprop
- **Weight Decay:** \(10^{-5}\) to \(10^{-3}\)

Optuna automatically evaluated different hyperparameter combinations and identified the configuration that achieved the best performance during optimization.

### Results
The optimized model achieved an accuracy of **90.2%**, demonstrating the importance of hyperparameter tuning in improving ANN performance.

