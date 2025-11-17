# 20 Newsgroups Text Classification with Deep Neural Network (PyTorch)

## Overview

This project implements a deep feed-forward neural network for multi-class text classification on the 20 Newsgroups dataset. It classifies news wire articles into one of 20 topics.

The model uses:
- Basic text preprocessing (lowercasing, removing special characters, tokenization)
- Vocabulary capped at 20,000 tokens with padding/truncation to fixed length
- An embedding layer followed by fully connected layers for classification
- Cross-entropy loss and Adam optimizer

Training, validation, and test performance are reported along with plots of loss and accuracy curves.

## Dataset

The [20 Newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) from scikit-learn is used. Data is split into 70% train, 15% validation, and 15% test.

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- NLTK
- NumPy
- Matplotlib
- tqdm

Install dependencies with:
pip install torch torchvision scikit-learn nltk numpy matplotlib tqdm

Also download NLTK tokenizer data with:
import nltk
nltk.download('punkt')

## Usage

- Run your script with: `python your_script_name.py`
- The model trains for 50 epochs on GPU if available, otherwise CPU
- Training and validation loss and accuracy are printed each epoch
- After training, test accuracy is evaluated and printed
- Loss and accuracy curves for training and validation sets are displayed using matplotlib

## Model Architecture

The `DeepReutersFFN` model consists of:

- An embedding layer that converts token indices to dense vectors
- Two fully connected layers with ReLU activations
- A final linear layer that outputs logits for each of the 20 classes

Input sequences are padded or truncated to length 500. The embedding output is flattened before being fed to fully connected layers.

## Code Structure

- Data loading and preprocessing: tokenization, vocabulary building, encoding, padding
- Dataset and DataLoader classes for batching
- Model definition using PyTorch's `nn.Module`
- Training and validation loops with progress bar
- Testing loop for evaluation on test set
- Plotting training and validation losses and accuracies

## Notes

- Index 0 is reserved for padding, 1 for unknown tokens
- CrossEntropyLoss is used, which combines LogSoftmax and NLLLoss
- Accuracy is the percentage of correctly predicted labels per epoch

## Acknowledgments

This implementation is inspired by common text classification tutorials using PyTorch and sklearn datasets.

---

Feel free to modify parameters such as batch size, learning rate, number of epochs, and model dimensions to optimize performance.
