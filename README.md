# Convolutional-Autoencoder-CAE-for-CIFAR-10
This repository contains a PyTorch implementation of a Convolutional Autoencoder (CAE), designed to perform the following tasks:

Autoencoder reconstruction: Train a model to encode and decode images from the CIFAR-10 dataset.

Image colorization: Convert grayscale images to RGB.

Image denoising: Remove Gaussian noise from images.

Features

✅ Builds a convolutional autoencoder (CNN Encoder + Decoder)

✅ Trains the model to reconstruct original colored images

✅ Implements grayscale image colorization (input: grayscale, output: RGB)

✅ Implements image denoising (adding Gaussian noise and training a denoiser)

Project Structure

1️⃣ Data Loading & Preprocessing
2️⃣ CAE Model Definition
3️⃣ Training & Evaluation
4️⃣ Experiment 1: Autoencoder Reconstruction on CIFAR-10
5️⃣ Experiment 2: Grayscale Image Colorization
6️⃣ Experiment 3: Image Denoising

Requirements

Python 3.x

PyTorch

Torchvision

NumPy

Matplotlib

To install dependencies, run:

pip install torch torchvision numpy matplotlib

Dataset

The CIFAR-10 dataset is automatically downloaded via torchvision.datasets.CIFAR10. It consists of 60,000 32×32 color images in 10 classes, with 50,000 training images and 10,000 test images.

Usage

Run the notebook to train and test the model:

jupyter notebook E6_assignment1.ipynb

Results

The autoencoder can effectively reconstruct CIFAR-10 images.

The model can generate reasonable colorized versions of grayscale images.

The denoising autoencoder can remove noise from corrupted images.

Author
JieyuLian

License

This project is licensed under the MIT License.

