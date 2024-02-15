# AI-programming-with-python-Udacity-Final-Project


## Overview

This repository contains an image classifier project created as part of the AI Programming with Python Nanodegree by Udacity. The project involves building and training a deep learning model to classify images into different categories.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
  

## Introduction

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

## Requirements

The packages you will need are as follows:

import torch
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

import train
import predict
import json


## Usage

## Train the model
python train.py "flowers" --save_dir "train_checkpoint.pth" --arch "vgg16" --learning_rate 0.002 --hidden_units 512  --epochs 2 --gpu

## Make predictions
python predict.py flowers/test/58/image_02663.jpg train_checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu

## Project Structure

- Load the data
- Label mapping
- Visualize a few image
- Building and training the classifier
- Testing the network
- Save the checkpoint
- Loading the checkpoint
- Inference for classification
- Image processing
- Class prediction
- Sanity checking


## License


