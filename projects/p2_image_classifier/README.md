# Intro to Machine Learning -TensorFlow Image Classifier Project

This project aims to build an image classification model using TensorFlow to classify images into 102 types of flowers. The model is further converted into a command-line application for usability and flexibility. The project demonstrates skills in deep learning, TensorFlow, and command-line application development.


### Project Overview
The project consists of two primary phases:
Model Development: Build and train an image classifier using TensorFlow to identify flower species.
Command Line Application: Convert the trained model into a command-line interface (CLI) for easy image classification.

### Features
* Deep Learning: Utilize TensorFlow and Keras for building and training the neural network.
* Command-line Application: Perform predictions on unseen flower images using a simple CLI tool.
* 102 Class Prediction: Classify images into one of 102 flower types using transfer learning.
* Utility Functions: Includes helper functions for loading data, preprocessing images, and managing class mappings for predictions.

## Prerequisites

To run the project effectively, use conda to set up the environment as described below.

### System Requirements

* ***GPU Enabled Environment:*** This project requires a GPU-powered system to train the model efficiently (e.g., use cloud services or appropriate locally-configured GPUs).

⚠️ Make sure to disable GPU when not in use to avoid exhausting your cloud time credits.

## Setting Up the Environment

The required dependencies for this project are specified in an environment.yml file. To set up your project environment, follow these steps:

1. if your system does not contain conda install it from the following [link](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html)
1. Create a new conda environment and install dependencies from the environment.yml file:

   * conda env create -f environment.yml

The environment.yml file includes the following key dependencies:

* tensorflow
* keras
* numpy
* Pillow
* scipy
* etc....

## Data

The dataset used for this project is too large to be stored in the project repository. It contains 102 types of flowers with approximately 20 images per class for training.


## Project Structure
Below is an overview of the project's file structure:

/projects/p2_image_classifier  
├── config.py           # Configuration file for project settings  
├── makeparse.py        # Handles command-line arguments using Click  
├── model_utils.py      # Helper functions for model creation and training  
├── predict.py          # Script for making predictions using the model  
├── utilities.py        # General utility functions for data loading and processing  
├── environment.yml     # Conda environment configuration file  
├── README.md           # Project documentation

### How to Run

Training the Model
Follow these steps to train the TensorFlow image classifier:

Training the Model is done inside of Jupyter Notebooks. After the model is trained download the trained model your computer.

### Predicting

python predict.py --image_path <path_to_image> --checkpoint<model name>

### Further Options

Various configurable options are implemented in the command-line parser to enhance usability. Refer to makeparse.py for argument

Future Work

* Learn to fine-tune models for even better classification accuracy.
* Support exporting predictions in different formats (e.g., JSON or CSV files).
* Add more functionality to the CLI, such as batch predictions.
