# Introduction

This repository concerns a programming challenge of a KTH course called "Machine learning DD2421". The objective is to pratice and create the best classification model. You must build and train a classifier given a labeled train set and then use it to infer the labels of a given unlabeled test set.


# File description

The train set (file `Trainset.csv`) is a csv file made of 1000 samples while the test set (file `Evaluation.csv`) is made of 10000 samples. You are not given any insights about the data and like real data, there are some problems with some of the entries in the training dataset file.

The file `training_functions.py` contains several functions useful for data pre-processing and automated functions for evaluating the different models. The file `fine_tuning.ipynb` is a jupyter notebook used to test and tine-tune different classification algorithms with cross-validation.


# Manual Installation and run

The code was run in Python 3.10

## Installation

You should start with a clean virtual environment and install the requirements for the code to run. You may create a Python 3.10 and install the required packages `requirements.txt`.
 

# Best model

In the end, the best model was obtained with a Random Forest with fine-tuned (cross-validation) hyperparameters.
