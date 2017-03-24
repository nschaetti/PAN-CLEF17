#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : word_vector_gender_classifier.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import numpy as np
import math
import sys
import tempfile
from six.moves import urllib
from deep_models.PAN17DeepNNModel import PAN17DeepNNModel
from tools.PAN17Metrics import PAN17Metrics
import cPickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 Deep-Learning for gender classification")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--output", type=str, help="Output model Pickle file", default="output.p")
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar='S', help="Random seed (default:1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar='N',
                        help="How many batches to wait before logging traning status")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        print("Loading data set %s" % args.file)
        data_set = pickle.load(f)

        # Sample size
        n_samples = len(data_set['authors'])
        fold_size = int(math.ceil(n_samples / 10.0))

        # Deep-Learning model
        deep_learning_model = PAN17DeepNNModel(classes=data_set['genders'])

        # Train with each document
        print("Training DNN model...")

        # Assess model error rate
        print("Calculating error rate...")

        #error_rate = PAN17Metrics.error_rate(deep_learning_model, docs_token, truths) * 100.0
        #print("Error rate : %f %%" % error_rate)
# end if