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
from deep_models.PAN17ConvNet import PAN17ConvNet
from deep_models.PAN17DeepNNModel import PAN17DeepNNModel
import cPickle as pickle
import torch


###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 Deep-Learning Words/BOW for language classification")

    # Argument
    parser.add_argument("--epoch", type=int, default=10, metavar='E', help="Number of Epoch (default:10)")
    parser.add_argument("--batch-size", type=int, default=64, metavar='B', help="Input batch size (default:64)")
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar='S', help="Random seed (default:1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar='N',
                        help="How many batches to wait before logging training status (default: 10)")
    parser.add_argument("--index", type=int, default=-1, metavar='I', help="Test set index (default:-1)")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        print("Loading data set %s" % args.file)
        data_set = pickle.load(f)
        print("Data set loaded...")
        # Sample size
        n_samples = len(data_set['words'])
        fold_size = int(math.ceil(n_samples / 10.0))

        # Get truths
        truths = []
        for truth in data_set['labels']:
            truths += [truth[1]]
        # end for

        # Deep-Learning model
        deep_learning_model = PAN17DeepNNModel(PAN17ConvNet(n_classes=5), classes=('great britain', 'canada',
                                                                                   'ireland', 'new zealand',
                                                                                   'australia'), cuda=args.cuda,
                                               lr=args.lr, momentum=args.momentum, log_interval=args.log_interval,
                                               seed=args.seed)

        # K-10 fold
        grams_set = np.array(data_set['words'])
        truths_set = np.array(truths)
        grams_set.shape = (10, fold_size)
        truths_set.shape = (10, fold_size)

        # Select training and test sets
        test = grams_set[-1]
        test_truths = truths_set[-1]
        training = np.delete(grams_set, -1, axis=0)
        training_truths = np.delete(truths_set, -1, axis=0)
        training.shape = (fold_size * 9)
        training_truths.shape = (fold_size * 9)

        # Data set
        print("Data set to Torch Tensors...")
        tr_data_set = deep_learning_model.to_torch_data_set(training.tolist(), training_truths)
        te_data_set = deep_learning_model.to_torch_data_set(test.tolist(), test_truths)

        # Train with each document
        print("Testing model...")
        for epoch in range(1, args.epoch+1):
            deep_learning_model.train(epoch, tr_data_set, batch_size=args.batch_size)
            deep_learning_model.test(epoch, te_data_set, batch_size=args.batch_size)
        # end for
# end if