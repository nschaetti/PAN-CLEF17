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
    parser = argparse.ArgumentParser(description="PAN17 Author Profiling Task software")

    # Argument
    parser.add_argument("--input_dataset", type=str, help="Input data set directory", default="../inputs")
    parser.add_argument("--input_run", type=int, default=1, metavar='R', help="Input run (default:1)")
    parser.add_argument("--output_dir", type=str, help="Input directory", default="../outputs")
    parser.add_argument("--data_server", type=str, help="Data server", default="None")
    parser.add_argument("--token", type=str, default="", metavar='T', help="Token")
    args = parser.parse_args()

    with open(args.output_dir + "/test.txt", 'w') as f:
        f.write(args.input_dataset)
        f.write(args.input_run)
        f.write(args.output_dir)
        f.write(args.data_server)
        f.write(args.token)
    # end with

# end if