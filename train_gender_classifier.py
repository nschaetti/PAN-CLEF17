#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import matplotlib.pyplot as plt
import pySpeeches as ps
from PAN17TruthLoader import PAN17TruthLoader
from PAN17LetterGramsReducer import PAN17LetterGramsReducer
from PAN17FeaturesMatrixGenerator import PAN17FeaturesMatrixGenerator
import cPickle as pickle

###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 data set creator")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    args = parser.parse_args()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)

        index = 0
        for matrix in data_set['2grams']:
            print("Gender : " + data_set['labels'][index][0])
            print("Language : " + data_set['labels'][index][1])
            plt.imshow(matrix, cmap="gray")
            plt.show()
            index += 1
        # end for
    # end with

# end if