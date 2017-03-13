#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import pySpeeches as ps
import cPickle as pickle

###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 data set creator")

    # Argument
    parser.add_argument("--file", type=str, help="Output Pickle file", default="pan17clef.p")
    args = parser.parse_args()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)
        print(data_set.keys())
        print("Languages : " + str(data_set['languages']))
        print("Genders : " + str(data_set['genders']))

        # Get authors
        author = data_set['corpus'].get_authors()[0]

        # For each document of the first author
        print("Author %s is a %s from %s with %i document(s)" % (author.get_name(), author.get_property("gender"),
                                                                 author.get_property("language"), author.get_size()))
        for doc in author.get_documents():
            print("Document %s" % doc.get_title())
            print(doc.get_tokens())
        # end for

# end if
