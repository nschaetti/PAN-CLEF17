#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import pySpeeches as ps
from PAN17TruthLoader import PAN17TruthLoader
from PAN17LetterGramsReducer import PAN17LetterGramsReducer
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

        # First author, first document
        doc = data_set['corpus'].get_authors()[0].get_documents()[4]
        print(doc.get_tokens())
        # Reducer
        reducer = PAN17LetterGramsReducer(
            letters="AaÀàÁáÂâÃãBbCcÇçDdEeÉéÈèÊêFfGgHhIiÍíÏïÎîJjKkLlMmNnOoÓóÔôÕõPpQqRrSsTtUuÚúÜüÛûVvWwXxYyZz",
            punctuations="?.!,;:#$§", add_punctuation=True, add_first_letters=True)

        # Maps the document
        print(doc.map(reducer))
# end if
