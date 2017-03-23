#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import cPickle as pickle

from reducer.PAN17LetterGramsReducer import PAN17LetterGramsReducer

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
    final_mapping = dict()
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)

        # Reducer
        #letters="AaÀàÁáÂâÃãBbCcÇçDdEeÉéÈèÊêFfGgHhIiÍíÏïÎîJjKkLlMmNnOoÓóÔôÕõPpQqRrSsTtUuÚúÜüÛûVvWwXxYyZz",
        reducer = PAN17LetterGramsReducer(
            letters="aàáâãbcçdeéèêfghiíïîjklmnoóôõpqrstuúüûvwxyz",
            punctuations="?.!,;:#$§", add_punctuation=True, add_first_letters=True, add_end_letters=True,
            add_end_grams=True, add_first_grams=True, upper_case=True)

        # For each author
        for author in data_set['corpus'].get_authors():
            # For each document
            author_mapping = dict()
            for doc in author.get_documents():
                # Maps the document
                doc_mapping = doc.map(reducer)
                # Reduce
                author_mapping = reducer.reduce([author_mapping, doc_mapping])
            # end for
            print(author_mapping)
            exit()
        # end for
    # end with
# end if
