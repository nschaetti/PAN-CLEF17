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
    parser.add_argument("--output", type=str, help="Output Pickle file", default="output.p")
    args = parser.parse_args()

    # Result
    result = dict()
    result['training'] = []
    result['labels'] = []

    # Load data set
    final_mapping = dict()
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)

        # Reducer
        reducer = PAN17LetterGramsReducer(
            letters="aàáâãbcçdeéèêfghiíïîjklmnoóôõpqrstuúüûvwxyz",
            punctuations="?.!,;:#$§", add_punctuation=True, add_first_letters=True, add_end_letters=True,
            add_end_grams=True, add_first_grams=True, upper_case=True)

        # Matrix generator
        matrix_generator = PAN17FeaturesMatrixGenerator(letters="aàáâãbcçdeéèêfghiíïîjklmnoóôõpqrstuúüûvwxyz",
                                              punctuations="?.!,;:#$§", upper_case=True)

        # For each author
        for author in data_set['corpus'].get_authors():
            print("Generating feature matrix for author %s" % author.get_name())
            # For each document
            author_mapping = dict()
            for doc in author.get_documents():
                # Maps the document
                doc_mapping = doc.map(reducer)
                # Reduce
                author_mapping = reducer.reduce([author_mapping, doc_mapping])
            # end for

            # Generate author matrix
            author_matrix = matrix_generator.generate_matrix(author_mapping)

            # Add to training
            result['training'] += [author_matrix]

            # Labels
            result['labels'] += [(author.get_property("gender"), author.get_property("language"))]

            #plt.imshow(author_matrix, cmap='gray')
            #plt.show()
        # end for
    # end with

    # Save
    print("Saving file %s" % args.output)
    with open(args.output, 'w') as f:
        pickle.dump(result, f)
    # end with

# end if
