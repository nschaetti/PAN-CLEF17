#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
from features.PAN17WordMatrixGenerator import PAN17WordMatrixGenerator


###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 word matrix creator")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--output", type=str, help="Output Pickle file", default="output.p")
    parser.add_argument("--upper", dest="upper", action="store_true", help="Case sensitive")
    parser.set_defaults(upper=False)
    args = parser.parse_args()

    # Result
    result = dict()
    result['words'] = []
    result['labels'] = []

    # Load data set
    final_mapping = dict()
    with open(args.file, 'r') as f:
        # Load
        print("Loading data set file %s" % args.file)
        data_set = pickle.load(f)

        # Matrix generator
        matrix_generator = PAN17WordMatrixGenerator(520, input_scaling=255.0, bow=False)

        # For each document
        print("Add each tokens to the index")
        index = 1
        for doc in data_set['corpus'].get_documents():
            print str(index) + " \r",
            # Add each token
            matrix_generator.add_tokens(doc.get_tokens())
            index += 1
        # end for
        print("")

        # Finalize index
        matrix_generator.finalize_token_index()

        # For each author
        for author in data_set['corpus'].get_authors():
            print("Generating feature matrix for author %s" % author.get_name())

            # For each document
            author_tokens = []
            for doc in author.get_documents():
                # Add tokens
                author_tokens += doc.get_tokens()
            # end for

            # Generate author matrix
            author_matrix = matrix_generator.generate_matrix(author_tokens)

            # Add to training
            result['words'] += [author_matrix]
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
