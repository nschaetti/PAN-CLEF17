#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import numpy as np
from language_models.PAN17LanguageModel import PAN17LanguageModel
from language_models.PAN17JelinekMercerSmoothing import PAN17JelinekMercerSmoothing
from tools.PAN17Metrics import PAN17Metrics
import cPickle as pickle

# ALPHABETS
ENGLISH_PUNCTUATIONS = u"?.!,;:#$§"
SPANISH_PUNCTUATIONS = u"?¿.!¡,;:#$§"
PORTUGUESE_PUNCTUATIONS = u"?.!,;:#$§"
ARABIC_PUNCTUATIONS = u":?؟‎.!,;،؍"


###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 RNN model")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--output", type=str, help="Output model Pickle file", default="output.p")
    args = parser.parse_args()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)

        # Probabilistic model
        language_model = PAN17LanguageModel(classes=data_set['genders'], upper=False)
        language_model.set_smoothing(PAN17JelinekMercerSmoothing(l=0.1))

        # Initialize model
        print("Initializing token counts...")
        for doc in data_set['corpus'].get_documents():
            for token in doc.get_tokens():
                language_model.init_token_count(token)
            # end for
        # end for

        # For each gender
        for gender in data_set['genders']:
            print("Calculating model for gender " + gender + "(" + str(len(data_set['authors_' + gender])) + ")")
            # For each author
            index = 1
            for author in data_set['authors_' + gender]:
                print("%i - Calculating model for author %s (%d)" % (index, author.get_author_name(), len(author.get_documents())))
                # For each doc
                for doc in author.get_documents():
                    # For each tokens
                    for token in doc.get_tokens():
                        language_model.inc_word(gender, token, 1.0)
                    # end for
                # end for
                index += 1
            # end for
        # end for

        # Finalize model
        language_model.finalize_model()

        # Create author profile
        docs_token = []
        truths = []
        for author in data_set['authors']:
            author_tokens = []
            for doc in author.get_documents():
                author_tokens += doc.get_tokens()
            # end for
            docs_token += [author_tokens]
            truths += [author.get_property("gender")]
        # end for

        # Assess model error rate
        print("Error rate : %f %%" % (PAN17Metrics.error_rate(language_model, docs_token, truths) * 100.0))
    # end with

    # Save model
    with open(args.output, 'w') as f:
        print("Saving output file %s" % args.output)
        # Write object
        pickle.dump(language_model, f)
    # end with

# end if