#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import numpy as np
import math
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
    parser = argparse.ArgumentParser(description="PAN17 language model for country classification")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--output", type=str, help="Output model Pickle file", default="output.p")
    args = parser.parse_args()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)

        # Sample size
        n_samples = len(data_set['authors'])
        fold_size = int(math.ceil(n_samples/10.0))

        # Probabilistic model
        language_model = PAN17LanguageModel(classes=data_set['languages'], upper=False)
        language_model.set_smoothing(PAN17JelinekMercerSmoothing(l=0.1))

        # K-10 fold
        author_sets = np.array(data_set['authors'])
        author_sets.shape = (10, fold_size)

        # For each fold
        error_rate_average = np.array([])
        for i in range(10):
            print("Error rate evaluation for K fold %d" % i)

            # Initialize model
            for doc in data_set['corpus'].get_documents():
                for token in doc.get_tokens():
                    language_model.init_token_count(token)
                # end for
            # end for

            # Select training and test sets
            test = author_sets[i]
            training = np.delete(author_sets, i, axis=0)
            training.shape = (fold_size * 9)

            # For each author in training
            for author in training:
                language = author.get_property("language")
                # For each doc
                for doc in author.get_documents():
                    # For each tokens
                    for token in doc.get_tokens():
                        language_model.inc_word(language, token, 1.0)
                    # end for
                # end for
            # end for

            # Finalize model
            language_model.finalize_model()

            # Create author profile
            docs_token = []
            truths = []
            for author in test:
                author_tokens = []
                for doc in author.get_documents():
                    author_tokens += doc.get_tokens()
                # end for
                docs_token += [author_tokens]
                truths += [author.get_property("language")]
            # end for

            # Assess model error rate
            error_rate = PAN17Metrics.error_rate(language_model, docs_token, truths) * 100.0
            print("Error rate : %f %%" % error_rate)
            error_rate_average = np.append(error_rate_average, error_rate)
        # end for
        print("10K Fold error rate is %f" % np.average(error_rate_average))
    # end with

    # Save model
    with open(args.output, 'w') as f:
        print("Saving output file %s" % args.output)
        # Write object
        pickle.dump(language_model, f)
    # end with

# end if