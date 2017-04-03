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
from tfidf.PAN17TfIdfModel import PAN17TfIdfModel
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
    parser = argparse.ArgumentParser(description="PAN17 TF-IDF for language classification")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--output", type=str, help="Output model Pickle file", default="output.p")
    args = parser.parse_args()

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        print("Loading data set %s" % args.file)
        data_set = pickle.load(f)

        # Sample size
        n_samples = len(data_set['authors'])
        fold_size = int(math.ceil(n_samples / 10.0))

        # Word vector model
        word_vector_model = PAN17TfIdfModel(classes=data_set['languages'], upper=False, use_punct=False,
                                             punc=ENGLISH_PUNCTUATIONS)

        # K-10 fold
        author_sets = np.array(data_set['authors'])
        author_sets.shape = (10, fold_size)

        # For each fold
        error_rate_average = np.array([])
        for i in range(10):

            # Initialize
            #print("Initializing word vector model...")
            for doc in data_set['corpus'].get_documents():
                for token in doc.get_tokens():
                    word_vector_model.init_token_count(token)
                # end for
            # end for

            # Select training and test sets
            test = author_sets[i]
            training = np.delete(author_sets, i, axis=0)
            training.shape = (fold_size * 9)

            # Train with each document
            #print("Training word vector model...")
            for author in training:
                language = author.get_property("language")
                # For each doc
                for doc in author.get_documents():
                    word_vector_model.train(doc.get_tokens(), language)
                # end for
            # end for

            # Finalize training
            #print("Finalizing training...")
            word_vector_model.finalize()

            # Create author profile
            #print("Creating author profiles...")
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
            #print("Calculating error rate...")
            error_rate = PAN17Metrics.error_rate(word_vector_model, docs_token, truths) * 100.0
            print("Error rate : %f %%" % error_rate)
            error_rate_average = np.append(error_rate_average, error_rate)
        # end for
        print("10K Fold error rate is %f" % np.average(error_rate_average))
    # end with

    # Save model
    with open(args.output, 'w+') as f:
        pickle.dump(word_vector_model, f)
    # end with

# end if