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

        # Word vector model
        word_vector_model = PAN17TfIdfModel(classes=data_set['languages'], upper=False, use_punct=False,
                                             punc=ARABIC_PUNCTUATIONS)

        # Initialize
        print("Initializing word vector model...")
        for doc in data_set['corpus'].get_documents():
            for token in doc.get_tokens():
                word_vector_model.init_token_count(token)
            # end for
        # end for

        # Train with each document
        print("Training word vector model...")
        for author in data_set['authors']:
            language = author.get_property("language")
            # For each doc
            for doc in author.get_documents():
                word_vector_model.train(doc.get_tokens(), language)
            # end for
        # end for

        # Finalize training
        print("Finalizing training...")
        word_vector_model.finalize()

        # Create author profile
        print("Creating author profiles...")
        docs_token = []
        truths = []
        for author in data_set['authors']:
            author_tokens = []
            for doc in author.get_documents():
                author_tokens += doc.get_tokens()
            # end for
            docs_token += [author_tokens]
            truths += [author.get_property("language")]
        # end for

        # Assess model error rate
        print("Calculating error rate...")
        error_rate = PAN17Metrics.error_rate(word_vector_model, docs_token, truths) * 100.0
        print("Error rate : %f %%" % error_rate)
    # end for

    # Save model
    with open(args.output, 'w+') as f:
        pickle.dump(word_vector_model, f)
    # end with

# end if