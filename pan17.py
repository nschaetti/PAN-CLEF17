#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : word_vector_gender_classifier.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import os
import argparse
import json
import numpy as np
import math
import cPickle as pickle

###########################
# FUNCTIONS
###########################


# Generate a config file
def generate_config_file(l, input_dir, config_dir):
    """

    :param l:
    :param input_dir:
    :param config_dir:
    :return:
    """

    # Names
    names = dict()
    names['ar'] = "PAN17 Arabic Corpus"
    names['en'] = "PAN17 English Corpus"
    names['es'] = "PAN17 Spanish Corpus"
    names['pt'] = "PAN17 Portuguese Corpus"

    # Description
    description = dict()
    description['ar'] = "Import PAN17 Arabic Corpus"
    description['en'] = "Import PAN17 English Corpus"
    description['es'] = "Import PAN17 Spanish Corpus"
    description['pt'] = "Import PAN17 Portuguese Corpus"

    # Cleaners
    cleaners = dict()
    cleaners['ar'] = "PAN17ArabicTextCleaner"
    cleaners['en'] = "PAN17EnglishTextCleaner"
    cleaners['es'] = "PAN17SpanishTextCleaner"
    cleaners['pt'] = "PAN17PortugueseTextCleaner"

    # JSON
    json_config = dict()
    json_config['type'] = "directory"
    json_config['name'] = names[l]
    json_config['description'] = description[l]
    json_config['entry_point'] = input_dir + "/" + l
    json_config['file_regex'] = "[0-9a-zA-Z]+\\.xml"
    json_config['text_cleaner'] = cleaners[l]
    json_config['dict_size'] = 30000
    json_config['check_doublon'] = False

    # Write the config file
    with open(os.path.join(config_dir, lang + ".json"), 'w+') as fi:
        json.dump(json_config, fi, sort_keys=True, indent=4)
    # end with

# end generate_config_file

###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 Author Profiling Task software")

    # Argument
    parser.add_argument("--input_dataset", type=str, help="Input data set directory", default="../inputs",
                        required=True)
    parser.add_argument("--input_run", type=str, default="none", metavar='R', help="Input run (default:1)",
                        required=True)
    parser.add_argument("--output_dir", type=str, help="Input directory", default="../outputs", required=True)
    parser.add_argument("--data_server", type=str, help="Data server", default="None")
    parser.add_argument("--token", type=str, default="", metavar='T', help="Token", required=True)
    parser.add_argument("--tfidf_models", type=str, default="tfidf.p", metavar='F', help="TF-IDF model filename")
    parser.add_argument("--cnn_models", type=str, default="cnn.p", metavar='C', help="CNN model filename")
    args = parser.parse_args()

    # For each languages
    for lang in ["en", "es", "pt", "ar"]:

        # Create config file
        generate_config_file(lang, args.input_dataset, "/home/schaetti17/config")

    # end for

# end if