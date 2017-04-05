#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : word_vector_gender_classifier.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import sys
sys.path.append("/home/schaetti17/pySpeeches")
import os
import argparse
import json
import numpy as np
import math
import cPickle as pickle
import pySpeeches as ps
from cleaners.PAN17ArabicTextCleaner import PAN17ArabicTextCleaner
from cleaners.PAN17EnglishTextCleaner import PAN17EnglishTextCleaner
from cleaners.PAN17PortugueseTextCleaner import PAN17PortugueseTextCleaner
from cleaners.PAN17SpanishTextCleaner import PAN17SpanishTextCleaner
from pySpeeches.importer.PySpeechesConfig import PySpeechesConfig
from tools.PAN17TruthLoader import PAN17TruthLoader
import xml.etree.cElementTree as ET
import logging

###########################
# FUNCTIONS
###########################


def write_xml_output(ids, lng, variety, gender, output_dir):
    config.info("Writing %s" % os.path.join(output_dir, ids + ".xml"))
    root = ET.Element("author", id=ids, lang=lng, variety=variety, gender=gender)
    tree = ET.ElementTree(root)
    tree.write(os.path.join(output_dir, ids + ".xml"))
# end write_xml_output


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
    source = dict()
    source['type'] = "directory"
    source['name'] = names[l]
    source['description'] = description[l]
    source['entry_point'] = input_dir + "/" + l
    source['file_regex'] = "[0-9a-zA-Z]+\\.xml"
    source['text_cleaner'] = cleaners[l]
    source['dict_size'] = 30000
    source['check_doublon'] = False
    json_config['sources'] = [source]

    # Write the config file
    with open(os.path.join(config_dir, lang + ".json"), 'w+') as fi:
        json.dump(json_config, fi, sort_keys=True, indent=4)
    # end with
# end generate_config_file


# Generate data set
def generate_data_set(config_file):

    # Load configuration file
    c_config = ps.importer.PySpeechesConfig.Instance()
    c_config.load(config_file)

    # New corpus
    corpus = ps.dataset.PySpeechesCorpus("PAN17 Author Profiling data set")

    # Set config
    config.set_corpus(corpus)

    # Import each sources
    for source in c_config.get_sources():
        # Directory importer
        importer = ps.importer.PySpeechesDirectoryImporter(source, eval(source.get_text_cleaner()),
                                                           ps.importer.PySpeechesXMLFileImporter)

        # Import source
        importer.import_source()
    # end for

    return corpus.get_authors()
# end generate_data_set


# Load model
def load_model(model_dir, lng, model_file):
    with open(os.path.join(model_dir, lng, model_file)) as fi:
        model = pickle.load(fi)
        return model
    # end with
# end load_model


# Classify variety
def classify_variety(the_author, the_model):
    # Get all tokens
    author_tokens = []
    for doc in the_author.get_documents():
        author_tokens += doc.get_tokens()
    # end for

    # Classify
    return the_model.classify(author_tokens)
# end classify_variety

###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 Author Profiling Task software")

    # Argument
    parser.add_argument("--input-dataset", type=str, help="Input data set directory", default="../inputs",
                        required=True)
    parser.add_argument("--input-run", type=str, default="none", metavar='R', help="Input run (default:1)",
                        required=True)
    parser.add_argument("--output-dir", type=str, help="Input directory", default="../outputs", required=True)
    parser.add_argument("--data-server", type=str, help="Data server", default="None")
    parser.add_argument("--token", type=str, default="", metavar='T', help="Token", required=True)
    parser.add_argument("--tfidf-models", type=str, default="tfidf.p", metavar='F', help="TF-IDF model filename")
    parser.add_argument("--cnn-models", type=str, default="cnn.p", metavar='C', help="CNN model filename")
    parser.add_argument("--log-warning", action='store_true', default=False, help="Log level warnings")
    parser.add_argument("--log-error", action='store_true', default=False, help="Log level error")
    parser.add_argument("--base-dir", type=str, default=".", metavar='B', help="Base directory")
    args = parser.parse_args()

    # Load configuration file
    config = PySpeechesConfig.Instance()
    config.set_log_level(logging.INFO)

    # Errors
    if args.log_error:
        config.set_log_level(logging.ERROR)
        # end if

    # Warnings
    if args.log_warning:
        config.set_log_level(logging.WARNING)
    # end if

    # Directories
    base_dir = args.base_dir
    inputs_dir = os.path.join(base_dir, "inputs")
    config_dir = os.path.join(base_dir, "config")
    models_dir = os.path.join(base_dir, "models")

    # For each languages
    for lang in ["en", "es", "pt", "ar"]:
        # Create output directory
        output_lang_dir = os.path.join(args.output_dir, lang)
        if not os.path.exists(output_lang_dir):
            os.makedirs(output_lang_dir)
        # end if

        # Create config file
        config.info("Creating configuration files for language %s..." % lang)
        generate_config_file(lang, args.input_dataset, config_dir)

        # Generate data files
        config.info("Generating data set for %s" % lang)
        data_set = generate_data_set(os.path.join(config_dir, lang + ".json"))

        # Loading models
        config.info("Loading models from %s/%s..." % (models_dir, args.tfidf_models))
        tf_idf_model = load_model(models_dir, lang, args.tfidf_models)

        # For each authors
        for c_author in data_set:
            # Author's name
            name = c_author.get_name()

            # Classification
            variety = classify_variety(the_author=c_author, the_model=tf_idf_model)

            # Write
            write_xml_output(name, lang, variety, "male", output_lang_dir)
        # end for
    # end for

# end if