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
import torch
import cPickle as pickle
import pySpeeches as ps
from cleaners.PAN17ArabicTextCleaner import PAN17ArabicTextCleaner
from cleaners.PAN17EnglishTextCleaner import PAN17EnglishTextCleaner
from cleaners.PAN17PortugueseTextCleaner import PAN17PortugueseTextCleaner
from cleaners.PAN17SpanishTextCleaner import PAN17SpanishTextCleaner
from features.PAN17FeaturesMatrixGenerator import PAN17FeaturesMatrixGenerator
from reducer.PAN17LetterGramsReducer import PAN17LetterGramsReducer
from deep_models.PAN17ConvNet import PAN17ConvNet
from deep_models.PAN17DeepNNModel import PAN17DeepNNModel
from pySpeeches.importer.PySpeechesConfig import PySpeechesConfig
from tools.PAN17TruthLoader import PAN17TruthLoader
import xml.etree.cElementTree as ET
import logging


# ALPHABETS
alphabet = dict()
punctuations = dict()
alphabet['en'] = u"aàáâãbcçdeéèêfghiíïîjklmnoóôõpqrstuúüûvwxyz"
punctuations['en'] = u"?.!,;:#$§"
alphabet['es'] = u"aàáâãbcçdeéèêfghiíïîjklmnñoóôõpqrstuúüûvwxyz"
punctuations['es'] = u"?¿.!¡,;:#$§"
alphabet['pt'] = u"aàáâãbcçdeéèêfghiíïîjklmnñoóôõpqrstuúüûvwxyz"
punctuations['pt'] = u"?.!,;:#$§"
alphabet['ar'] = u":?؟‎.!,;،؍‎‎؎‎ﺏﺒﺐﺑﺕﺖﺘﺗﺙﺚﺜﺛﺝﺞﺠﺡﺢﺤﺥﺨﺩﺫﺭﺯﺱﺵﺹﺽﻁﻅﻉﻍﻑﻕﻙﻝﻡﻥﻩﻭﻱءئإؤأـنيتثغخشصفعسمكحوهدجبا؛زطىلنقرتثذضظب"
punctuations['ar'] = u":?؟‎.!,;،؍"


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


# Classify gender
def classify_gender(the_author, the_model):
    # Reducer
    reducer = PAN17LetterGramsReducer(
        letters=alphabet[lang],
        punctuations=punctuations[lang], add_punctuation=True, add_first_letters=True, add_end_letters=True,
        add_end_grams=True, add_first_grams=True, upper_case=False)

    # Matrix generator
    matrix_generator = PAN17FeaturesMatrixGenerator(letters=alphabet[lang],
                                                    punctuations=punctuations[lang], upper_case=False)

    # For each document
    author_mapping = dict()
    for doc in the_author.get_documents():
        # Maps the document
        doc_mapping = doc.map(reducer)
        # Reduce
        author_mapping = reducer.reduce([author_mapping, doc_mapping])
    # end for

    # Generate author matrix
    m = matrix_generator.generate_matrix(author_mapping)
    author_matrix = the_model.matrix_to_tensor(m)
    author_tensor = torch.DoubleTensor(1, 1, author_matrix.size()[1], author_matrix.size()[2])
    author_tensor[0] = author_matrix

    # Predict gender
    pred = the_model.predict(author_tensor)

    # Delete matrix
    del m
    del author_matrix
    del author_tensor

    return pred
# classify_gender

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
    parser.add_argument("--cnn-models", type=str, default="cnn.pth", metavar='C', help="CNN model filename")
    parser.add_argument("--log-warning", action='store_true', default=False, help="Log level warnings")
    parser.add_argument("--log-error", action='store_true', default=False, help="Log level error")
    parser.add_argument("--base-dir", type=str, default=".", metavar='B', help="Base directory")
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar='S', help="Random seed (default:1)")
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
        config.info("Loading TFIDF models from %s/%s/%s..." % (models_dir, lang, args.tfidf_models))
        tf_idf_model = load_model(models_dir, lang, args.tfidf_models)

        # Loading models
        config.info("Loading Deep Learning models from %s/%s..." % (models_dir, args.cnn_models))
        cnn_model_path = os.path.join(models_dir, lang, args.cnn_models)
        cnn_model = PAN17DeepNNModel(PAN17DeepNNModel.load(cnn_model_path), classes=("male", "female"),
                                     lr=args.lr, momentum=args.momentum, seed=args.seed)
        #cnn_model = PAN17DeepNNModel.load(cnn_model_path)

        # For each authors
        for c_author in data_set:
            # Author's name
            name = c_author.get_name()

            # Classification
            gender = classify_gender(the_author=c_author, the_model=cnn_model)
            variety = classify_variety(the_author=c_author, the_model=tf_idf_model)

            # Write
            write_xml_output(name, lang, variety, gender, output_lang_dir)
        # end for

        # Delete variables
        del data_set
        del tf_idf_model
        del cnn_model
    # end for

# end if