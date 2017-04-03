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
import pySpeeches as ps
from cleaners.PAN17ArabicTextCleaner import PAN17ArabicTextCleaner
from cleaners.PAN17EnglishTextCleaner import PAN17EnglishTextCleaner
from cleaners.PAN17PortugueseTextCleaner import PAN17PortugueseTextCleaner
from cleaners.PAN17SpanishTextCleaner import PAN17SpanishTextCleaner
from tools.PAN17TruthLoader import PAN17TruthLoader
import xml.etree.cElementTree as ET

###########################
# FUNCTIONS
###########################


def write_xml_output(ids, lng, variety, gender, output_dir):
    #print("Writing %s" % os.path.join(output_dir, ids + ".xml"))
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
def generate_data_set(config_file, output_file):

    # Load configuration file
    config = ps.importer.PySpeechesConfig.Instance()
    config.load(config_file)

    # New corpus
    corpus = ps.dataset.PySpeechesCorpus("PAN17 Author Profiling data set")

    # Set config
    config.set_corpus(corpus)

    # Final data set
    data_set = dict()

    # Data set informations
    data_set['languages'] = []
    data_set['genders'] = []

    # Import each sources
    for source in config.get_sources():
        # Get authors
        truths = PAN17TruthLoader.load_truth_file(source.get_entry_point() + "/truth.txt")

        # Create collections
        for author_id, gender, language in truths:
            if gender not in data_set:
                data_set[gender] = ps.dataset.PySpeechesDocumentCollection(gender)
            # end if
            if language not in data_set:
                data_set[language] = ps.dataset.PySpeechesDocumentCollection(language)
            # end if
            if "authors_" + gender not in data_set:
                data_set["authors_" + gender] = []
            # end if
            if "authors_" + language not in data_set:
                data_set["authors_" + language] = []
            # end if
            if gender not in data_set['genders']:
                data_set['genders'] += [gender]
            # end if
            if language not in data_set['languages']:
                data_set['languages'] += [language]
                # end if
        # end for

        # Directory importer
        importer = ps.importer.PySpeechesDirectoryImporter(source, eval(source.get_text_cleaner()),
                                                           ps.importer.PySpeechesXMLFileImporter)

        # Import source
        importer.import_source()

        # For each authors
        for author_id, gender, language in truths:
            # Get author object
            author = corpus.get_author(author_id)
            # Set gender
            author.set_property("gender", gender)
            # Set language
            author.set_property("language", language)
        # end for
    # end for

    # For each author
    for author in corpus.get_authors():
        gender = author.get_property("gender")
        language = author.get_property("language")
        for doc in author.get_documents():
            # print("Adding document %s to %s and %s" % (doc.get_title(), gender, language))
            data_set[gender].add_document(doc, check_doublon=False)
            data_set[language].add_document(doc, check_doublon=False)
        # end for
        data_set["authors_" + gender] += [author]
        data_set["authors_" + language] += [author]
    # end for

    # Corpus
    data_set['corpus'] = corpus
    data_set['authors'] = corpus.get_authors()

    # Save
    print("Saving file %s" % output_file)
    with open(output_file, 'w+') as f:
        pickle.dump(data_set, f)
    # end with

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
    docs_token = []
    author_tokens = []
    for doc in the_author.get_documents():
        author_tokens += doc.get_tokens()
    # end for
    docs_token += [author_tokens]

    # Classify
    return the_model.classify(docs_token)
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
    parser.add_argument("--no-update", action='store_true', default=False, help="Don't update data set if available")
    args = parser.parse_args()

    # For each languages
    for lang in ["en", "es", "pt", "ar"]:
        data_set_file = os.path.join("/home/schaetti17/inputs", lang, "pan17" + lang + ".p")
        # Generate cleaned data set
        if not args.no_update or not os.path.exists(os.path.join("/home/schaetti17/inputs", lang,
                                                                 "pan17" + lang + ".p")):
            # Create config file
            print("Creating configuration files for language %s..." % lang)
            generate_config_file(lang, args.input_dataset, "/home/schaetti17/config")

            # Generate data files
            print("Generating data set for %s" % lang)
            generate_data_set(os.path.join("/home/schaetti17/config", lang + ".json"), data_set_file)
        # end if

        # Loading models
        print("Loading models...")
        tf_idf_model = load_model("/home/schaetti17/models", lang, args.tfidf_models)

        # Load data set
        print("Load data from %s" % data_set_file)
        with open(data_set_file, 'r') as f:
            # Load
            data_set = pickle.load(f)

            # For each authors
            for c_author in data_set['authors']:
                # Author's name
                name = c_author.get_name()

                # Classification
                variety = classify_variety(the_author=c_author, the_model=tf_idf_model)

                # Write
                write_xml_output(name, lang, variety, "male", args.output_dir)
            # end for
        # end with
    # end for

# end if