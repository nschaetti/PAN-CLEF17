#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import cPickle as pickle
import pySpeeches as ps
from cleaners.PAN17ArabicTextCleaner import PAN17ArabicTextCleaner
from cleaners.PAN17EnglishTextCleaner import PAN17EnglishTextCleaner
from cleaners.PAN17PortugueseTextCleaner import PAN17PortugueseTextCleaner
from cleaners.PAN17SpanishTextCleaner import PAN17SpanishTextCleaner
from tools.PAN17TruthLoader import PAN17TruthLoader

###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 data set creator")

    # Argument
    parser.add_argument("--config", type=str, help="JSON config file", default="config.json")
    parser.add_argument("--file", type=str, help="Output Pickle file", default="pan17clef.p")
    parser.add_argument("--name", type=str, help="Corpus' name", default="PAN@CLEF2017")
    args = parser.parse_args()

    # Load configuration file
    config = ps.importer.PySpeechesConfig.Instance()
    config.load(args.config)

    # New corpus
    corpus = ps.dataset.PySpeechesCorpus(name=args.name)

    # Set config
    config.set_corpus(corpus)

    # Final data set
    data_set = dict()

    # Data set informations
    data_set['languages'] = []
    data_set['genders'] = []

    # Import each sources
    the_author = None
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
            #print("Author %s is a %s from %s" % (author_id, gender, language))
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
            #print("Adding document %s to %s and %s" % (doc.get_title(), gender, language))
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
    print("Saving file %s" % args.file)
    with open(args.file, 'w') as f:
        pickle.dump(data_set, f)
    # end with

# end if
