#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import pySpeeches as ps
from PAN17PortugueseTextCleaner import PAN17PortugueseTextCleaner
from PAN17EnglishTextCleaner import PAN17EnglishTextCleaner
from PAN17SpanishTextCleaner import PAN17SpanishTextCleaner
from PAN17ArabicTextCleaner import PAN17ArabicTextCleaner
from PAN17TruthLoader import PAN17TruthLoader

###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 data set creator")

    # Argument
    parser.add_argument("--config", type=str, help="JSON config file", default="config.json")
    parser.add_argument("--file", type=str, help="Output Pickle file", default="pan17clef.p")
    args = parser.parse_args()

    # Load configuration file
    config = ps.importer.PySpeechesConfig.Instance()
    config.load(args.config)

    # New corpus
    corpus = ps.dataset.PySpeechesCorpus(name="PAN@CLEF17")

    # Set config
    config.set_corpus(corpus)

    # Import each sources
    for source in config.get_sources():
        # Get authors
        truths = PAN17TruthLoader.load_truth_file(source.get_entry_point() + "/truth.txt")

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

    # Save
    print("Saving file %s" % args.file)
    corpus.save(args.file)

# end if
