#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : create_dataset.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import argparse
import pySpeeches as ps
import numpy as np
from PAN17TruthLoader import PAN17TruthLoader
from PAN17LetterGramsReducer import PAN17LetterGramsReducer
from PAN17DiscreteSymbolGenerator import PAN17DiscreteSymbolGenerator
import cPickle as pickle

# ALPHABETS
ENGLISH_ALPHABET = u"aàáâãbcçdeéèêfghiíïîjklmnoóôõpqrstuúüûvwxyz"
ENGLISH_PUNCTUATIONS = u"?.!,;:#$§"
SPANISH_ALPHABET = u"aàáâãbcçdeéèêfghiíïîjklmnñoóôõpqrstuúüûvwxyz"
SPANISH_PUNCTUATIONS = u"?¿.!¡,;:#$§"
PORTUGUESE_ALPHABET = u"aàáâãbcçdeéèêfghiíïîjklmnñoóôõpqrstuúüûvwxyz"
PORTUGUESE_PUNCTUATIONS = u"?.!,;:#$§"
ARABIC_ALPHABET = u"‎ـﺍﺏﺒﺐﺑﺕﺖﺘﺗﺙﺚﺜﺛﺝﺞﺠﺟﺡﺢﺤﺣﺥﺨﺩﺫﺭﺯﺱﺵﺹﺽﻁﻅﻉﻍﻑﻕﻙﻝﻡﻥﻩﻭﻱءئإؤأ‎‎؎‎؏‎‎٭۞‎۩‎۝﴾﴿ﷰ‎ﷱ‎ﷲﷳ‎ﷴﷵ‎ﷷ‎ﷶ‎ﷸ‎ﷹﷺﷻ‎﷼‎بتثجحخد‎ذر‎زس‎ش‎صضطظ‎عغف‎ق‎ك‎لمن‎‎‎ه‎وي‎آ‎ة‎ى؀ال ا؛AaBbCcÇçDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZzتُهْةٍرٌلٌ"
ARABIC_PUNCTUATIONS = u":?؟‎.!,;،؍"


###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 RNN model")

    # Argument
    parser.add_argument("--file", type=str, help="Input data set Pickle file", default="pan17clef.p")
    parser.add_argument("--upper", dest="upper", action="store_true", help="Case sensitive")
    parser.set_defaults(upper=False)
    args = parser.parse_args()

    # Letter to symbol
    discrete_generator = PAN17DiscreteSymbolGenerator(alphabet=ENGLISH_ALPHABET + ENGLISH_PUNCTUATIONS + " ")

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        data_set = pickle.load(f)

        # For each author
        for author in data_set['corpus'].get_authors():
            # For each document
            symbol_input = np.array([])
            for doc in author.get_documents():
                doc_symbols = np.array(discrete_generator.tokens_to_symbols(doc.get_tokens()))
                #symbol_input = np.vstack((symbol_input, discrete_generator.tokens_to_symbols(doc.get_tokens())))
                print(doc_symbols.shape)
            # end for
            print(symbol_input)
            print(symbol_input.shape)
            exit()
        # end for
    # end with

# end if