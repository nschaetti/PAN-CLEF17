#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core/downloader/PySpeechesConfig.py
# Description : .
# Date : 20th of February 2017
#
# This file is part of pySpeeches.  pySpeeches is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Import package
from pySpeeches.cleaning.cleaning_functions import *
from pySpeeches.cleaning.PySpeechesCleaner import *

#
# Regex for twitter quote
#
QUOTE_PATTERN = "(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9_]+)"

#
# English Alphabet
#
ENGLISH_ALPHABET = "AaÀàÁáÂâÃãBbCcÇçDdEeÉéÈèÊêFfGgHhIiÍíÏïÎîJjKkLlMmNnOoÓóÔôÕõPpQqRrSsTtUuÚúÜüÛûVvWwXxYyZz?.!,;:#$§"
ENGLISH_PUNCTUATIONS = "?.!,;:"


# An Author
class PAN17EnglishTextCleaner(PySpeechesCleaner):

    # Constructor
    def __init__(self):
        """
        Constructor.
        """
        super(PAN17EnglishTextCleaner, self).__init__()
    # end __init__

    # Remove Twitter quotes
    @staticmethod
    def remove_twitter_quotes(text):
        """
        Remove Twitter quotes?.!,;:#$§
        :param text: Text to clean.
        :return: Cleaned text.
        """
        text = re.sub(QUOTE_PATTERN, "", text)
        return text
    # end remove_twitter_quotes

    # Keep only Portuguese letters
    @staticmethod
    def keep_only_letters(text):
        """
        Keep only letters
        :param text: Text to clean
        :return: Cleaned text.
        """
        cleaned_text = ""
        for letter in text:
            if letter in ENGLISH_ALPHABET or letter == " ":
                cleaned_text += letter
        # end for
        return cleaned_text
    # Clean text

    @staticmethod
    def clean_text(text):
        """
        Clean text.
        :param text: Text to clean.
        :return: Cleaned text.
        """
        # Clean twitter
        text = PyCleaningTool.remove_urls(text)                                 # Remove URls.
        text = PAN17EnglishTextCleaner.remove_twitter_quotes(text)              # Remove Twitter quotes.

        # Clean numbers
        text = text.replace(u"#", u"")                                            # No #
        text = text.replace(u"$", u"")                                            # No $
        text = text.replace(u"§", u"")                                            # No §
        text = PyCleaningTool.format_numbers(text)                              # 100'000 to 100000

        # Punctuations
        text = text.replace(u"?",  u" ? ")                                       # ???? to ? ? ? ?
        text = text.replace(u"!",  u" ! ")                                       # !!!! to ! ! ! !
        text = text.replace(u".",  u" . ")                                       # .... to . . . .
        text = text.replace(u",", u" , ")                                       # ,, to ,
        text = text.replace(u"…",  u" . . . ")                                  # ... to . . .
        text = text.replace(u"\n", u" ")                                        # New line to space
        text = text.replace(u"'",  u" ")                                        # Replace ' by space
        text = text.replace(u"’",  u" ")                                        # Replace ’ by space
        for p in ENGLISH_PUNCTUATIONS:
            text = text.replace(p,  u" " + p + u" ")
        # end for

        # Characters and spaces
        text = PyCleaningTool.remove_useless_characters(text)                   # Remove useless characters.
        text = PyCleaningTool.inverse_dot_and_quote(text)                       # Inverse dot and quote.
        text = PyCleaningTool.remove_sharp_hash_tag(text)                       # Remove hash tags.
        text = PyCleaningTool.insert_space_between_special_word(text)           # Insert space between special words.
        text = PyCleaningTool.space_between_each_word(text)                     # We want a space between each words.
        text = PAN17EnglishTextCleaner.keep_only_letters(text)                  # Keep only Portuguese letters.
        text = PyCleaningTool.many_spaces_to_one_space(text)  # No multiple spaces
        text = text.strip()  # No space at the beginning or at the end of the tweet.
        return text
    # end clean_text

# end PyMillerCenterCleaner