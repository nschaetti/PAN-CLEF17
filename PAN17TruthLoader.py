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


# An Author
class PAN17TruthLoader(object):

    # Constructor
    def __init__(self):
        """
        Constructor.
        """
        pass
    # end __init__

    # Load a truth file
    @staticmethod
    def load_truth_file(filename):
        truths = []
        # Open file
        with open(filename, 'r') as f:
            # Read file
            truth_data = f.read()
            # For each line
            for line in truth_data.split('\n'):
                if len(line) > 0:
                    # Split data
                    author_data = line.split(":::")
                    # Add
                    truths += [author_data]
                # end if
            # end for
        # end with
        return truths
    # end load_truth_file

# end PAN17TruthLoader