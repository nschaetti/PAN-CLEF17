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


class PAN17Metrics(object):

    # Error rate
    @staticmethod
    def error_rate(classifier, docs, truth):
        index = 0
        count = 0
        for doc in docs:
            #print("%d on %d" % (index, len(docs)))
            predicted = classifier.classify(doc)
            if predicted != truth[index]:
                count += 1
            # end if
            index += 1
        # end for
        return float(count) / float(len(docs))
    # end error_rate

# end PAN17Metrics