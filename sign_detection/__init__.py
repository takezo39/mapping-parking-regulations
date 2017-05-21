#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detecting parking regulation signs from video taken from a driving car. Associating  GPS coordinates with signs. Deriving parking pockets and their associated parking rules.
"""

import os
import logging
import sys

__author__ = "Mark Hashimoto"
__copyright__ = "Copyright 2017, Data Science Retreat portfolio project"
__version__ = "0.1"
__email__ = "mark.m.hashimoto@gmail.com"

base_dir = os.path.dirname(__file__)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)

# Create formatter and add it to the handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set STDERR handler as the only handler
logger.handlers = [handler]

def create_argument_parser():
    """Creates general purpose command-line options

    Pass it in as part of the 'parents' argument to your own ArgumentParser.
    More arguments can be added to your own ArgumentParser"""
    import time
    import argparse

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-month', default=time.strftime("%Y-%m"), type=str,
                        help="Month to be filtered in the form year-month")

    parser.add_argument('-parsedate', default=time.strftime("%Y-%m-%d"),
                        type=str, help="Date for which data will be studied.")

    parser.add_argument('-DB', default="DB", type=str,
                        help='Database to be queried')

    parser.add_argument('-force', action='store_true',
                        help="If True, force processing actions"
                             "and ignore backups.")

    parser.add_argument('-debug', action='store_true',
                        help="Do not run the main function")

    parser.add_argument('-logging_level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level of detail.')

    return parser
