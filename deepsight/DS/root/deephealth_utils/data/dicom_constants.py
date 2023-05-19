#import sys
#if sys.version[0] == '2':  # Python 2.X
#     from format_helpers import *
# elif sys.version[0] == '3':  # Python 3.X
#     from .format_helpers import *
import pandas as pd
import os

ALT_TAGS = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),'alt_tags.csv'))
ALT_TAGS['tag'] = ALT_TAGS['tag'].apply(lambda x: eval(x))
