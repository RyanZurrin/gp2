import os, sys
sys.path.insert(0, '/Users/tylersorenson/Desktop/DeepHealth/mammo_utils/deid')
from pythonWrapper import scrub_wrapper

def test_scrubbing(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):# and filename == 'test8.txt':
            input_file = os.path.relpath(directory, os.getcwd()) + '/' + filename
            name, ext = os.path.splitext(filename)
            output_file = 'output/' + name + '_output' + ext
            scrub_wrapper(input_file, output_file)


test_scrubbing('/Users/tylersorenson/Desktop/DeepHealth/mammo_utils/deid/tests/input')
