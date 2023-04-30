import sys
import os
import filecmp
from pythonWrapper import scrub_wrapper
import random
import pdb
import numpy as np
from datetime import datetime

def contains_phi(input):
    if not os.path.isfile(input):
        n = datetime.now().microsecond
        temp_filename_in = 'text_phi_check_input_' + str(n) + '.txt'
        with open(temp_filename_in, 'w') as new_file:
            new_file.write(input.encode('utf-8'))
            input = temp_filename_in
    n = datetime.now().microsecond
    temp_filename_out = 'text_phi_check_output_' + str(n) + '.txt'
    scrub_wrapper(input, temp_filename_out)

    result = not filecmp.cmp(input, temp_filename_out)
    if os.path.isfile(temp_filename_in):
        os.remove(temp_filename_in)
    os.remove(temp_filename_out)
    return result

def replace_phi(input, shell=False):
    n = datetime.now().microsecond  # give essentially a random name in case multiple processes are running
    in_file = 'input' + str(n) + '.txt'
    out_file = 'scrubbed' + str(n) + '.txt'
    with open(in_file, 'w') as text_file:
        text_file.write(input)
    scrub_wrapper(in_file, out_file, shell=shell)
    with open(out_file, 'r') as scrubbed_file:
        result = scrubbed_file.read()
    os.remove(in_file)
    os.remove(out_file)
    return result
