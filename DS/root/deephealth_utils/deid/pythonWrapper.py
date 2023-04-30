# Originally created June 2018 by Tyler Sorenson
#
# Python wrapper for the perl PHI removal scripts
# Uses a direct path to access all files

import subprocess
import sys
import os
import pdb

def scrub_wrapper(file_in, file_out, shell=False, tmp_file_path=None):
    file_in = os.path.abspath(file_in)
    file_out = os.path.abspath(file_out)

    if tmp_file_path is None:
        f_name, ext = os.path.splitext(os.path.basename(file_in))
        file_in_has_header = f_name + '_input_with_header' + ext
    else:
        file_in_has_header = tmp_file_path

    path_to_deid = os.path.abspath(os.path.dirname(__file__))

    subprocess.call(['perl', path_to_deid + '/addHeader.pl', file_in, file_in_has_header], shell=shell)
    subprocess.call(['perl', path_to_deid + '/deid.pl', file_in_has_header, file_out, path_to_deid + '/deid-output.config', path_to_deid, 'NO COMMENTARY'], shell=shell)
    subprocess.call(['perl', path_to_deid + '/scrubFormatter.pl', file_out], shell=shell)
    subprocess.call(['perl', '-pi', '-e', 'chomp if eof', file_out], shell=shell)

#scrub_wrapper(sys.argv[1], sys.argv[2])
