import sys
import re
import time

from plumbum import cli, local, colors, TEE
from wavelets import extractwaveletcoef

MATLAB_DIR = '../MATLAB/'
CONFIG_DIR = '../configs/'
LIBS_DIR = '../libs/'
sys.path.append(LIBS_DIR)

# Currently the the following are hardcoded Fixme
preproc_csvs_dir = local.path(MATLAB_DIR + 'temp/')
finish_mat_dir = local.path('../OUT/')
DEFAULT_MEG_CONFIG = CONFIG_DIR + 'MEG_features.conf'
DEFAULT_ACOUSTIC_CONFIG = CONFIG_DIR + 'Acoustic_features.conf'
DEFAULT_MEG_DIR = 'convert_to_mat/'
DEFAULT_MEG_MATFILE = 'meg_features.mat'

# Define the commands that we will be using
# Locals
mv = local['mv']
mkdir = local['mkdir']['-p']
rm = local['rm']['-r']

matlab_cmd = local['matlab']
matlab_nogui = matlab_cmd['-nojvm']['-nodisplay']['-nosplash']['-r']
open_smile = local['SMILExtract']


# Todo remove when this is no longer necessary
MATLAB_REPLACE_TOKEN = '<REPLACE>'
mat_no_gui = '-nosplash -nojvm -nodisplay -r \"addpath ' + MATLAB_DIR + "';" + MATLAB_REPLACE_TOKEN + '\"'


def preprocess(subject, audioonly=False):
    """
    Run matlab with the preprocess script, to extract csv's for the provided subject
    :param subject:
    :return: The output of the command
    """
    with local.cwd(MATLAB_DIR):
        if audioonly:
            cmd = "AUDIOONLY=1;subj_in=%s;preprocess" % ''.join([i for i in subject if i.isdigit()])
        else:
            cmd = "subj_in=%s;preprocess" % ''.join([i for i in subject if i.isdigit()])
        return matlab_nogui[cmd] & TEE


def mat_compress(infile, outfile):
    with local.cwd(MATLAB_DIR):
        cmd = "in_file='%s';out_file='%s';matCompress" % (infile, outfile)
        return matlab_nogui[cmd] & TEE
        # return matlab_cmd(mat_no_gui.replace(MATLAB_REPLACE_TOKEN, ))


def opensmile_extract_features(config, input_csv, output_csv):
    exec_args = open_smile['-C', config, '-I', input_csv, '-O', output_csv]
    return exec_args & TEE


def find_completed(toplevel):
    """
    Look in the output directory, and determine the completed subjects
    :return:
    """
    toplevel = local.path(toplevel)
    if not toplevel.exists():
        return []

    completed = {}
    for subject in toplevel:
        if 'CC' not in subject.name:
            continue
        else:
            completed[subject.name] = {test.name: test.list() for test in subject if len(test.list()) > 0}

    return completed
