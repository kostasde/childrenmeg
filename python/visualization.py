import argparse
import utils

from models import *
from mne.viz import plot_topomap


def maximize_activation(input, output, lr=0.2, input_generator=utils.pink_noise, l2_reg=0.05, max_steps=1e4):
    """
    Given input and output tensors, perform gradient ascent from data created by input generator until there is some
    semblance of convergence, or a maximum number of iterations have occurred
    :return: The new input data achieved, as np.array
    """



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Produce max-activations and associated visualizations for SCNN and '
                                                 'Attention LSTM models.')
    parser.add_argument('model', choices=['alstm', 'scnn'], help='Which model to produce output for.')
    parser.add_argument('--activ', help='Previously stored file with max activations to be calculated for model')
    parser.add_argument('--save-dir', help='Directory to save all output to.')

