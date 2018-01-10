import sys
import re
import time

import shutil
# import matplotlib.animation as anim
# import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from mne.viz import plot_topomap
from plumbum import cli, local, colors, TEE, FG, ProcessExecutionError, ProcessTimedOut

#import matlab.engine as mtlb
#from wavelets import extractwaveletcoef

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

#matlab_cmd = local['matlab']
# print('Starting MATLAB...')
#mtlbeng = mtlb.start_matlab()
#mtlbeng.addpath(MATLAB_DIR, nargout=0)
#print('MATLAB ready!')
#matlab_nogui = matlab_cmd['-nojvm']['-nodisplay']['-nosplash']['-r']
# open_smile = local['SMILExtract']


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


def mat_compress(infile, outfile, timeout=None):
    try:
        mtlbeng.workspace['in_file'] = infile
        mtlbeng.workspace['out_file'] = outfile
        mtlbeng.matCompress(nargout=0)
    # pretty stupid, FIXME
    except (NameError, AttributeError):
        print('Failed to use MATLAB python API')
        print('Falling back to command line matlab')
        with local.cwd(MATLAB_DIR):
            cmd = "in_file='%s';out_file='%s';matCompress" % (infile, outfile)
            return matlab_nogui[cmd] & FG(timeout=timeout)
            # return matlab_cmd(mat_no_gui.replace(MATLAB_REPLACE_TOKEN, ))


def opensmile_extract_features(config, input_csv, output_csv, timeout=None):
    exec_args = open_smile['-C', config, '-I', input_csv, '-O', output_csv]
    return exec_args & TEE(timeout=timeout)


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


def run_catch_fail(function, *args, autotries=-1, failedon=None):
    """
    This function will run the function provided, and catch unexpected error codes. It will automatically try the
    function again for the number of provided autotries, then wait for the user to respond.
    """
    while True:
        try:
            return function(*args)
        except (ProcessExecutionError, ProcessTimedOut) as e:
            if autotries != 0:
                print('Failed! Autotrying...')
                autotries -= 1
                time.sleep(10)
                continue
            if failedon is not None:
                print('Failed on: ' + failedon)
            print('STDOUT: ', e.stdout)
            with colors.red:
                print('STDERR: ', e.stderr)
                print('ERRNO: ', e.errno)
            if cli.terminal.ask('Try again'):
                continue
            if cli.terminal.ask('Skip?'):
                break
            print('Exiting...')
            exit(-1)


def loopandsmile(toplevellist, config: Path, preserve=False, savemat=True, savepick=True):
    """
    Loop through the assumed directory structure and run opensmile on each epoch.
    :param toplevellist:
    :param config:
    :param preserve:
    :param savemat:
    :return:
    """

    for subject in toplevellist:
        for experiment in subject.iterdir():
            print('Subject:', subject, 'Experiment:', experiment)
            for epoch in [x for x in experiment.iterdir() if x.suffix == '.csv']:
                # make a back up, might break under windows?
                if not preserve:
                    shutil.copy(epoch.as_posix(), epoch.with_suffix('.bak').as_posix())
                shutil.copy(epoch.as_posix(), epoch.with_suffix('.temp').as_posix())

                # Extract features to the same name as input file
                opensmile_extract_features(config.absolute().as_posix(),
                                           epoch.absolute().with_suffix('.temp').as_posix(),
                                           epoch.absolute().as_posix(), timeout=30)
                epoch.with_suffix('.temp').unlink()

                if savepick:
                    print('Saving', subject, experiment, epoch.stem, 'as numpy file...')
                    x = pd.read_csv(epoch, delimiter=';').as_matrix()
                    np.save(str(epoch.absolute().with_suffix('.npy')), x)
                    print('Saved:', epoch.absolute().with_suffix('.npy'))

                if savemat:
                    mat_compress(epoch.absolute().as_posix(), epoch.absolute().with_suffix('.mat').as_posix(),
                                 timeout=10)

                if savemat or savepick:
                    # remove original
                    epoch.unlink()


def cart2spherical(x, y, z):
    xy = np.hypot(x, y)
    return np.arctan2(y, x), np.arctan2(xy, z), np.hypot(xy, z)


def spher2cart(theta, phi, rho):
    xy = rho*np.sin(phi)
    return xy*np.cos(theta), xy*np.sin(theta), rho*np.cos(phi)


# TODO further verify calculations
def azimuthal_projection(pts, coordsystem='cart'):
    """
    Projects a 3D representation of channel locations into a 2D cartesian representation
    :param pts: A tuplish arguemnt of length three conforming to the coordsystem
    :param coordsystem: Either 'cart' or 'sphere'
    :return: x, y cartesian representation of the point.
    """
    if len(pts) != 3:
        raise TypeError('Points provided returned length,', len(pts),
                        'Point must be tupleish sequence of 3 spherical coordinates.')

    if coordsystem == 'cart':
        [theta, phi, rho] = cart2spherical(**pts)
    elif coordsystem == 'sphere':
        theta, phi, rho = pts
    else:
        raise TypeError('Coordinate system provided', coordsystem, 'is not supported')

    # Chop off what should be z values of ~0
    # return spher2cart(theta, np.pi/2, np.pi/2 - phi)[:2]
    return spher2cart(theta, phi, rho)[:2]


def chan2spatial(chanlocfile, coordsystem='sphere', channels=(range(36, 187))):
    """
    Provides a transformation to convert the channel locations into a 2D spatial tensor
    :param chanlocfile: File to load the channel locations from.
    :param coordsystem: The coordinate system the file uses, should be 'cart' or 'sphere'
    :param channels: Which channels to keep
    :param grid: Dimensions length the x and y axis can take, to keep a consistent model size
    :return: Matrix to apply to incoming tensors of the form [samples x ... x channels] into
    [samples x ... x X_loc x Y_loc]
    """
    if not isinstance(chanlocfile, Path):
        chanlocfile = Path(chanlocfile)

    if not chanlocfile.exists():
        FileNotFoundError('Could not find channel location file', chanlocfile)

    columns = {'sphere': ['sph_theta', 'sph_phi', 'sph_radius'], 'cart': ['X', 'Y', 'Z']}

    chans = pd.read_csv(chanlocfile).as_matrix(columns[coordsystem])[channels]
    locs = np.apply_along_axis(azimuthal_projection, 1, chans, coordsystem=coordsystem)

    # shift the x and y positions so they are always centered, and span no further than the size of the grid
    locs -= locs.mean(axis=0)
    locs /= abs(locs).max()

    # Scale to grid size, and round to integers
    # locs *= grid/2
    # locs += grid/2
    # locs = np.round(locs).astype('int32')

    # Create transformation matrix
    # xform = np.zeros((grid, grid))
    # for i, l in enumerate(locs):
    #     xform[i, l[0], l[1]] = 1

    return locs


def pink_noise(shape, cutoff_f=None):
    """
    Create noise that has spectral activity decrease with 1/f
    :param shape:
    :param cutoff_f: enforce a cut-off frequency
    :return:
    """
    uneven = shape[0] % 2
    x = np.random.randn(shape[0]//2 + 1 + uneven, *shape[1:]) + 1j*np.random.rand(shape[0]//2 + 1 + uneven, *shape[1:])
    S = np.sqrt(np.arange(x.shape[0]) + 1.)[:, np.newaxis]
    x[0, :] = 0.0
    y = np.real(np.fft.irfft(x/S, axis=0))
    if uneven:
        y = y[:-1]
    return y


# def interpolated_image(weights, layout='../CTF151.npy', gridsize=100):
#
#

# def animated(x, samplefreq=200):
#     """
#     Provides an animated plot of the data provided in x
#     :param x: A 3D tensor made up of (Samples) x (X) x (Y)
#     :param samplefreq: The frequency at which the samples are taken
#     :return:
#     """
#     fig = plt.figure()
#     im = plt.imshow(x[0, :, :], animated=True)
#
#     def plotsample(t):
#         im.set_array(x[t, :, :])
#
#     a = anim.FuncAnimation(fig, plotsample, frames=range(x.shape[0]), interval=1000/samplefreq)
#
#     plt.show()

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in np.itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
