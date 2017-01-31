import numpy as np
from scipy.signal import cwt, ricker
from scipy.io import savemat

NUM_OF_CHANNELS = 151
SCALES = np.arange(4, 32)
MAT_VAR_NAME = 'WaveletFeatures'


def extractwaveletcoef(infile, outfile):
    """
    This will extract wavelet coefficients from the specified file.
    Uses ricker wavelet, may not be the best choice...
    :param infile: location of csv formatted file
    :param outfile: destination for a file, will be formatted as mat file
    """
    csvin = np.loadtxt(infile, delimiter=',').astype(np.float32).T
    numscales = len(SCALES)

    numrows = csvin.shape[0]
    if numrows != NUM_OF_CHANNELS:
        print('WARNING: in file:', infile)
        print(numrows, '!=151 MEG channels detected in file')

    outmat = np.empty((*csvin.shape, numscales), dtype=np.float32)

    # print('Extracting wavelet coefficients...')
    for row in range(numrows):
        outmat[row, :, :] = cwt(csvin[row, :], ricker, SCALES).T

    print('Saving to: ', outfile, '...')
    savemat(outfile, {MAT_VAR_NAME: outmat}, do_compression=True)
    print('Saved.')

