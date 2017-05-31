import numpy as np
from pathlib import Path
from scipy.signal import cwt, ricker
from scipy.io import savemat

#TODO pywavelets library and dwt rather than cwtaa

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

    outfile = Path(outfile)
    if outfile.exists():
        print('Warning: Overwriting output file', outfile)

    numrows = csvin.shape[0]
    if numrows != NUM_OF_CHANNELS:
        print('WARNING: in file:', infile)
        print(numrows, '!=151 MEG channels detected in file')

    outmat = np.empty((*csvin.shape, numscales), dtype=np.float32)

    # print('Extracting wavelet coefficients...')
    for row in range(numrows):
        outmat[row, :, :] = cwt(csvin[row, :], ricker, SCALES).T

    print('Saving to: ', outfile, '...')
    if outfile.suffix == 'mat':
        savemat(str(outfile), {MAT_VAR_NAME: outmat}, do_compression=True)
    elif outfile.suffix == 'npy':
        np.save(str(outfile), outmat)
    elif outfile.suffix == '':
        np.save(str(outfile.with_suffix('npy')), outmat)
    else:
        raise NameError('Unknown suffix for outfile: "{0}", will not proceed'.format(outfile.suffix))
    print('Saved.')

