# THIS WAS ONLY USED AS A TEMPORARY HACK TO OBTAIN A CSV FORMATTED DATASET, IT IS LIKELY TO CAUSE ROUNDING ERRORS!!
# USE WITH CAUTION
import glob
import numpy as np
from scipy.io import loadmat
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Convert all mat files in the provided directory back into csv files.\n'
                'This runs recursively through all subdirectories'
)
parser.add_argument('toplevel', help='Top level of data directory')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.toplevel[-1] != '/': args.toplevel += '/'

    for file in glob.iglob(args.toplevel+'**/*.mat'):
        print('Converting:', file)
        data = loadmat(file)
        if 'header' not in data or 'features' not in data:
            print('ERROR!')
            print('File:', file)
            print('Not formatted correctly. Skipping...')
            print()
            continue

        header = ','.join(np.concatenate(data['header'][0]))
        np.savetxt(str(Path(file).with_suffix('.csv')), data['features'], header=header, comments='', delimiter=',')