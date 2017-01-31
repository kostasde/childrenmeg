from utils import *
import argparse
from pathlib import Path
import shutil

# Command line arguments
parser = argparse.ArgumentParser(
    description='Extract openSMILE features from eeglab extracted data'
)
parser.add_argument('toplevel', type=str, help='Top level of where the study and datasets are located')
parser.add_argument('--config', '-c', default=DEFAULT_MEG_CONFIG, type=str, help='openSMILE config file to use')
parser.add_argument('--save-mat', '-z', action='store_true', help='Save the output as a compressed .mat file')
parser.add_argument('--no-preserve', '-n', action='store_true', help='Do not preserve the original activations')

if __name__ == '__main__':
    args = parser.parse_args()

    toplevel = Path(args.toplevel)
    config = Path(args.config)

    if not toplevel.is_dir():
        print('Could not find top level path: ', args.toplevel)
        parser.print_help()
        exit(-1)

    if not config.is_file():
        print('Could not find config file: ', args.config)
        parser.print_help()
        exit(-1)

    # For each subject directory
    for subject in [(x / 'MEG') for x in toplevel.iterdir() if x.is_dir()]:
        for experiment in subject.iterdir():
            for epoch in experiment.iterdir():
                # make a back up, might break under windows?
                if not args.no_preserve:
                    shutil.copy(epoch.as_posix(), epoch.with_suffix('.bak').as_posix())
                shutil.copy(epoch.as_posix(), epoch.with_suffix('.temp').as_posix())

                # Extract features to the same name as input file
                opensmile_extract_features(config.absolute().as_posix(),
                                           epoch.absolute().with_suffix('.temp').as_posix(),
                                           epoch.absolute().as_posix())
                epoch.with_suffix('.temp').unlink()

                if args.save_mat:
                    mat_compress(epoch.absolute().as_posix(), epoch.absolute().with_suffix('.mat').as_posix())
                    # remove original
                    epoch.unlink()
