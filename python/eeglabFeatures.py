from utils import *
import argparse
from pathlib import Path
from threading import Thread

# Command line arguments
parser = argparse.ArgumentParser(
    description='Extract openSMILE features from eeglab extracted data'
)
parser.add_argument('toplevel', type=str, help='Top level of where the study and datasets are located')
parser.add_argument('--meg-config', '-mc', default=DEFAULT_MEG_CONFIG, type=str,
                    help='openSMILE config file to use for MEG')
parser.add_argument('--audio-config', '-ac', default=DEFAULT_ACOUSTIC_CONFIG, type=str,
                    help='openSMILE config file to use for audio')
parser.add_argument('--save-mat', '-z', action='store_true',
                    help='Save the output as a compressed .mat file')
parser.add_argument('--save-pickle', help='Save the output as a pickled numpy array', action='store_true')
parser.add_argument('--no-preserve', '-n', action='store_true',
                    help='Do not preserve the original activations')
parser.add_argument('--no-meg', action='store_true', help='Do not extract MEG features')
parser.add_argument('--no-audio', action='store_true', help='Do not extract audio features')

parser.add_argument('--threads', type=int, default=1, choices=range(1, 5), help='Number of threads to use to process')


if __name__ == '__main__':
    args = parser.parse_args()

    toplevel = Path(args.toplevel)
    meg_config = Path(args.meg_config)
    audio_config = Path(args.audio_config)

    if not toplevel.is_dir():
        print('Could not find top level path: ', args.toplevel)
        parser.print_help()
        exit(-1)

    if not meg_config.is_file() or not audio_config.is_file():
        print('Could not find config file')
        parser.print_help()
        exit(-1)

    if args.threads > 1:
        print('Starting ', args.threads, 'threads...')

    megfiles = [(x / 'MEG') for x in toplevel.iterdir() if x.is_dir() and 'CC' in x.stem]
    n = round(len(megfiles)/args.threads)+1
    megbuckets = [megfiles[i :i + n] for i in range(0, len(megfiles), n)]

    threads = []
    for i in range(1, args.threads):
        threads.append(Thread(None, run_catch_fail, None, (loopandsmile, megbuckets[i], meg_config, args.no_preserve,
                                                           args.save_mat, args.save_pickle)))
        if not args.no_meg:
            threads[-1].start()

    if not args.no_meg:
        run_catch_fail(loopandsmile, megbuckets[0], meg_config,
                     args.no_preserve, args.save_mat, args.save_pickle)

    if not args.no_audio:
        run_catch_fail(loopandsmile, [(x / 'Audio') for x in toplevel.iterdir() if x.is_dir() and 'CC' in x.stem],
                       audio_config, args.no_preserve, args.save_mat, autotries=-1)
