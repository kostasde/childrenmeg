import argparse
import shutil
from utils import *
from pathlib import Path
from threading import Thread


def handle_open_smile(args):
    meg_config = Path(args.meg_config)
    audio_config = Path(args.audio_config)

    if not meg_config.is_file() or not audio_config.is_file():
        print('Could not find config file')
        parser.print_help()
        exit(-1)

    if args.threads > 1:
        print('Starting ', args.threads, 'threads...')

    megfiles = [(x / 'MEG') for x in args.toplevel.iterdir() if x.is_dir() and 'CC' in x.stem]
    n = round(len(megfiles) / args.threads) + 1
    megbuckets = [megfiles[i:i + n] for i in range(0, len(megfiles), n)]

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
        run_catch_fail(loopandsmile, [(x / 'Audio') for x in args.toplevel.iterdir() if x.is_dir() and 'CC' in x.stem],
                       audio_config, args.no_preserve, args.save_mat, autotries=-1)


def handle_restore(args):
    for f in args.toplevel.glob('**/*.bak'):
        if not args.silent:
            print('Renaming {0} to {1}'.format(f, f.with_suffix('.csv')))
        shutil.move(str(f), str(f.with_suffix('.csv')))


def handle_hide(args):
    for f in args.toplevel.glob('**/*.csv'):
        if not args.silent:
            print('Renaming {0} to {1}'.format(f, f.with_suffix('.bak')))
        shutil.move(str(f), str(f.with_suffix('.bak')))


def handle_clean(args):
    extensions = [x.strip(' .') for x in str(args.extension).split(',')]
    for suff in extensions:
        suff = str('.'+suff)
        print('\n\nDeleting files with suffix: ', suff)
        for f in args.toplevel.glob('**/*.{0}'.format(suff)):
            if not args.silent:
                print('Removing file: ', f)
            f.unlink()


# Command line arguments
parser = argparse.ArgumentParser(
    description='Manage the eeglab dataset directory structure. Extract features and clean files'
)

subparsers = parser.add_subparsers()

os_parser = subparsers.add_parser('os', help='Extract openSMILE features from eeglab extracted data')
os_parser.set_defaults(func=handle_open_smile)
os_parser.add_argument('toplevel', type=str, help='Top level of where the study and datasets are located')
os_parser.add_argument('--meg-config', '-mc', default=DEFAULT_MEG_CONFIG, type=str,
                    help='openSMILE config file to use for MEG')
os_parser.add_argument('--audio-config', '-ac', default=DEFAULT_ACOUSTIC_CONFIG, type=str,
                    help='openSMILE config file to use for audio')
os_parser.add_argument('--save-mat', '-z', action='store_true',
                    help='Save the output as a compressed .mat file')
os_parser.add_argument('--save-pickle', help='Save the output as a pickled numpy array', action='store_true')
parser.add_argument('--no-preserve', '-n', action='store_true',
                    help='Do not preserve the original activations')
os_parser.add_argument('--no-meg', action='store_true', help='Do not extract MEG features')
os_parser.add_argument('--no-audio', action='store_true', help='Do not extract audio features')
os_parser.add_argument('--threads', type=int, default=1, choices=range(1, 5), help='Number of threads to use to process')

res_parser = subparsers.add_parser('restore', help='Restore ".csv" files from the ".bak" backups.')
res_parser.add_argument('toplevel', type=str, help='Top level of where the study and datasets are located')
res_parser.add_argument('--silent', '-s', help='', action='store_true')
res_parser.set_defaults(func=handle_restore)

hid_parser = subparsers.add_parser('hide', help='Hide ".csv" files as ".bak" backups.')
hid_parser.add_argument('toplevel', type=str, help='Top level of where the study and datasets are located')
hid_parser.add_argument('--silent', '-s', help='', action='store_true')
hid_parser.set_defaults(func=handle_hide)

clean_parser = subparsers.add_parser('clean', help='Clean certain file types from the directory structure.')
clean_parser.add_argument('toplevel', type=str, help='Top level of where the study and datasets are located')
clean_parser.add_argument('extension', type=str, help='Type of extension to clear in a comma separated list. '
                                                      'eg. csv, bak')
clean_parser.set_defaults(func=handle_clean)


if __name__ == '__main__':
    args = parser.parse_args()

    # Toplevel exists for all commands
    args.toplevel = Path(args.toplevel)
    if not args.toplevel.is_dir():
        print('Could not find top level path: ', args.toplevel)
        parser.print_help()
        exit(-1)

    # Run the actual command
    args.func(args)

