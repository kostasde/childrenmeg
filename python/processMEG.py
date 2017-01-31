import sys

MATLAB_DIR = '../MATLAB/'
CONFIG_DIR = '../configs/'
LIBS_DIR = '../libs/'
sys.path.append(LIBS_DIR)

from .utils import *

from plumbum import cli, local, colors, TEE
from plumbum.cli import progress, terminal
from plumbum.commands.processes import ProcessExecutionError


class ProcessMEG(cli.Application):
    """
    This script stitches together MATLAB and OPENSmile
    """
    # Basic set of switches/flags
    verbose = cli.Flag(["-v", "--verbose"], help="Enable verbose output")
    meg_conf = cli.SwitchAttr(["-C", '--meg-config'], str, default=DEFAULT_MEG_CONFIG,
                              help="Specify the configuration file for openSMILE to process the MEG signals")
    acoustic_conf = cli.SwitchAttr(['-A', '--acoustic-config'], str, default=DEFAULT_ACOUSTIC_CONFIG,
                                   help="Specify the configuration file for openSMILE to process the Acoustic signals")
    meg_dir = cli.SwitchAttr('--meg-dir', str, default=DEFAULT_MEG_DIR, help='Top directory for all the raw MEG data')
    dest_dir = cli.SwitchAttr('--dest-dir', str, default=finish_mat_dir,
                              help='The destination of the completed analysis')
    subjects = cli.SwitchAttr(['-S', '--subjects'], int, list=True, excludes=['--all', '--remaining'])

    # Special case switches/flags
    all_subjects = cli.Flag(['--all'], help="Perform analysis on all the subject directories available")
    remaining_subjects = cli.Flag(['--remaining'], help="Perform analysis on subjects that are remaining")

    update = cli.Flag(['-U', '--update'], help="Overwrite existing data")
    only_acoustic = cli.Flag(['--only-acoustic'], help="Only output acoustic features", excludes=['--only-meg', '--only-wavelet'])
    only_meg = cli.Flag(['--only-meg'], help="Only output meg features", excludes=['--only-acoustic', '--only-wavelet'])
    only_wavelets = cli.Flag(['--only-wavelet'], help='Extract MEG wavelet features', excludes=['--only-acoustic', '--only-meg'])

    @staticmethod
    def run_catch_fail(self, function, *args, autotries=2, failedon=None):
        """
        This function will run the function provided, and catch unexpected error codes. It will automatically try the
        function again for the number of provided autotries, then wait for the user to respond.
        """
        while True:
            try:
                stdout = function(*args)
                if self.verbose:
                    print(stdout)
                    break
            except ProcessExecutionError as e:
                if autotries != 0:
                    autotries -= 1
                    time.sleep(60)
                    continue
                if failedon is not None:
                    print('Failed on: ' + failedon)
                print('STDOUT: ', e.stdout)
                with colors.red:
                    print('STDERR: ', e.stderr)
                    print('ERRNO: ', e.errno)
                if terminal.ask('Try again'):
                    continue
                if terminal.ask('Skip?'):
                    break
                print('Exiting...')
                exit(-1)

    def main(self):
        self.dest_dir = local.path(self.dest_dir)
        self.meg_dir = local.path(self.meg_dir)
        self.meg_conf = local.path(self.meg_conf)

        # Run some checks on the provided arguments
        if not self.meg_dir.is_file():
            print("Provided MEG source directory:")
            with colors.red:
                print(self.meg_dir)
            print("Does not exist")

        if not self.meg_conf.is_file():
            print("Provided config file:")
            with colors.red:
                print(self.meg_dir)
            print("Does not exist")

        if not self.dest_dir.is_dir():
            if terminal.ask("Destination directory: %s does not exist, create" % str(self.dest_dir), True):
                self.dest_dir.mkdir()
            else:
                return -1

        if self.verbose:
            print("Verbose mode")
            print("Raw MEG found at:", self.meg_dir)
            print("Using config file: ", self.meg_conf)
            with colors.green:
                print("Outputting to: ", self.dest_dir)
            self.completed_subjects = find_completed(self.dest_dir).keys()
            print("Completed subjects: ", str(self.completed_subjects))
            if self.update:
                with colors.yellow:
                    print('WARNING: Overwriting existing data for subjects')
            if self.only_acoustic:
                print('Extracting acoustic data')

        if self.all_subjects or self.remaining_subjects:
            # determine remaining
            # FIXME hardcoded path is stupid
            toplevel = local.path('/media/demetres/My Book/LizPang2015/Children')
            if not toplevel.exists():
                print('No subjects available.')
                return

            pattern = re.compile("^CC[0-9]+(-[0-9]+)?")

            self.subjects = []
            self.subjects += [subject.name for subject in toplevel if pattern.fullmatch(subject.name) is not None]

            if self.remaining_subjects and self.update:
                print('Warning: Remaining and update specified, going through all...')
            if self.all_subjects and not self.update:
                # easiest way to ensure that all get processed
                self.update = True

        if not self.update:
            self.subjects = [x for x in self.subjects if x not in self.completed_subjects]
        if self.verbose:
            print('Subjects to process:', self.subjects)
        subjectsprogress = progress.ProgressAuto(self.subjects, body=True)
        subjectsprogress.start()
        # Go through each subject remaining
        for subject in self.subjects:
            with colors.green:
                print('Subject: ', subject)

            # First extract the data for opensmile processing, if not already done
            if preproc_csvs_dir.is_dir() and (len(preproc_csvs_dir.list()) > 0) and \
                    terminal.ask('Previously pre-processed data found, would you like to proceed ' +
                                         'presuming it is for subject ' + str(subject) +
                                                 ' (%d tests found)' % len(preproc_csvs_dir.list())):
                pass
            else:
                print('Preprocessing raw MEG for opensmile feature extraction...')
                stdout = preprocess(subject, audioonly=self.only_acoustic)
                if self.verbose:
                    print(stdout[1])
                with colors.red:
                    if stdout[0] != 0 or'FAILED' in stdout[1] or 'Complete.' not in stdout[1]:
                        print('Failed to extract data for subject:', subject)
                        print('Skipping...')
                        continue
                with colors.green:
                    print("Completed outputting preprocessed csv files.")

            # Run extractions on all the preprocessed data
            for test in preproc_csvs_dir:
                # Ignore the absolute path for just the last directory title
                test_label = test.name.split('/')[-1]
                if self.verbose:
                    print("Test: ", test_label)

                directory = '/'.join([self.dest_dir, subject, test_label, ''])

                if not self.only_acoustic:
                    dest_meg_dir = directory + 'MEG/'
                    mkdir(dest_meg_dir)
                    temp_meg_dir = test.glob('MEG').pop()
                else:
                    temp_meg_dir = None

                if not self.only_meg or not self.only_wavelets:
                    dest_acoustic_dir = directory + 'Acoustic/'
                    mkdir(dest_acoustic_dir)
                    temp_acoustic_dir = test.glob('Acoustic').pop()
                else:
                    temp_acoustic_dir = None

                temp_location = local.path(preproc_csvs_dir + '/temp.csv')

                # MEG processing
                if temp_meg_dir is not None and temp_meg_dir.is_dir() and \
                        not (self.only_acoustic or self.only_wavelets):
                    print('Extracting MEG features...')
                    for epoch in progress.Progress(temp_meg_dir, has_output=True):
                        if self.verbose:
                            print(time.strftime('%X %x %Z'))

                        # Check if the epoch has already been processed
                        if not self.update:
                            if epoch.name.split('.')[0] in [x.name.split('.')[0] for x in local.path(dest_meg_dir)]:
                                print("Already exists, skipping...")
                                continue

                        ProcessMEG.run_catch_fail(self, opensmile_extract_features, self.meg_conf, epoch,
                                                  temp_location, failedon=epoch)
                        ProcessMEG.run_catch_fail(self, mat_compress, temp_location,
                                                  dest_meg_dir+epoch.name.split('.')[0]+'.mat', failedon=epoch)
                        temp_location.delete()

                # Acoustic Processing
                if temp_acoustic_dir is not None and temp_acoustic_dir.is_dir() and \
                        not (self.only_meg or self.only_wavelets):
                    print('Extracting Audio features...')
                    for epoch in progress.Progress(temp_acoustic_dir, has_output=True):
                        if self.verbose:
                            print(time.strftime('%X %x %Z'))

                        # Check if the epoch has already been processed
                        if not self.update:
                            if epoch.name.split('.')[0] in \
                                    [x.name.split('.')[0] for x in local.path(dest_acoustic_dir)]:
                                print("Already exists, skipping...")
                                continue

                        ProcessMEG.run_catch_fail(self, opensmile_extract_features, self.acoustic_conf, epoch,
                                                  temp_location, failedon=epoch)
                        ProcessMEG.run_catch_fail(self, mat_compress, temp_location,
                                                  dest_acoustic_dir+epoch.name.split('.')[0] + '.mat', failedon=epoch)
                        temp_location.delete()

                # Wavelet Processing
                if temp_meg_dir is not None and temp_meg_dir.is_dir() and \
                        not (self.only_acoustic or self.only_meg):
                    print("Extracting wavelet coefficients...")
                    for epoch in progress.Progress(temp_meg_dir, has_output=True):
                        if self.verbose:
                            print(time.strftime('%X %x %Z'))

                        outname = epoch.name.split('.')[0] + '_wavelets'

                        # Check if the epoch has already been processed
                        if not self.update:
                            if outname in [x.name.split('.')[0] for x in local.path(dest_meg_dir)]:
                                print("Already exists, skipping...")
                                continue

                        ProcessMEG.run_catch_fail(self, extractwaveletcoef, epoch, dest_meg_dir+outname, failedon=epoch)

                test.delete()

            # Increment
            with colors.green & colors.bold:
                subjectsprogress.increment()

        subjectsprogress.done()


if __name__ == '__main__':
    ProcessMEG.run()
