import pickle
import numpy as np
from time import sleep

from keras.preprocessing.image import Iterator as KerasDataloader

# from functools import lru_cache
from cachetools import cached, RRCache
from abc import ABCMeta, abstractmethod
from scipy.io import loadmat
from pathlib import Path

PREV_EVAL_FILE = 'preprocessed.pkl'

SUBJECT_TABLE = 'subject_table.mat'
SUBJECT_STRUCT = 'subject_struct.mat'

# Suffix in use
LOAD_SUFFIX = '.npy'

# Headers that we will import for the subjects
HEADER_ID = 'ID'
HEADER_AGE = 'age'
HEADER_SEX = 'sex'

# Test types
TEST_PDK = 'PDK'
TEST_PA = 'PA'
TEST_VG = 'VG'
TEST_MO = 'MO'

# todo make this less ugly
MEG_COLUMNS = Path('realcol.npy')
if MEG_COLUMNS.exists():
    print('Found MEG Index')
    megind = np.load(str(MEG_COLUMNS))
else:
    megind = None


def random_slices(dataset: np.ndarray, sizeofslices=(0.1, 0.9)):
    """
    Randomly segments the data into slices of size _sizeofslices_
    :param dataset: The dataset to perform this, splits the rows
    :param sizeofslices: tuple of size of slices, does not need to sum to 1
    :return: len(sizeofslices) list of ndarrays
    """
    to_return = []
    # Normalize to sum to 1
    sizeofslices = np.array(sizeofslices) / np.sum(sizeofslices)
    ind = np.arange(dataset.shape[0])
    for slice in sizeofslices:
        slice_ind = np.random.choice(ind, replace=False, size=int(slice * dataset.shape[0]))

        # Create masks to select the data
        mask = np.full(dataset.shape, False, bool)
        mask[slice_ind, :] = True
        to_return.append(dataset[mask].reshape((len(slice_ind), -1)))

    return to_return


def zscore(l, nanprevention=1e-10):
    l = np.float32(l)
    return (l - np.mean(l, axis=0)) / (np.std(l, axis=0) + nanprevention)


def parsesubjects(subjecttable):
    """
    Given the path to the subject table, read and return a dictionary for the subjects
    :param subjecttable: String path, or Path object
    :return:
    """
    subject_dictionary = {}
    if isinstance(subjecttable, Path):
        struct = loadmat(str(subjecttable), squeeze_me=True)['subject_struct']
    elif isinstance(subjecttable, str):
        struct = loadmat(subjecttable, squeeze_me=True)['subject_struct']
    else:
        raise TypeError('Path to subject table not a recognized type')

    idlist = list(struct[HEADER_ID])
    agelist = list(struct[HEADER_AGE])
    sexlist = list(struct[HEADER_SEX])

    for subject in idlist:
        index = idlist.index(subject)
        subject_dictionary[subject] = \
            {HEADER_AGE: agelist[index], HEADER_SEX: sexlist[index]}

    return subject_dictionary


class CrossValidationComplete(Exception):
    pass


class SubjectFileLoader(KerasDataloader):
    """
    This implements a thread-safe loader based on the examples related to Keras ImageDataGenerator

    This class is a somewhat generic generator used for the training, validation and test datasets of the Dataset classes.
    """

    cache = RRCache(32768)

    # @lru_cache(maxsize=8192)
    @cached(cache)
    def get_features(self, path_to_file):
        """
        Loads arrays from file, and returned as a flattened vector, cached to save some time
        :param path_to_file:
        :return: numpy vector
        """
        # return loadmat(path_to_file, squeeze_me=True)['features'].ravel()
        l = np.load(path_to_file[0])
        #
        # if self.megind is not None:
        #     l = l[:, self.megind].squeeze()

        # l = zscore(l)
        if self.flatten:
            l = l.ravel()

        return l

    def __init__(self, x, longest_vector, subject_hash, target_cols, batchsize=-1,
                 flatten=True, shuffle=True, seed=None):

        self.x = np.asarray(x)
        self.longest_vector = longest_vector
        self.subject_hash = subject_hash
        self.targets = target_cols
        self.flatten = flatten

        if batchsize < 0:
            batchsize = x.shape[0]

        super().__init__(x.shape[0], batchsize, shuffle=shuffle, seed=seed)

    def _load(self, index_array, batch_size):
        x = np.zeros([batch_size, self.longest_vector])
        y = np.zeros([batch_size, 1])

        for i, row in enumerate(index_array):
            ep = tuple(str(f) for f in self.x[row, 1:])
            temp = self.get_features(ep)
            x[i, :len(temp)] = temp
            y[i, :] = np.array([self.subject_hash[self.x[row, 0]][column] for column in self.targets])

        return x[~np.isnan(x).any(axis=1)], y[~np.isnan(x).any(axis=1)]

    def __next__(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        return self._load(index_array, current_batch_size)


class BaseDataset:

    DATASET_TARGETS = [HEADER_AGE]
    NUM_BUCKETS = 10

    # Not sure I like this...
    GENERATOR = SubjectFileLoader

    def __init__(self, toplevel, PDK=True, PA=True, VG=True, MO=False, batchsize=2):
        self.toplevel = Path(toplevel)
        # some basic checking to make sure we have the right directory
        if not self.toplevel.exists() or not self.toplevel.is_dir():
            raise NotADirectoryError("Provided top level directory is not directory")
        self.subject_hash = parsesubjects(self.toplevel / SUBJECT_STRUCT)

        self.batchsize = batchsize
        self.traindata = None

        tests = []
        # Assemble which experiments we are going to be using
        if PDK:
            tests.append(TEST_PDK)
        if PA:
            tests.append(TEST_PA)
        if VG:
            tests.append(TEST_VG)
        if MO:
            tests.append(TEST_MO)

        if self.preprocessed_file in [x.name for x in self.toplevel.iterdir() if not x.is_dir()]:
            with (self.toplevel / self.preprocessed_file).open('rb') as f:
                print('Loaded previous preprocessing!')
                self.buckets, self.longest_vector, self.slice_length,\
                self.testpoints, self.loaded_subjects = pickle.load(f)
                # list of subjects that we will use for the cross validation
                # self.leaveoutsubjects = np.unique(self.datapoints[:, 0])
                # Todo: warn/update pickled file if new subjects exist
        else:
            print('Preprocessing data...')

            self.loaded_subjects, self.longest_vector, self.slice_length = self.files_to_load(tests)

            # TODO: Need more stable way of doing test subjects, should be set outside
            testsubjects = np.random.choice(list(self.loaded_subjects.keys()),
                                            int(len(self.loaded_subjects)/10), replace=False)
            self.testpoints = np.array([item for x in testsubjects for item in self.loaded_subjects[x]])
            print('Subjects used for testing:', testsubjects)

            datapoint_ordering = sorted(self.loaded_subjects, key=lambda x: -len(self.loaded_subjects[x]))
            self.buckets = [[] for x in range(self.NUM_BUCKETS)]
            # Fill the buckets up and down
            for i in range(len(datapoint_ordering)):
                if int(i / self.NUM_BUCKETS) % 2:
                    index = self.NUM_BUCKETS - (i % self.NUM_BUCKETS) - 1
                    self.buckets[int(index)].extend(self.loaded_subjects[datapoint_ordering[i]])
                else:
                    self.buckets[int(i % self.NUM_BUCKETS)].extend(self.loaded_subjects[datapoint_ordering[i]])

            # self.datapoints = self.loaded_subjects[np.in1d(self.loaded_subjects[:, 0], self.leaveoutsubjects), :]

            # # leave out 10% of randomly selected data for test validation
            # self.testpoints, self.loaded_subjects = self.random_slices(self.datapoints, (0.1, 0.9))

            with (self.toplevel / self.preprocessed_file).open('wb') as f:
                pickle.dump((self.buckets, self.longest_vector, self.slice_length,
                             self.testpoints, self.loaded_subjects), f)
                # numpoints = self.datapoints.size[0]
                # ind = np.arange(numpoints)
                # ind = np.random.choice(ind, replace=False, size=int(0.2*numpoints)

        self.next_leaveout(force=0)


    @property
    @abstractmethod
    def modality_folders(self) -> list:
        """
        Subclasses must implement this so that it reports the name of the folder(s) to find experiments, once in the
        subject folder.
        :return:
        """
        pass

    def files_to_load(self, tests):
        """
        This should be implemented by subclasses to specify what files
        :param tests: The type of tests that should make up the dataset
        :return: A dictionary for the loaded subjects
        :rtype: tuple
        """
        longest_vector = -1
        slice_length = -1
        loaded_subjects = {}
        for subject in [x for x in self.toplevel.iterdir() if x.is_dir() and x.name in self.subject_hash.keys()]:
            print('Loading subject', subject.stem, '...')
            loaded_subjects[subject.stem] = []
            for experiment in [t for e in self.modality_folders if (subject / e).exists()
                               for t in (subject / e).iterdir() if t.name in tests]:

                for epoch in [l for l in experiment.iterdir() if l.suffix == LOAD_SUFFIX]:
                    try:
                        # f = loadmat(str(epoch), squeeze_me=True)
                        f = np.load(str(epoch))
                        # slice_length = max(slice_length, len(f['header']))
                        # longest_vector = max(longest_vector,
                        #                           len(f['features'].reshape(-1)))
                        slice_length = max(slice_length, f.shape[1])
                        longest_vector = max(longest_vector, f.shape[0]*f.shape[1])
                        loaded_subjects[subject.stem].append((subject.stem, epoch))
                    except Exception:
                        print('Warning: Skipping file, error occurred loading:', epoch)

        return loaded_subjects, longest_vector, slice_length

    @property
    @abstractmethod
    def preprocessed_file(self):
        pass

    def next_leaveout(self, force=None):
        """
        Moves on to the next group to leaveout.
        :return: Number of which leaveout, `None` if complete
        """
        if force is not None:
            self.leaveout = force

        if self.leaveout == self.NUM_BUCKETS:
            print('Have completed cross-validation')
            self.leaveout = None
            # raise CrossValidationComplete
            return self.leaveout

        # Select next bucket to leave out as evaluation
        self.eval_points = np.array(self.buckets[self.leaveout])

        # Convert the remaining buckets into one list
        self.traindata = np.array(
            [item for sublist in self.buckets for item in sublist if self.buckets.index(sublist) != self.leaveout]
        )

        self.leaveout += 1

        return self.leaveout

    def current_leaveout(self):
        return self.leaveout

    def sanityset(self, fold=5, batchsize=None):
        """
        Provides a generator for a small subset of data to ensure that the model can train to it
        :return: 
        """
        if batchsize is None:
            batchsize = self.batchsize

        return self.GENERATOR(np.array(self.buckets[fold]), self.longest_vector, self.subject_hash,
                              self.DATASET_TARGETS, batchsize)

    def trainingset(self, batchsize=None):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        if self.traindata is None:
            raise AttributeError('No fold initialized... Try calling next_leaveout')

        return self.GENERATOR(self.traindata, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              batchsize)

    def evaluationset(self, batchsize=None):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return self.GENERATOR(self.eval_points, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              batchsize)

    def testset(self, batchsize=None):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return self.GENERATOR(self.testpoints, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              batchsize)

    def inputshape(self):
        return self.longest_vector

    def outputshape(self):
        return len(self.DATASET_TARGETS)


class BaseDatasetAgeRanges(BaseDataset, metaclass=ABCMeta):

    # Age Ranges, lower inclusive, upper exclusive
    AGE_4_5 = (4, 6)
    AGE_6_7 = (6, 8)
    AGE_8_9 = (8, 10)
    AGE_10_11 = (10, 12)
    AGE_12_13 = (12, 14)
    AGE_14_15 = (14, 16)
    AGE_16_18 = (16, 19)
    AGE_RANGES = [AGE_4_5, AGE_6_7, AGE_8_9, AGE_10_11, AGE_12_13, AGE_14_15, AGE_16_18]

    class AgeSubjectLoader(SubjectFileLoader):

        def _load(self, batch: np.ndarray, cols: list):
            x, y_float = super()._load(batch, cols)
            y = np.zeros([batch.shape[0], len(BaseDatasetAgeRanges.AGE_RANGES)])

            age_col = BaseDataset.DATASET_TARGETS.index(HEADER_AGE)
            for i in range(len(BaseDatasetAgeRanges.AGE_RANGES)):
                low = BaseDatasetAgeRanges.AGE_RANGES[i][0]
                high = BaseDatasetAgeRanges.AGE_RANGES[i][1]
                y[np.where((y_float[:, age_col] >= low) & (y_float[:, age_col] < high))[0], i] = 1

            return x[~np.isnan(x).any(axis=1)], y[~np.isnan(x).any(axis=1)]

    GENERATOR = AgeSubjectLoader

    def outputshape(self):
        return len(self.AGE_RANGES)


# To make the MEG dataset, we ensure that the files that are loaded are from the MEG directory
class MEGDataset(BaseDataset):

    def __init__(self, toplevel, PDK=True, PA=True, VG=True, MO=False, batchsize=2):
        super().__init__(toplevel, PDK, PA, VG, MO, batchsize)

    @property
    def modality_folders(self) -> list:
        return ['MEG']

    @property
    def preprocessed_file(self):
        return self.__class__.__name__ + PREV_EVAL_FILE


# Acoustic dataset is loaded by loading files from directory labelled Acoustic
class AcousticDataset(BaseDataset):

    @property
    def modality_folders(self) -> list:
        return ['Audio']

    @property
    def preprocessed_file(self):
        return self.__class__.__name__ + PREV_EVAL_FILE


class MEGAgeRangesDataset(MEGDataset, BaseDatasetAgeRanges):
    # Should work as is
    pass


class AcousticAgeRangeDataset(AcousticDataset, BaseDatasetAgeRanges):
    # Should work as is
    pass


class FusionDataset(MEGDataset, AcousticDataset):

    def __init__(self, toplevel, PDK=True, PA=True, VG=True, MO=False, batchsize=2):
        super().__init__(toplevel, PDK, PA, VG, MO, batchsize)
        if MEG_COLUMNS.exists():
            print('Found MEG Index')
            self.megind = np.load(str(MEG_COLUMNS))
        else:
            self.megind = None

    class FusionFileLoader(SubjectFileLoader):

        @cached(SubjectFileLoader.cache)
        def get_features(self, path_to_file):
            """
            Loads arrays from file, and returned as a flattened vector, cached to save some time
            :param path_to_file:
            :return: numpy vector
            """
            # m = loadmat(str(path_to_file[0]), squeeze_me=True)['features'].ravel()
            # a = loadmat(str(path_to_file[1]), squeeze_me=True)['features'].ravel(

            #        print(path_to_file)

            m = np.load(str(path_to_file[0]))
            a = np.load(str(path_to_file[1]))

            if megind is not None:
                m = m[:, megind].squeeze()

            # m = zscore(m)
            # a = zscore(a)

            #        if np.isnan(m.max()):
            #            print('Bad MEG file.')
            #            exit()
            #        elif np.isnan(a.max()):
            #            print('Bad Audio File.')
            #            exit()
            if self.flatten:
                m = m.ravel()
                a = a.ravel()

            return np.concatenate((m, a))

            # return loadmat(path_to_file, squeeze_me=True)['features'].reshape(-1)

    GENERATOR = FusionFileLoader

    def files_to_load(self, tests):
        longest_vector = -1
        slice_length = -1
        loaded_subjects = {}
        for subject in [x for x in self.toplevel.iterdir() if x.is_dir() and x.name in self.subject_hash.keys()]:
            print('Loading subject', subject.stem, '...')
            loaded_subjects[subject.stem] = []

            # Determine overlap of MEG and Audio data
            audiotests = {t.stem: t for t in (subject / self.modality_folders[0]).iterdir() if t.name in tests}
            megtests = {t.stem: t for t in (subject / self.modality_folders[1]).iterdir() if t.name in tests}

            matched = set(audiotests.keys()).intersection(set(megtests.keys()))

            for experiment in matched:

                audioepochs = {x.stem: x for x in audiotests[experiment].iterdir() if x.suffix == LOAD_SUFFIX}
                megepochs = {x.stem: x for x in megtests[experiment].iterdir() if x.suffix == LOAD_SUFFIX}
                matched = set(audioepochs.keys()).intersection(set(megepochs.keys()))

                for epoch in matched:
                    try:
                        # megf = loadmat(str(megepochs[epoch]), squeeze_me=True)
                        # audf = loadmat(str(audioepochs[epoch]), squeeze_me=True)

                        megf = np.load(str(megepochs[epoch]))
                        audf = np.load(str(audioepochs[epoch]))
                        slice_length = max(slice_length, megf.shape[1] + audf.shape[1])
                        longest_vector = max(longest_vector, megf.shape[0]*megf.shape[1] + audf.shape[0]*audf.shape[1])

                        # slice_length = max(slice_length, len(megf['header']) + len(audf['header']))
                        # longest_vector = max(longest_vector,
                        #                      len(megf['features'].reshape(-1)) + len(audf['features'].reshape(-1)))
                        loaded_subjects[subject.stem].append((subject.stem, audioepochs[epoch], megepochs[epoch]))
                    except Exception:
                        print('Warning: Skipping file, error occurred loading:', epoch)

        return loaded_subjects, longest_vector, slice_length

    @property
    def modality_folders(self):
        return AcousticDataset.modality_folders.fget(self) + MEGDataset.modality_folders.fget(self)


class FusionAgeRangesDataset(FusionDataset, BaseDatasetAgeRanges):

    class FusionAgeRangesFileLoader(FusionDataset.FusionFileLoader):

        def _load(self, batch: np.ndarray, cols: list):
            x, y_float = super(FusionDataset.FusionFileLoader, self)._load(batch, cols)

            y = np.zeros([batch.shape[0], len(BaseDatasetAgeRanges.AGE_RANGES)])

            age_col = BaseDataset.DATASET_TARGETS.index(HEADER_AGE)
            for i in range(len(BaseDatasetAgeRanges.AGE_RANGES)):
                low = BaseDatasetAgeRanges.AGE_RANGES[i][0]
                high = BaseDatasetAgeRanges.AGE_RANGES[i][1]
                y[np.where((y_float[:, age_col] >= low) & (y_float[:, age_col] < high))[0], i] = 1

            return x[~np.isnan(x).any(axis=1)], y[~np.isnan(x).any(axis=1)]

    GENERATOR = FusionAgeRangesFileLoader

