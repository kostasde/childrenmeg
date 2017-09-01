import pickle
import numpy as np
import utils
from tqdm import tqdm

from keras.preprocessing.image import Iterator as KerasDataloader

# from functools import lru_cache
from cachetools import cached, RRCache
from abc import ABCMeta, abstractmethod
from scipy.io import loadmat
from scipy import interpolate
from pathlib import Path

PREV_EVAL_FILE = 'preprocessed.pkl'

SUBJECT_TABLE = 'subject_table.mat'
SUBJECT_STRUCT = 'subject_struct.mat'

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


class SubjectFileLoader(KerasDataloader):
    """
    This implements a thread-safe loader based on the examples related to Keras ImageDataGenerator

    This class is a somewhat generic generator used for the training, validation and test datasets of the Dataset classes.
    """

    def __init__(self, x, toplevel, longest_vector, subject_hash, target_cols, slice_length, f_loader, batchsize=-1,
                 flatten=True, shuffle=True, seed=None, evaluate=False):

        self.x = np.asarray(x)
        self.toplevel = toplevel
        self.longest_vector = longest_vector
        self.slice_length = slice_length
        self.subject_hash = subject_hash
        self.targets = target_cols
        self.f_loader = f_loader
        self.flatten = flatten
        self.evaluate = evaluate

        if batchsize < 0:
            batchsize = x.shape[0]

        super().__init__(x.shape[0], batchsize, shuffle=shuffle, seed=seed)

    def _load(self, index_array, batch_size, nancheck=False):
        if self.flatten:
            # batches x time x features
            x = np.zeros([batch_size, self.longest_vector])
        else:
            # batches x (flattened time*features)
            x = np.zeros([batch_size, int(np.ceil(self.longest_vector/self.slice_length)), self.slice_length])

        y = np.zeros([batch_size, 1])

        for i, row in enumerate(index_array):
            ep = tuple(str(f) for f in self.x[row, 1:])
            try:
                temp = self.f_loader(ep)
                if 0 in temp.shape:
                    raise ValueError('Empty data.')
            # if temp is None:
            except ValueError as e:
                print('Error occurred in: {0}.\n{1}'.format(ep, e))
                print('Skipping Datapoint...')
                temp = np.full_like(x[i], np.nan)
            if self.flatten:
                temp = temp.ravel()
                x[i, :len(temp)] = temp
            else:
                x[i, :, :temp.shape[-1]] = temp

            y[i, :] = np.array([self.subject_hash[self.x[row, 0]][column] for column in self.targets])

        dims = tuple(i for i in range(1, len(x.shape)))
        if nancheck:
            return x[~np.isnan(x).any(axis=dims)], y[~np.isnan(x).any(axis=dims)]
        else:
            return x, y

    def __next__(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        return self._load(index_array, current_batch_size)


class SpatialChannelAugmentation(SubjectFileLoader):

    LOCATION_LOOKUP = {}
    CHAN_LOCS_FILE = 'chanlocs.csv'

    @staticmethod
    def chan_locations(toplevel, subject, grid):
        cls = SpatialChannelAugmentation
        if subject in cls.LOCATION_LOOKUP.keys():
            return cls.LOCATION_LOOKUP[subject]
        else:
            print('Looking up channels file for subject:', subject)
            if not isinstance(toplevel, Path):
                toplevel = Path(toplevel)
            l = [x for x in toplevel.glob('**/' + subject + '/') if x.is_dir()]
            if len(l) > 1:
                print('Warning, multiple directories found for ', subject)
                for i in l:
                    print(5*' ', i)
                print('Using, ', l[0])

            cls.LOCATION_LOOKUP[subject] = utils.chan2spatial((l[0] / cls.CHAN_LOCS_FILE), grid=grid)
            return cls.LOCATION_LOOKUP[subject]

    def __init__(self, loader: SubjectFileLoader, cov=1e-2*np.eye(2), gridsize=100):
        self.__class__ = type(loader.__class__.__name__, (self.__class__, loader.__class__), {})
        self.__dict__ = loader.__dict__

        self.loader = loader
        self.cov = cov
        self.gridsize = gridsize

    def _load(self, index_array, batch_size, **kwargs):
        x, y = self.loader._load(index_array, batch_size)
        # Determine which subjects are involved
        subject_labels = self.x[index_array, 0]

        locs = np.zeros((batch_size, x.shape[-1], 2))

        for i, subject in enumerate(subject_labels):
            l = SpatialChannelAugmentation.chan_locations(self.loader.toplevel, subject, grid=self.gridsize)

            # apply the noise to it if not evaluation
            if not self.evaluate:
                l += np.random.multivariate_normal([0, 0], self.cov, l.shape[0])

            locs[i] = l

        return [x, locs], y


class TemporalAugmentation(SubjectFileLoader):
    # TODO Replace this with a keras layer so that this is implemented in GPU

    DATASET_SAMPLE_RATE = 200

    def __init__(self, loader: SubjectFileLoader, croplen=2.0, limits=(-0.5, 3.0),
                 inflate=True, cropstyle='uniform',):
        self.__class__ = type(loader.__class__.__name__, (self.__class__, loader.__class__), {})
        self.__dict__ = loader.__dict__

        self.loader = loader
        self.croplen = croplen
        self.cropstyle = cropstyle
        self.inflate = 1

        # If we are inflating the size of an epoch, modify the loader's size
        if inflate:
            # crops, time should always be first dimension, ensure croplen is less than time slices
            f = loader.f_loader((loader.x[0][1],))
            self.crops = (f.shape[0] - croplen*self.DATASET_SAMPLE_RATE)

            if self.crops > 0:
                self.inflate *= self.crops

    def _load(self, index_array, batch_size, **kwargs):
        x, y = self.loader._load(index_array, batch_size)

        if len(x.shape) > 3:
            raise TypeError('Cannot handle data with shape {0}, must be 2 or 3'.format(len(x.shape)))
        if len(x.shape) == 2:
            # Basically undoing operation that was done
            # TODO see if this redundant operation can be removed
            x = x.reshape((x.shape[0], -1, self.slice_length))

        x_new = np.zeros((x.shape[0], int(self.croplen*self.DATASET_SAMPLE_RATE), *x.shape[2:]))

        # select resampling offset and starting point
        if not self.evaluate:
            if self.cropstyle == 'uniform':
                starts = np.random.choice(np.arange(self.crops), size=x.shape[0]).astype(int)
            else:
                starts = np.zeros(x.shape[0])
            for i, s in enumerate(starts):
                x_new[i, :] = \
                    x[i, s:s+int(self.croplen*self.DATASET_SAMPLE_RATE), :]
        else:
            # Just do the first croplen length for evaulation, should consider interpolated version
            x_new = x[:, :x_new.shape[1], :]

        if self.loader.flatten:
            x = x_new.reshape((x.shape[0], -1))
        else:
            x = x_new

        return x, y


class BaseDataset:

    DATASET_TARGETS = [HEADER_AGE]
    NUM_BUCKETS = 10
    LOAD_SUFFIX = '.npy'

    # Not sure I like this...
    GENERATOR = SubjectFileLoader

    cache = RRCache(2*16384)

    @staticmethod
    # @lru_cache(maxsize=8192)
    @cached(cache)
    def get_features(path_to_file):
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
        return l

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
        for subject in tqdm([x for x in self.toplevel.iterdir() if x.is_dir() and x.name in self.subject_hash.keys()]):
            tqdm.write('Loading subject ' + subject.stem + '...')
            loaded_subjects[subject.stem] = []
            for experiment in tqdm([t for e in self.modality_folders if (subject / e).exists()
                                    for t in (subject / e).iterdir() if t.name in tests]):

                for epoch in tqdm([l for l in experiment.iterdir() if l.suffix == self.LOAD_SUFFIX]):
                    try:
                        # f = loadmat(str(epoch), squeeze_me=True)
                        f = self.get_features(tuple([epoch]))
                        # slice_length = max(slice_length, len(f['header']))
                        # longest_vector = max(longest_vector,
                        #                           len(f['features'].reshape(-1)))
                        slice_length = max(slice_length, f.shape[1])
                        longest_vector = max(longest_vector, f.shape[0]*f.shape[1])
                        loaded_subjects[subject.stem].append((subject.stem, epoch))
                    except Exception as e:
                        tqdm.write('Warning: Skipping file, error occurred loading: ' + str(epoch))

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

    def sanityset(self, fold=3, batchsize=None, flatten=True):
        """
        Provides a generator for a small subset of data to ensure that the model can train to it
        :return: 
        """
        if batchsize is None:
            batchsize = self.batchsize

        return self.GENERATOR(np.array(self.buckets[fold][int(0*len(self.buckets[fold])):int(1*len(self.buckets[fold]))]),
                              self.toplevel, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              self.slice_length, self.get_features, batchsize=batchsize, flatten=flatten)

    def trainingset(self, batchsize=None, flatten=True):
        """
        Provides a generator object with the current training set
        :param flatten: Whether to flatten the resulting 
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        if self.traindata is None:
            raise AttributeError('No fold initialized... Try calling next_leaveout')

        return self.GENERATOR(self.traindata, self.toplevel, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              self.slice_length, self.get_features, batchsize=batchsize, flatten=flatten)

    def evaluationset(self, batchsize=None, flatten=True):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return self.GENERATOR(self.eval_points, self.toplevel, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              self.slice_length, self.get_features, batchsize=batchsize, flatten=flatten, evaluate=True)

    def testset(self, batchsize=None, flatten=True):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return self.GENERATOR(self.testpoints, self.toplevel, self.longest_vector, self.subject_hash, self.DATASET_TARGETS,
                              self.slice_length, self.get_features, batchsize=batchsize, flatten=flatten, evaluate=True)

    def inputshape(self):
        return int(self.longest_vector // self.slice_length), self.slice_length

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
            y = np.zeros([x.shape[0], len(BaseDatasetAgeRanges.AGE_RANGES)])

            age_col = BaseDataset.DATASET_TARGETS.index(HEADER_AGE)
            for i in range(len(BaseDatasetAgeRanges.AGE_RANGES)):
                low = BaseDatasetAgeRanges.AGE_RANGES[i][0]
                high = BaseDatasetAgeRanges.AGE_RANGES[i][1]
                y[np.where((y_float[:, age_col] >= low) & (y_float[:, age_col] < high))[0], i] = 1

            dims = tuple(i for i in range(1, len(x.shape)))
            return x, y

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
    """
    This dataset concatenates the MEG and Audio datasets.
    """

    def __init__(self, toplevel, PDK=True, PA=True, VG=True, MO=False, batchsize=2):
        super().__init__(toplevel, PDK, PA, VG, MO, batchsize)
        if MEG_COLUMNS.exists():
            print('Found MEG Index')
            self.megind = np.load(str(MEG_COLUMNS))
        else:
            self.megind = None

    @staticmethod
    # @cached(BaseDataset.cache)
    def get_features(path_to_file):
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

        # if megind is not None:
        #     m = m[:, megind].squeeze()

        # m = zscore(m)
        # a = zscore(a)

        #        if np.isnan(m.max()):
        #            print('Bad MEG file.')
        #            exit()
        #        elif np.isnan(a.max()):
        #            print('Bad Audio File.')
        #            exit()

        return np.concatenate((m, a), axis=-1)

        # return loadmat(path_to_file, squeeze_me=True)['features'].reshape(-1)

    def files_to_load(self, tests):
        longest_vector = -1
        slice_length = -1
        loaded_subjects = {}
        for subject in tqdm([x for x in self.toplevel.iterdir() if x.is_dir() and x.name in self.subject_hash.keys()]):
            tqdm.write('Loading subject ' + subject.stem + '...')
            loaded_subjects[subject.stem] = []

            # Determine overlap of MEG and Audio data
            audiotests = {t.stem: t for t in (subject / self.modality_folders[0]).iterdir() if t.name in tests}
            megtests = {t.stem: t for t in (subject / self.modality_folders[1]).iterdir() if t.name in tests}

            matched = set(audiotests.keys()).intersection(set(megtests.keys()))

            for experiment in matched:

                audioepochs = {x.stem: x for x in audiotests[experiment].iterdir() if x.suffix == self.LOAD_SUFFIX}
                megepochs = {x.stem: x for x in megtests[experiment].iterdir() if x.suffix == self.LOAD_SUFFIX}
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
                        tqdm.write('Warning: Skipping file, error occurred loading: ' + str(epoch))

        return loaded_subjects, longest_vector, slice_length

    @property
    def modality_folders(self):
        return AcousticDataset.modality_folders.fget(self) + MEGDataset.modality_folders.fget(self)


class FusionAgeRangesDataset(FusionDataset, BaseDatasetAgeRanges):
    pass


# Classes that are used for raw data rather than opensmile feature-sets
class MEGRawRanges(MEGAgeRangesDataset):
    LOAD_SUFFIX = '.npy'

    cache = RRCache(10000)

    class DownSamplingLoader(BaseDatasetAgeRanges.AgeSubjectLoader):
        DOWNSAMPLE_FACTOR = 20

        # def _load(self, batch: np.ndarray, cols: list):
        #     x, y = super(MEGRawRanges.DownSamplingLoader, self)._load(batch, cols)
        #     if self.DOWNSAMPLE_FACTOR <= 1:
        #         return x, y
        #     # Simple mean interpolation downsampling
        #     if len(x.shape) > 3:
        #         raise TypeError('Cannot handle data with shape {0}, must be 2 or 3'.format(len(x.shape)))
        #     if len(x.shape) == 2:
        #         # Basically undoing operation that was done
        #         # TODO see if this redundant operation can be removed
        #         x = x.reshape((x.shape[0], -1, self.slice_length))
        #     pad = self.DOWNSAMPLE_FACTOR - x.shape[1] % self.DOWNSAMPLE_FACTOR
        #     x = np.append(x, np.nan * np.zeros((x.shape[0], pad, self.slice_length)), axis=1)
        #     x = np.reshape(x, (x.shape[0], -1, self.DOWNSAMPLE_FACTOR, self.slice_length))
        #     x = np.nanmean(x, axis=-2)
        #     if len(x.shape) == 2 or self.flatten:
        #         x = np.reshape(x, (x.shape[0], -1))
        #     return x, y

    GENERATOR = DownSamplingLoader

    # Do not cache the raw data
    @staticmethod
    @cached(cache)
    def get_features(path_to_file):
        return np.load(path_to_file[0])

    @property
    def modality_folders(self):
        return ['raw/MEG']

    def inputshape(self):
        # FIXME should not have magic number, comes from assumed sample rate of 200
        return 700, self.slice_length


class FusionRawRanges(FusionAgeRangesDataset):
    LOAD_SUFFIX = '.csv'

    @staticmethod
    @cached(FusionAgeRangesDataset.cache)
    def get_features(path_to_file):
        m, a = super().get_features(path_to_file[0]), super().get_features(path_to_file[1])
        return np.concatenate((m, a))


class MEGRawRangesTA(MEGRawRanges):

    @staticmethod
    def GENERATOR(*args, **kwargs):
        return TemporalAugmentation(BaseDatasetAgeRanges.GENERATOR(*args, **kwargs))

    def inputshape(self):
        # FIXME should not have magic number, comes from assumed sample rate of 200
        return 400, self.slice_length


class MEGRawRangesSA(MEGRawRanges):

    GRID_SIZE = 32

    @staticmethod
    def GENERATOR(*args, **kwargs):
        return SpatialChannelAugmentation(MEGRawRanges.GENERATOR(*args, **kwargs),
                                          gridsize=MEGRawRangesSA.GRID_SIZE)

    # def inputshape(self):
    #     # FIXME should not have magic number, comes from assumed sample rate of 200
    #     return 700, self.GRID_SIZE, self.GRID_SIZE


class MEGRawRangesTSA(MEGRawRangesTA):
    @staticmethod
    def GENERATOR(*args, **kwargs):
        return SpatialChannelAugmentation(MEGRawRangesTA.GENERATOR(*args, **kwargs))




