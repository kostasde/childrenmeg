import numpy as np
import h5py

import keras
from keras.preprocessing.image import Iterator as KerasDataloader

from models import TYPE_CLASSIFICATION, TYPE_REGRESSION
from MNISTDataset import ArrayFeeder


class BCICompetitionIV2aSingleSubjectRegression:

    NUM_BUCKETS = 1

    # Fixme Batchsize here is deprecated, only should be at generators
    def __init__(self, toplevel, shuffle=True, seed=None, batchsize=-1, subject: int = 9):

        self.file = h5py.File(toplevel, 'r')
        self.group = self.file['raw/A0{0}'.format(subject)]

        # Just loading it all into memory from the hdf5 for now...
        self.train = self.group['T']
        self.test = self.group['E']

        self.x = np.array(self.train['X'])
        self.y = np.array(self.train['Y']) - 1

        self.x_test = np.array(self.test['X'])
        self.y_test = np.array(self.test['Y']) - 1

        # Whitened/z-scored
        self.x = (self.x - self.x.mean(1, keepdims=True)) / self.x.std(1, keepdims=True)

        preshuffle = np.arange(self.x.shape[0])
        np.random.shuffle(preshuffle)
        self.x = self.x[preshuffle, :, :]
        self.y = self.y[preshuffle, :]

        div = int(0.2 * self.x.shape[0])
        self.x = [self.x[:div, :, :], self.x[div:, :, :]]
        self.y = [self.y[:div, :], self.y[div:, :]]

        if batchsize < 0:
            batchsize = 128
        self.batchsize = batchsize

        self.next_leaveout(force=0)

    def next_leaveout(self, force=None):
        """
        Moves on to the next group to leaveout.
        :return: Number of which leaveout, `None` if complete
        """
        if force is not None:
            self.leaveout = force

        if self.leaveout == self.NUM_BUCKETS:
            print('Have completed cross-validation')
            # raise CrossValidationComplete
            return None

        # Select next bucket to leave out as evaluation
        self.x_eval = self.eval_points = self.x[self.leaveout]
        self.y_eval = self.y[self.leaveout]

        # Convert the remaining buckets into one list
        self.x_train = self.traindata = np.concatenate(
            [arr for i, arr in enumerate(self.x) if i != self.leaveout]
        )
        self.y_train = np.concatenate(
            [arr for i, arr in enumerate(self.y) if i != self.leaveout]
        )

        self.leaveout += 1

        return self.leaveout

    def current_leaveout(self):
        return self.leaveout

    def trainingset(self, batchsize=None, flatten=True):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        if self.x_train is None:
            raise AttributeError('No fold initialized... Try calling next_leaveout')

        return ArrayFeeder(self.x_train, self.y_train, batchsize, flatten=flatten)

    def evaluationset(self, batchsize=None, flatten=True):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return ArrayFeeder(self.x_eval, self.y_eval, batchsize, flatten=flatten)

    def testset(self, batchsize=None, flatten=True):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return ArrayFeeder(self.x_test, self.y_test, batchsize, flatten=flatten)

    def inputshape(self):
        return self.x_train.shape[1:]

    def outputshape(self):
        return 1


class BCICompetitionIV2aSingleSubjectClassification(BCICompetitionIV2aSingleSubjectRegression):

    TYPE = TYPE_CLASSIFICATION
    NUM_CLASSES = 4

    def __init__(self, toplevel, batchsize=-1, shuffle=True, seed=None, subject: int = 1):
        super().__init__(toplevel, shuffle, seed, batchsize, subject)

        self.y_test = keras.utils.to_categorical(self.y_test, self.NUM_CLASSES)
        self.y_eval = keras.utils.to_categorical(self.y_eval, self.NUM_CLASSES)
        self.y_train = keras.utils.to_categorical(self.y_train, self.NUM_CLASSES)
        self.y = [keras.utils.to_categorical(y, self.NUM_CLASSES) for y in self.y]

    def outputshape(self):
        return self.y_train.shape[-1]
