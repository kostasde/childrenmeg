import pickle
import numpy as np

import keras
from keras.preprocessing.image import Iterator as KerasDataloader
from keras.datasets import mnist

from models import TYPE_CLASSIFICATION, TYPE_REGRESSION


class ArrayFeeder(KerasDataloader):

    def __init__(self, x, y, batchsize, shuffle=True, seed=None, flatten=True):
        self.x, self.y = x, y
        self.flatten = flatten
        super().__init__(n=x.shape[0], batch_size=batchsize, shuffle=shuffle, seed=seed)

    def __next__(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        if self.flatten:
            return np.reshape(self.x[index_array], [self.x[index_array].shape[0], -1]), self.y

        return self.x[index_array], self.y[index_array]


class MNISTRegression:
    """
    This implements a thread-safe loader based on the examples related to Keras ImageDataGenerator

    This class is implements the same interface that MEGDataset should use, and is meant as a validator of models.
    """
    TYPE = TYPE_REGRESSION
    NUM_BUCKETS = 10

    def __init__(self, toplevel, batchsize=-1, shuffle=True, seed=None):

        (self.x, self.y), (self.x_test, self.y_test) = mnist.load_data()

        # self.x = np.reshape(self.x, [self.x.shape[0], -1])
        self.x = self.x.astype(np.float32) / 255

        # self.x_test = self.testpoints = np.reshape(self.x_test, [self.x_test.shape[0], -1])
        self.x_test = self.x_test.astype(np.float32) / 255

        self.x = np.split(self.x, self.NUM_BUCKETS)
        self.y = np.split(self.y, self.NUM_BUCKETS)

        if batchsize < 0:
            batchsize = self.x_train.shape[0]//10
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

    def trainingset(self, batchsize=None):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        if self.x_train is None:
            raise AttributeError('No fold initialized... Try calling next_leaveout')

        return ArrayFeeder(self.x_train, self.y_train, batchsize)

    def evaluationset(self, batchsize=None):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return ArrayFeeder(self.x_eval, self.y_eval, batchsize)

    def testset(self, batchsize=None):
        """
        Provides a generator object with the current training set
        :param batchsize:
        :return: Generator of type :class`.SubjectFileLoader`
        """
        if batchsize is None:
            batchsize = self.batchsize

        return ArrayFeeder(self.x_test, self.y_test, batchsize)

    def inputshape(self):
        return self.x_train.shape[-1]

    def outputshape(self):
        return 1


class MNISTClassification(MNISTRegression):

    TYPE = TYPE_CLASSIFICATION

    def __init__(self, toplevel, batchsize=-1, shuffle=True, seed=None):
        super().__init__(toplevel, batchsize, shuffle, seed)

        self.y_test = keras.utils.to_categorical(self.y_test, self.NUM_BUCKETS)
        self.y_eval = keras.utils.to_categorical(self.y_eval, self.NUM_BUCKETS)
        self.y_train = keras.utils.to_categorical(self.y_train, self.NUM_BUCKETS)
        self.y = [keras.utils.to_categorical(y, self.NUM_BUCKETS) for y in self.y]

    def outputshape(self):
        return self.y_train.shape[-1]
