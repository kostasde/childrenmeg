import numpy as np

import keras

from models import TYPE_CLASSIFICATION, TYPE_REGRESSION
from BCIIV2a import BCICompetitionIV2aSingleSubjectRegression, BCICompetitionIV2aSingleSubjectClassification


class ConstantClassification(BCICompetitionIV2aSingleSubjectClassification):

    NUM_CLASSES = 10
    CONSTANT = 10.5
    POINTS = 10000
    SEQUENCE_LEN = 10

    def __init__(self, toplevel, shuffle=False, seed=None, batchsize=-1,):
        # super().__init__(toplevel, batchsize, shuffle, seed)

        def constant_dataset(points=self.POINTS):
            x = self.CONSTANT * np.ones((points, self.SEQUENCE_LEN))
            y = keras.utils.to_categorical(np.ones((points, 1)), self.NUM_CLASSES)
            return x, y

        self.x, self.y = constant_dataset()
        self.x_test, self.y_test = constant_dataset(self.POINTS//100)

        div = int(0.2 * self.x.shape[0])
        self.x = [self.x[:div, :], self.x[div:, :]]
        self.y = [self.y[:div, :], self.y[div:, :]]

        if batchsize < 0:
            batchsize = 128
        self.batchsize = batchsize

        self.next_leaveout(force=0)

