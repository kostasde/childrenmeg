import argparse
import pickle
from pathlib import Path
import numpy as np

import keras
from keras.models import Sequential

from MEGDataset import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main interface to train and run experiments from.')

    dataset = MEGDataset('/mnt/elephant_sized_space/ALL2', batchsize=100)

    model = Sequential()
    model.add(keras.layers.Dense(1, activation='linear', input_dim=dataset.inputshape()))

    model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.mse])

    print('Train Model')
    model.fit_generator(dataset.trainingset(), np.ceil(dataset.traindata.shape[0]/dataset.batchsize),
                        validation_data=dataset.evaluationset(),
                        validation_steps=np.ceil(dataset.eval_points.shape[0]/dataset.batchsize),
                        workers=2, epochs=3)

    print('Test performance')
    model.evaluate_generator(dataset.testset(), np.ceil(dataset.testpoints.shape[0]/dataset.batchsize), workers=1)