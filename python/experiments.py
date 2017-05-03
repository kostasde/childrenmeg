import argparse
import pickle
from pathlib import Path
import numpy as np

import keras
from keras.models import Sequential

from MEGDataset import *
from models import *


if __name__ == '__main__':

    DATASETS = {
        'MEG': [MEGDataset, MEGAgeRangesDataset],
        'Audio': [AcousticDataset, AcousticAgeRangeDataset],
        'Fusion': [FusionDataset, FusionAgeRangesDataset]
    }
    MODELS = {x.__name__: x for x in MODELS}

    parser = argparse.ArgumentParser(description='Main interface to train and run experiments from.')
    parser.add_argument('toplevel')
    parser.add_argument('model', choices=MODELS.keys())
    parser.add_argument('dataset', choices=DATASETS.keys())

    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--save-model-params', '-p', default=None, help='If provided, this saves model parameters for '
                                                                        'each fold of an experiment in the provided'
                                                                        'directory.')
    args = parser.parse_args()

    # Load the appropriate dataset, considering whether it is regression or classification
    dataset = DATASETS[args.dataset][MODELS[args.model].TYPE](args.toplevel, batchsize=args.batch_size)

    print('-'*30)
    print('Using ', dataset.__class__.__name__)
    print('-'*30)

    model = MODELS[args.model](dataset.inputshape(), dataset.outputshape())
    model.compile()
    model.summary()

    # Callbacks
    savebest = keras.callbacks.ModelCheckpoint(args.save_model_params + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               verbose=1, save_best_only=True)
    lrreduce = keras.callbacks.ReduceLROnPlateau(min_lr=1e-12, verbose=1, epsilon=0.05, patience=0)
    early_stop = keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min')

    print('Train Model')
    model.fit_generator(dataset.trainingset(), np.ceil(dataset.traindata.shape[0]/dataset.batchsize),
                        validation_data=dataset.evaluationset(),
                        validation_steps=np.ceil(dataset.eval_points.shape[0]/dataset.batchsize),
                        workers=4, epochs=args.epochs, callbacks=[savebest, lrreduce, early_stop])

    print('Test performance')
    print(model.evaluate_generator(
        dataset.testset(),
        np.ceil(dataset.testpoints.shape[0]/dataset.batchsize),
        workers=4
    ))
