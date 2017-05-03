import pickle
from pathlib import Path
import numpy as np
import argparse

from hyperopt import hp, STATUS_OK, STATUS_FAIL, fmin, Trials, tpe

from MEGDataset import *
from models import *


def hp_search(model_constructor, dataset_constructor, args):

    early_stop = keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=3)
    lrreduce = keras.callbacks.ReduceLROnPlateau(min_lr=1e-12, verbose=1, epsilon=0.05, patience=0)

    def loss(hyperparams):
        print('-'*30)
        print(hyperparams)
        print('-'*30)

        dataset = dataset_constructor(args.toplevel, batchsize=int(hyperparams[Searchable.PARAM_BATCH]))
        model = model_constructor(dataset.inputshape(), dataset.outputshape(), params=hyperparams)
        model.compile()
        model.summary()

        best = np.inf
        try:
            model.fit_generator(dataset.trainingset(), np.ceil(dataset.traindata.shape[0] / dataset.batchsize),
                                validation_data=dataset.evaluationset(),
                                validation_steps=np.ceil(dataset.eval_points.shape[0] / dataset.batchsize),
                                workers=4, epochs=args.epochs, callbacks=[early_stop, lrreduce])
        except Exception:
            return {'status': STATUS_FAIL}

        cost, acc = model.evaluate_generator(dataset.testset(), np.ceil(dataset.testpoints.shape[0]/dataset.batchsize),
                                             workers=4)

        return {'loss': best, 'status': STATUS_OK}

    trials = Trials()
    best_model = fmin(loss, space=model_constructor.search_space(), trials=trials,
                      algo=tpe.suggest, verbose=1, max_evals=args.max_evals)

    print('All Losses:', trials.losses())
    print('Best Model Found:', best_model)

    return best_model, trials

if __name__ == '__main__':

    DATASETS = {
        'MEG': [MEGDataset, MEGAgeRangesDataset],
        'Audio': [AcousticDataset, AcousticAgeRangeDataset],
        'Fusion': [FusionDataset, FusionAgeRangesDataset]
    }
    MODELS = {x.__name__: x for x in MODELS}

    parser = argparse.ArgumentParser(description='Run bayesian hyperparameter search')
    parser.add_argument('toplevel', help='Directory where dataset is located')
    parser.add_argument('model', choices=MODELS.keys())
    parser.add_argument('dataset', choices=DATASETS.keys())

    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to run each trial')
    parser.add_argument('--max-evals', default=10, type=int, help='Number of search trials to run')
    parser.add_argument('--save-best', '-p', help='File to save the best model parameters to.',
                        type=argparse.FileType('wb'))

    args = parser.parse_args()

    model_constructor = MODELS[args.model]
    dataset_constructor = DATASETS[args.dataset][MODELS[args.model].TYPE]

    best_model, trials = hp_search(model_constructor, dataset_constructor, args)

    if args.save_best is not None:
        print('Saving best model hyperparameters to:', args.save_best)
        pickle.dump(best_model, args.save_best)

