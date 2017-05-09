import pickle
from pathlib import Path
import numpy as np
import argparse

from hyperopt import hp, STATUS_OK, STATUS_FAIL, fmin, Trials, tpe

from MEGDataset import *
from models import *
from experiments import DATASETS


def hp_search(model_constructor, dataset_constructor, args):

    trials_file = Path(args.save_trials)

    early_stop = keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=10)
    lrreduce = keras.callbacks.ReduceLROnPlateau(min_lr=1e-12, verbose=1, epsilon=0.05, patience=5, factor=0.5)

    try:
        trials = pickle.load(trials_file.open('rb'))
        print('Loaded previous {0} trials from {1}'.format(len(trials.losses()), str(args.save_trials)))
    except (EOFError, FileNotFoundError) as e:
        print('Creating new trials file at:', args.save_trials)
        trials = Trials()
        pickle.dump(trials, trials_file.open('wb'))

    def loss(hyperparams):
        print('-'*30)
        print(hyperparams)
        print('-'*30)

        dataset = dataset_constructor(args.toplevel, batchsize=int(hyperparams[Searchable.PARAM_BATCH]))
        model = model_constructor(dataset.inputshape(), dataset.outputshape(), params=hyperparams)
        model.compile()
        model.summary()

        try:
            model.fit_generator(dataset.trainingset(), np.ceil(dataset.traindata.shape[0] / dataset.batchsize),
                                validation_data=dataset.evaluationset(),
                                validation_steps=np.ceil(dataset.eval_points.shape[0] / dataset.batchsize),
                                workers=4, epochs=args.epochs, callbacks=[early_stop, lrreduce])
        except Exception as e:
            print('Training failed with:', e)
            return {'status': STATUS_FAIL}

        metrics = model.evaluate_generator(dataset.evaluationset(),
                                           np.ceil(dataset.testpoints.shape[0]/dataset.batchsize), workers=4)

        if trials_file.exists():
            print('Saving trial...')
            pickle.dump(trials, trials_file.open('wb'))

        return {'loss': metrics[1], 'status': STATUS_OK}

    best_model = fmin(loss, space=model_constructor.search_space(), trials=trials,
                      algo=tpe.suggest, verbose=1, max_evals=args.max_evals)

    if trials_file.exists():
        print('Saving all trials...')
        pickle.dump(trials, trials_file.open('wb'))

    print('All Losses:', trials.losses())
    print('Best Model Found:', best_model)

    return best_model, trials

if __name__ == '__main__':

    MODELS = {x.__name__: x for x in MODELS}

    parser = argparse.ArgumentParser(description='Run bayesian hyperparameter search')
    parser.add_argument('toplevel', help='Directory where dataset is located')
    parser.add_argument('model', choices=MODELS.keys())
    parser.add_argument('dataset', choices=DATASETS.keys())

    parser.add_argument('--epochs', default=40, type=int, help='Number of epochs to run each trial')
    parser.add_argument('--max-evals', default=10, type=int, help='Number of search trials to run')
    parser.add_argument('--save-best', '-p', help='File to save the best model parameters to.',
                        type=argparse.FileType('wb'))
    parser.add_argument('--save-trials', help='File to use to load and store previous/future results. This allows the '
                                              'continuation of testing and the ability to explore how all different '
                                              'hyperparameters performed.', type=str)

    args = parser.parse_args()

    model_constructor = MODELS[args.model]
    dataset_constructor = DATASETS[args.dataset][MODELS[args.model].TYPE]

    best_model, trials = hp_search(model_constructor, dataset_constructor, args)

    if args.save_best is not None:
        print('Saving best model hyperparameters to:', args.save_best)
        pickle.dump(best_model, args.save_best)

