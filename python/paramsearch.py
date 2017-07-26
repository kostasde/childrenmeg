import pickle
from pathlib import Path
import numpy as np
import argparse
import tty
import sys
import warnings
from select import select

from hyperopt import hp, STATUS_OK, STATUS_FAIL, fmin, Trials, tpe

from MEGDataset import *
from models import *
from experiments import DATASETS


class SkipOnKeypress(keras.callbacks.Callback):
    """
    This class allows a model to end training on the event of a keypress.
    """
    def __init__(self, keys='q', confirmation=False):
        super(SkipOnKeypress, self).__init__()

        self.keys = keys
        self.confirmation = confirmation

        if type(keys) is not str:
            warnings.warn("""Keys not a string, falling back to using 'q'.""", RuntimeWarning)
            self.keys = 'q'

    def on_epoch_end(self, epoch, logs=None):#batch, logs=None):
        i, o, e = select([sys.stdin], [], [], 0.0001)
        if sys.stdin in i:
            if sys.stdin.read(1)[0] in self.keys:
                if self.confirmation:
                    if input('Do you want to stop training this model?') not in ['y', 'yes']:
                        return

                self.model.stop_training = True
                if self.model.stop_training:
                    print('Stopping...')

    # def on_train_end(self, logs=None):


def hp_search(model_constructor, dataset_constructor, args):

    trials_file = Path(args.save_trials)

    callbacks = [
        keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=args.patience),
        keras.callbacks.EarlyStopping(min_delta=0.001, verbose=1, mode='min', patience=args.patience//2),
        keras.callbacks.ReduceLROnPlateau(verbose=1, epsilon=0.05, patience=args.patience//10, factor=0.5),
        keras.callbacks.TerminateOnNaN(),
        SkipOnKeypress()
    ]

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
            s = dataset.trainingset(flatten=model.NEEDS_FLAT)
            e = dataset.evaluationset(flatten=model.NEEDS_FLAT)

            model.fit_generator(s, np.ceil(s.n / s.batch_size),
                                validation_data=e, validation_steps=np.ceil(e.n / e.batch_size),
                                workers=4, epochs=args.epochs, callbacks=callbacks)
            metrics = model.evaluate_generator(e, np.ceil(e.n / e.batch_size), workers=4)
        except Exception as _e:
            print('Training failed with:', _e)
            return {'status': STATUS_FAIL}

        if trials_file.exists():
            print('Saving trial...')
            pickle.dump(trials, trials_file.open('wb'))

        if metrics[0] == np.nan or metrics[0] is np.nan:
            print('NaN Found loss, forcing evaluation loss of inf...')
            l = np.inf
        elif 0 <= args.opt_metric < len(model.metrics_names):
            l = metrics[args.opt_metric]
            print('Using optimization metric:', model.metrics_names[args.opt_metric], 'Value:', l)
        else:
            l = metrics[0]

        return {'loss': l, 'status': STATUS_OK}

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

    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to run each trial')
    parser.add_argument('--opt-metric', '-m', type=int, default=0, help='From the metrics that the model calculates,'
                                                                        ' the index of the metric that will be used to'
                                                                        'evaluate the minimization during the parameter'
                                                                        ' search.')
    parser.add_argument('--patience', default=10,
                        help='How many epochs of no change from which we determine there is no'
                             'need to proceed and can stop early.', type=int)
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

