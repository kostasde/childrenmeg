import re
import argparse
import pickle

# import utils
from MEGDataset import *
from MNISTDataset import *
from models import *

# np.random.seed(10)

WORKERS = 4
MODEL_FILE = 'Model.hdf5'

DATASETS = {
    'MEG': [MEGDataset, MEGAgeRangesDataset],
    'Audio': [AcousticDataset, AcousticAgeRangeDataset],
    'Fusion': [FusionDataset, FusionAgeRangesDataset],
    'MNIST': [MNISTRegression, MNISTClassification],
    'MEGraw': [None, MEGAugmentedRawRanges],
    'FusionRaw': [None, FusionAugmentedRawRanges]
}


def train_and_test(model, dataset, args, callbacks=None):
    print('Train Model')
    if args.sanity_set:
        print('Using small subset of data')
        s = dataset.sanityset(flatten=model.needsflatdata)
    else:
        s = dataset.trainingset(flatten=model.needsflatdata)
    model.fit_generator(s, np.ceil(s.n / s.batch_size),
                        validation_data=dataset.evaluationset(flatten=model.needsflatdata),
                        validation_steps=np.ceil(dataset.eval_points.shape[0] / dataset.batchsize),
                        workers=args.workers, epochs=args.epochs, callbacks=callbacks)

    if args.test:
        print('Test performance')
        s = dataset.testset()
    else:
        print('Validation Performance')
        s = dataset.evaluationset()

    metrics = model.evaluate_generator(s, np.ceil(s.n / s.batch_size), workers=args.workers)
    print(metrics)
    return metrics


def test(model, dataset, args):
    """
    Method that loops through folds of saved models and returns test performance on each fold.
    :param model: Model, must have testset() method
    :param dataset: 
    :param args: 
    :return: Array of metrics x folds, reporting test metric of each 
    """
    ts = dataset.testset()
    return model.evaluate_generator(ts, np.ceil(ts.n/ts.batch_size), workers=args.workers)


def print_metrics(metrics):
    print('=' * 100)
    metrics = np.array(metrics)
    mean = np.mean(metrics, axis=0)
    stddev = np.std(metrics, axis=0)
    for i, m in enumerate([model.loss, *model.metrics]):
        if hasattr(m, '__name__'):
            print(m.__name__)
        else:
            print(m)
        print('.' * 30)
        print(metrics[:, i])
        print('Mean', mean[i], 'Stddev', stddev[i])
        print('-' * 100)
    print('=' * 100)


if __name__ == '__main__':

    MODELS = {x.__name__: x for x in MODELS}

    parser = argparse.ArgumentParser(description='Main interface to train and run experiments from.')
    parser.add_argument('toplevel')
    parser.add_argument('model', choices=MODELS.keys())
    parser.add_argument('dataset', choices=DATASETS.keys())

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--test', '-t', action='store_true', help='Actually test the best trained model for each fold')
    parser.add_argument('--patience', default=50, help='How many epochs of no change from which we determine there is no'
                                                      'need to proceed and can stop early.', type=int)
    parser.add_argument('--cross-validation', '-x', action='store_true', help='Loop through all the folds of the '
                                                                              'dataset to perform cross validation and'
                                                                              'report the score')
    parser.add_argument('--hyper-params', '-y', default=None, help='File that contains the hyper-parameters to be used'
                                                                   'to train the model. Expecting a pickle file, or '
                                                                   'another file that can be interpreted by pickle',
                        type=argparse.FileType('rb'))
    parser.add_argument('--sanity-set', '-s', action='store_true', help='Use the small dataset to ensure that training'
                                                                       'data is going down.')
    parser.add_argument('--save-model-params', '-p', default=None, help='If provided, this saves the best model '
                                                                        'parameters for each fold of an experiment in '
                                                                        'the provided directory.')
    parser.add_argument('--workers', '-w', default=WORKERS, type=int, help='The number of threads to use to load data.')
    args = parser.parse_args()

    # Load the appropriate dataset, considering whether it is regression or classification
    dataset = DATASETS[args.dataset][MODELS[args.model].TYPE](args.toplevel, batchsize=args.batch_size)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(verbose=1, epsilon=0.05, patience=5, factor=0.5),
                 keras.callbacks.EarlyStopping(min_delta=0.005, verbose=1, mode='min', patience=args.patience//2),
                 keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=args.patience)]
    if args.save_model_params is not None:
        args.save_model_params = Path(args.save_model_params)
        callbacks.append(keras.callbacks.ModelCheckpoint(str(args.save_model_params / 'Fold-1-weights.hdf5'), verbose=1,
                                                         save_best_only=True))

    print('-' * 30)
    print('Using ', dataset.__class__.__name__)
    print('-'*30)

    if args.hyper_params is not None:
        args.hyper_params = pickle.load(args.hyper_params)
        print('Loaded provided Hyper-parameters')
        print(args.hyper_params)

    model = MODELS[args.model](dataset.inputshape(), dataset.outputshape(), params=args.hyper_params)
    model.compile()
    model.summary()

    # Test without training, use saved models. If available, quit when finished, otherwise train.
    if args.test and args.save_model_params is not None:
        print('Using', args.save_model_params, 'to perform tests...')
        d = [x for x in Path(args.save_model_params).glob('Fold-*-weights.hdf5')]
        metrics = []
        if len(d) > 0:
            dataset.next_leaveout(force=0)
            d.sort(key=lambda x: int(re.findall(r'\d+', str(x))[0]))
            for f in d:
                print('Loading model from', str(f))
                model.load_weights(f)
                print('Loaded previous weights!')
                metrics.append(test(model, dataset, args))
                if dataset.next_leaveout() is None:
                    break
            print_metrics(metrics)
            exit()
        else:
            print('Could not use existing files...')
            print('Training and then testing instead.')
            print('-'*30)
            print()

    # First fold
    metrics = [(train_and_test(model, dataset, args, callbacks=callbacks))]

    # Loop through remaining folds
    while args.cross_validation:
        fold = dataset.next_leaveout()
        if fold is None:
            break

        print('-' * 30)
        print('Testing Fold:', fold)
        print('-' * 30)

        if args.save_model_params is not None:
            callbacks[-1] = keras.callbacks.ModelCheckpoint(str(args.save_model_params /
                                                                'Fold-{0}-weights.hdf5'.format(fold)),
                                                            verbose=1, save_best_only=True)
        metrics.append(train_and_test(model, dataset, args, callbacks=callbacks))

    print('\n\nComplete.')

    print_metrics(metrics)



