import argparse

# from models import MODELS
from MEGDataset import *
from MNISTDataset import *

from models import *
# np.random.seed(10)


def train_and_test(model, dataset, args, callbacks=None):
    print('Train Model')
    if args.sanity_set:
        print('Using small subset of data')
        s = dataset.sanityset(flatten=model.NEEDS_FLAT)
        keras.models.Sequential()
    else:
        s = dataset.trainingset(flatten=model.NEEDS_FLAT)

    if args.no_eval:
        print('Warning: Will not evaluate at end of epoch.')
        model.fit_generator(s, np.ceil(s.n / s.batch_size),  # use_multiprocessing=True,
                            workers=args.workers, epochs=args.epochs)
    else:
        e = dataset.evaluationset(flatten=model.NEEDS_FLAT)
        model.fit_generator(s, np.ceil(s.n / s.batch_size),  # use_multiprocessing=True,
                            validation_data=e, validation_steps=np.ceil(e.n / e.batch_size),
                            workers=args.workers, epochs=args.epochs, callbacks=callbacks)

    if args.test:
        print('Test performance')
        s = dataset.testset(flatten=model.NEEDS_FLAT)
    else:
        print('Validation Performance')
        s = dataset.evaluationset(flatten=model.NEEDS_FLAT)

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
    ts = dataset.testset(flatten=model.NEEDS_FLAT)
    return model.evaluate_generator(ts, np.ceil(ts.n/ts.batch_size), workers=args.workers)


def print_metrics(metrics, confusion_matrix=None):
    """
    Pretty Formatted printing of metricts. Also deals with compiling and printing the mean and deviation confusion
    matrix.
    :param metrics: List of test/evaluation metrics.
    :param confusion_matrix: A confusion matrix callback if being used.
    :return:
    """
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

DATASETS = {
    'MEG': [MEGDataset, MEGAgeRangesDataset],
    'Audio': [AcousticDataset, AcousticAgeRangeDataset],
    'Fusion': [FusionDataset, FusionAgeRangesDataset],
    'MNIST': [MNISTRegression, MNISTClassification],
    'MNISTNoise': [MNISTNoisyRegression, MNISTNoisyClassification],
    'MEGraw': [None, MEGRawRanges],
    'MEGTAugRaw': [None, MEGRawRangesTA],
    'MEGSAugRaw': [None, MEGRawRangesSA],
    'FusionRaw': [None, FusionRawRanges]
}

if __name__ == '__main__':

    WORKERS = 4
    MODEL_FILE = 'Model.hdf5'

    # Perform argparse early
    MODELS = {x.__name__: x for x in MODELS}

    parser = argparse.ArgumentParser(description='Main interface to train and run experiments from.')
    parser.add_argument('toplevel')
    parser.add_argument('model', choices=MODELS.keys())
    parser.add_argument('dataset', choices=DATASETS.keys())

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--test', '-t', action='store_true', help='Actually test the best trained model for each fold')
    parser.add_argument('--confusion-matrix', '-f1', action='store_true', help='Produce the confusion matrix for the '
                                                                               'classifier during evaluation and '
                                                                               'testing')
    parser.add_argument('--patience', default=10,
                        help='How many epochs of no change from which we determine there is no'
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
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation after each epoch.')
    parser.add_argument('--save-model-params', '-p', default=None, help='If provided, this saves the best model '
                                                                        'parameters for each fold of an experiment in '
                                                                        'the provided directory.')
    parser.add_argument('--workers', '-w', default=WORKERS, type=int, help='The number of threads to use to load data.')
    parser.add_argument('--fold', default=0, type=int, help='When not performing cross validation, selects which fold '
                                                            'to be used as evaluation set.')
    args = parser.parse_args()

    # Load the appropriate dataset, considering whether it is regression or classification
    dataset = DATASETS[args.dataset][MODELS[args.model].TYPE](args.toplevel, batchsize=args.batch_size)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(verbose=1, patience=args.patience//5, factor=0.5, mode='min'),
                 keras.callbacks.EarlyStopping(min_delta=0.005, verbose=1, mode='min', patience=args.patience//2),
                 keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=args.patience),
                 keras.callbacks.TensorBoard(histogram_freq=1, write_grads=True, write_images=True,
                                             write_batch_performance=True),
                 ]
    more_metrics = []
    if args.confusion_matrix:
        args.confusion_matrix = ConfusionMatrix(len(BaseDatasetAgeRanges.AGE_RANGES))
        callbacks.append(args.confusion_matrix)
        more_metrics += args.confusion_matrix.get_metrics()
    else:
        args.confusion_matrix = None
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
    model.compile(metrics=more_metrics)
    model.summary()

    # Test without training, use saved models. If available, quit when finished, otherwise train.
    if args.test and args.save_model_params is not None:
        print('Using', args.save_model_params, 'to perform tests...')
        d = [x for x in Path(args.save_model_params).glob('Fold-*-weights.hdf5')]
        metrics = []
        if len(d) > 0:
            dataset.next_leaveout(force=args.fold)
            d.sort(key=lambda x: int(re.findall(r'\d+', str(x))[0]))
            for f in d:
                print('Loading model from', str(f))
                model.load_weights(f)
                print('Loaded previous weights!')
                metrics.append(test(model, dataset, args))
                if dataset.next_leaveout() is None:
                    break
            print_metrics(metrics, args.confusion_matrix)
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

    print_metrics(metrics, args.confusion_matrix)
