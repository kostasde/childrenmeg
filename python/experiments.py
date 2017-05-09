import argparse
import pickle

# import utils
from MEGDataset import *
from MNISTDataset import *
from models import *

# np.random.seed(10)

DATASETS = {
    'MEG': [MEGDataset, MEGAgeRangesDataset],
    'Audio': [AcousticDataset, AcousticAgeRangeDataset],
    'Fusion': [FusionDataset, FusionAgeRangesDataset],
    'MNIST': [MNISTRegression, MNISTClassification]
}


def train_and_test(model, dataset, args, callbacks=None):
    print('Train Model')
    if args.sanity_set:
        print('Using small subset of data')
        s = dataset.sanityset()
    else:
        s = dataset.trainingset()
    model.fit_generator(s, np.ceil(s.n / s.batch_size),
                        validation_data=dataset.evaluationset(),
                        validation_steps=np.ceil(dataset.eval_points.shape[0] / dataset.batchsize),
                        workers=4, epochs=args.epochs, callbacks=callbacks)

    print('Test performance')
    metrics = model.evaluate_generator(
        dataset.testset(),
        np.ceil(dataset.testpoints.shape[0] / dataset.batchsize),
        workers=4
    )
    print(metrics)
    return metrics


if __name__ == '__main__':

    MODELS = {x.__name__: x for x in MODELS}

    parser = argparse.ArgumentParser(description='Main interface to train and run experiments from.')
    parser.add_argument('toplevel')
    parser.add_argument('model', choices=MODELS.keys())
    parser.add_argument('dataset', choices=DATASETS.keys())

    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--test', '-t', action='store_true', help='Actually test the best trained model for each fold')
    parser.add_argument('--patience', default=5, help='How many epochs of no change from which we determine there is no'
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
    args = parser.parse_args()

    # Load the appropriate dataset, considering whether it is regression or classification
    dataset = DATASETS[args.dataset][MODELS[args.model].TYPE](args.toplevel, batchsize=args.batch_size)

    print('-' * 30)
    print('Using ', dataset.__class__.__name__)
    print('-'*30)

    model = MODELS[args.model](dataset.inputshape(), dataset.outputshape(), params=None)
    model.compile()
    model.summary()

    # Callbacks
    callbacks = []
    if args.save_model_params is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(args.save_model_params +
                                            '/Fold-1-weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                            verbose=1, save_best_only=True)
        )
    callbacks.append(keras.callbacks.ReduceLROnPlateau(min_lr=1e-12, verbose=1, epsilon=0.05, patience=5, factor=0.5))
    callbacks.append(keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=args.patience))

    metrics = []

    # First fold
    metrics.append(train_and_test(model, dataset, args, callbacks=callbacks))

    # Loop through remaining folds
    while args.cross_validation:
        fold = dataset.next_leaveout()
        if fold is None:
            break

        print('-' * 30)
        print('Testing Fold:', fold)
        print('-' * 30)

        callbacks[0] = keras.callbacks.ModelCheckpoint(args.save_model_params +
                                                       '/Fold-{0}-weights.{epoch:02d}-{val_loss:.2f}.hdf5'.format(fold),
                                                       verbose=1, save_best_only=True)

        metrics.append(train_and_test(model, dataset, args, callbacks=callbacks))

    print('\n\nComplete.')
    print('-' * 30)
    metrics = np.array(metrics)
    mean = np.mean(metrics, axis=0)
    stddev = np.std(metrics, axis=0)
    for i, m in enumerate([model.loss, *model.metrics]):
        print(m.__name__, ': Mean', mean[i], 'Stddev', stddev[i])
    print('-' * 30)



