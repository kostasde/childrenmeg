import argparse
import utils

from sklearn.metrics import confusion_matrix
from scipy.stats import mode
from keras.backend import clear_session
from keras import backend as K

# from models import MODELS
from MEGDataset import *
from MNISTDataset import *
from BCIIV2a import *
from ArtificialDataset import *

from models import *
# np.random.seed(10)


def train_and_test(model_maker, dataset, args, callbacks=None):
    print('Train Model')
    model = model_maker()

    batchsize = args.batch_size if args.batch_size > 0 else int(model.batchsize)

    if args.sanity_set:
        print('Using small subset of data')
        s = dataset.sanityset(flatten=model.NEEDS_FLAT, batchsize=batchsize)
    else:
        s = dataset.trainingset(flatten=model.NEEDS_FLAT, batchsize=batchsize)

    if args.no_eval:
        print('Warning: Will not evaluate at end of epoch.')
        model.fit_generator(s, np.ceil(s.n / s.batch_size),  # use_multiprocessing=True,
                            workers=args.workers, epochs=args.epochs)
    else:
        e = dataset.evaluationset(flatten=model.NEEDS_FLAT, batchsize=batchsize)
        model.fit_generator(s, np.ceil(s.n / s.batch_size),  # use_multiprocessing=True,
                            validation_data=e, validation_steps=np.ceil(e.n / e.batch_size),
                            workers=args.workers, epochs=args.epochs, callbacks=callbacks)

    if args.test:
        print('Test performance')
        s = dataset.testset(flatten=model.NEEDS_FLAT, batchsize=batchsize)
    else:
        print('Validation Performance')
        s = dataset.evaluationset(flatten=model.NEEDS_FLAT, batchsize=batchsize)

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
    batchsize = args.batch_size if args.batch_size > 0 else int(model.batchsize)
    ts = dataset.testset(flatten=model.NEEDS_FLAT, batchsize=batchsize)
    iterations = np.ceil(ts.n/ts.batch_size)
    metrics = []
    for i in range(int(iterations)):
        x, y = next(ts)
        metrics.append(model.evaluate(x, y, batch_size=batchsize))
    return np.array(metrics).mean(axis=0)


def predict(model, dataset, args):
    """
    Method to explicitly calculate the predictions of a model, rather than calculate just metrics.
    :param model:
    :param dataset:
    :param args:
    :return: Prediction array(s), True array(s)
    """
    batchsize = args.batch_size if args.batch_size > 0 else int(model.batchsize)
    ts = dataset.testset(flatten=model.NEEDS_FLAT, batchsize=batchsize, fnames=True)
    iterations = np.ceil(ts.n/ts.batch_size)
    y_true = []
    y_pred = []
    pred_max = np.zeros(dataset.outputshape())
    best_fnames = ['' for _ in range(dataset.outputshape())]
    for i in range(int(iterations)):
        fnames, x, y = next(ts)
        y_p = model.predict(x, batch_size=batchsize)
        if hasattr(dataset, 'CROP_VOTE'):
            y_p = y_p.reshape(batchsize, y_p.shape[0]//batchsize, *y_p.shape[1:])
            y_p = y_p.mean(axis=1)
            y = y.reshape(batchsize, y.shape[0]//batchsize, *y.shape[1:])
            y = y.mean(axis=1)
        val_maxes = np.max(y_p, axis=0)
        arg_maxes = np.argmax(y_p, axis=0)
        for j in range(dataset.outputshape()):
            if pred_max[j] < val_maxes[j]:
                pred_max[j] = val_maxes[j]
                best_fnames[j] = fnames[arg_maxes[j]]
        y_true.append(y)
        y_pred.append(y_p)
    y_true = np.vstack(y_true).squeeze()
    y_pred = np.vstack(y_pred).squeeze()
    assert len(y_pred.shape) == 2 and len(y_true.shape) == 2
    return np.argmax(y_pred, -1), np.argmax(y_true, -1), best_fnames


def train_filter_visualization(model, args):
    """
    This will use gradient ascent to develop inputs that maximize the output of the convolutional filters in the
    provided model.

    Some of this code taken from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    :param model:
    :param args:
    :return:
    """
    location = Path(args.activ_vis)
    max_activations = {}
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    conv_dict = dict([(layer, layer_dict[layer]) for layer in layer_dict if 'conv' in layer])
    print("Calculating max-activations for layers:", conv_dict.keys())
    for layer in sorted(conv_dict.keys()):
        print("Looking at layer:", layer)
        max_activations[layer] = []
        # K.set_learning_phase(0)
        layer_output = conv_dict[layer].output
        for filter_index in range(layer_output._keras_shape[-1]):
            print('Filter: {0}/{1}'.format(filter_index+1, layer_output._keras_shape[-1]))
            loss = K.mean(layer_output[..., filter_index]) - 0.05*K.mean(K.square(model.input))
            grads = K.gradients(loss, model.input)[0]
            grads /= (K.sqrt(K.mean(K.square(grads))) + K.constant(1e-5))

            # this function returns the loss and grads given the input picture
            iterate = K.function([model.input, K.learning_phase()], [loss, grads])

            # Create a noise input signal
            in_data = utils.pink_noise(model.input._keras_shape[1:])[np.newaxis, :]

            print('Maximizing Activation')
            patience = 5
            best = -np.inf
            while patience > 0:
                loss_value, grads_value = iterate([in_data, 0])
                # print('Loss:', loss_value)
                in_data += grads_value * 0.2
                if loss_value > best:
                    best = loss_value
                    patience = 5
                else:
                    patience -= 1
            else:
                print('Best Loss:', best)
            max_activations[layer].append(in_data.squeeze())
        print('Saving to ', str(location))
        pickle.dump(max_activations, location.open('wb'))


def print_metrics(metrics, predictions, args):
    """
    Pretty Formatted printing of metricts. Also deals with compiling and printing the mean and deviation confusion
    matrix.
    :param metrics: List of test/evaluation metrics.
    :param confusion_matrix: A confusion matrix callback if being used.
    :return:
    """
    print('=' * 100)
    print('Metrics')
    print('=' * 100)
    metrics = np.array(metrics)
    mean = np.mean(metrics, axis=0)
    stddev = np.std(metrics, axis=0)
    cm = []
    acc_true = []
    for i, m in enumerate([model.loss, *model.metrics]):
        if hasattr(m, '__name__'):
            print(m.__name__)
        else:
            print(m)
        print('.' * 30)
        print(metrics[:, i])
        print('Mean', mean[i], 'Stddev', stddev[i])
        print('-' * 100)
    if args.confusion_matrix:
        print('=' * 100)
        print('Confusion Matrix')
        print('=' * 100)
        for y_pred, y_true, best_fnames in predictions:
            if len(y_true) < len(y_pred) or len(y_pred) < len(y_true):
                print('Warning: Unbalanced labels and predictions!')
                y_pred = y_pred[:min((len(y_true), len(y_pred)))]
                y_true = y_true[:min((len(y_true), len(y_pred)))]
            c = confusion_matrix(y_true, y_pred)
            cm.append(c)
            acc_true.append(np.mean(y_pred == y_true))
            print(c)
            print('-' * 100)
            print('Highest Confidence: ', best_fnames)
            print('-' * 100)
        print('-' * 100)
        print('Mean:')
        print(np.array(cm).mean(axis=0))
        print('=' * 100)
        print('True Accuracy:')
        print(acc_true)
        print('=' * 100)
    print('=' * 100)

DATASETS = {
    'MEG': [MEGDataset, MEGAgeRangesDataset],
    'Audio': [AcousticDataset, AcousticAgeRangeDataset],
    'Fusion': [FusionDataset, FusionAgeRangesDataset],

    'MEGraw': [None, MEGRawRanges],
    'FusionRaw': [None, FusionRawRanges],
    'MEGTAugRaw': [None, MEGRawRangesTA],
    'MEGSAugRaw': [None, MEGRawRangesSA],
    'MEGTSA': [None, MEGRawRangesTSA],

    'BCISS': [BCICompetitionIV2aSingleSubjectRegression, BCICompetitionIV2aSingleSubjectClassification],
    'BCIMS': [BCIIVMultiSubjectRegression, BCIIVMultiSubjectClassification],
    'BCISSTAug': [None, BCISSTAug],
    'BCIMSTAug': [None, BCIMSTAug],

    'Constant': [None, ConstantClassification],
    'MNIST': [MNISTRegression, MNISTClassification],
    'MNISTNoise': [MNISTNoisyRegression, MNISTNoisyClassification],
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
    parser.add_argument('--batch-size', default=-1, type=int)
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
    parser.add_argument('--fold', default=1, type=int, help='When not performing cross validation, selects which fold '
                                                            'to be used as evaluation set.')
    parser.add_argument('--tensorboard', '-tb', action='store_true', help='Enable the tensorboard graphing callback.')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping while preserving the '
                                                                         'patience measure for LR reduction.')
    parser.add_argument('--null-hyp', action='store_true', help='Run a test with initial state weights')
    parser.add_argument('--activ-vis', default=None, help='Location to save the inputs for maximal activations for '
                                                          'each convolution layer.')
    parser.add_argument('--save-metrics', help='CSV file to save the outputted metrics to')
    args = parser.parse_args()

    # Load the appropriate dataset, considering whether it is regression or classification
    dataset = DATASETS[args.dataset][MODELS[args.model].TYPE](args.toplevel, batchsize=args.batch_size)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(verbose=1, patience=args.patience//5, factor=0.5, epsilon=0.05), ]
    if args.tensorboard:
        callbacks += [keras.callbacks.TensorBoard(log_dir='./logs/{0}/{1}/f{2}'.format(args.model, args.dataset,
                                                                                       args.fold), write_images=True)]
    if not args.no_early_stopping:
        callbacks += [
            keras.callbacks.EarlyStopping(min_delta=0.005, verbose=1, mode='min', patience=args.patience//2),
            keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=args.patience),
        ]
    more_metrics = []
    # TODO support for more metrics
    if args.save_model_params is not None:
        args.save_model_params = Path(args.save_model_params)
        # fixme this has become mostly a placeholder, probably a better way of doing this.
        callbacks.append(keras.callbacks.ModelCheckpoint(
            str(args.save_model_params / 'Fold-{0}-weights.hdf5'.format(args.fold)), verbose=1, save_best_only=True)
        )

    print('-' * 30)
    print('Using ', dataset.__class__.__name__)
    print('-'*30)

    # Load Any provided hyperparameters
    if args.hyper_params is not None:
        args.hyper_params = pickle.load(args.hyper_params)
        print('Loaded provided Hyper-parameters')
        print(args.hyper_params)

    # Utility function for making the model type that we will use
    def model_maker():
        model = MODELS[args.model](dataset.inputshape(), dataset.outputshape(), params=args.hyper_params)
        model.compile(metrics=more_metrics)
        model.summary()
        return model

    # Test without training, use saved models if available, quit when finished, otherwise train models.
    if args.test and args.save_model_params is not None:
        print('Using', args.save_model_params, 'to perform tests...')
        if args.confusion_matrix:
            print('Printing confusion matrix for each fold.')
        d = [x for x in Path(args.save_model_params).glob('Fold-*-weights.hdf5')]
        metrics = []
        predictions = []
        if len(d) > 0:
            dataset.next_leaveout(force=args.fold-1)
            d.sort(key=lambda x: int(re.findall(r'\d+', str(x))[0]))
            for f in d:
                model = model_maker()
                print('Loading model from', str(f))
                model.load_weights(f)
                print('Loaded previous weights!')
                metrics.append(test(model, dataset, args))
                if args.confusion_matrix:
                    predictions.append(predict(model, dataset, args))
                if args.activ_vis:
                    train_filter_visualization(model, args)
                if dataset.next_leaveout() is None:
                    break
                # Ensure GPU removes current model
                clear_session()
            print_metrics(metrics, predictions, args)
            exit()
        else:
            print('Could not use existing files...')
            print('Training and then testing instead.')
            print('-'*30)
            print()
    elif args.null_hyp and args.test:
        print('Testing null hypothesis...')
        dataset.next_leaveout(force=args.fold-1)
        if args.confusion_matrix:
            print('Printing confusion matrix for each fold.')
        metrics = []
        predictions = []
        while True:
            print('Testing fold {0}.'.format(dataset.current_leaveout()))
            model = model_maker()
            metrics.append(test(model, dataset, args))
            if args.confusion_matrix:
                predictions.append(predict(model, dataset, args))
            if not dataset.next_leaveout():
                break
        print_metrics(metrics, predictions, args)
        exit()

    # Train the models
    # First fold
    dataset.next_leaveout(force=args.fold-1)
    metrics = []
    # Loop through folds
    while True:
        fold = dataset.current_leaveout()
        print('-' * 30)
        print('Training Fold:', fold)
        print('-' * 30)

        if args.save_model_params is not None:
            callbacks[-1] = keras.callbacks.ModelCheckpoint(str(args.save_model_params /
                                                                'Fold-{0}-weights.hdf5'.format(fold)),
                                                            verbose=1, save_best_only=True,)
                                                            # monitor='val_categorical_crossentropy')
        metrics.append(train_and_test(model_maker, dataset, args, callbacks=callbacks))

        if not args.cross_validation or not dataset.next_leaveout():
            break
        # Ensure GPU is cleared
        clear_session()

        if args.tensorboard:
            callbacks[1] = keras.callbacks.TensorBoard(
                log_dir='./logs/{0}/{1}/f{2}'.format(args.model, args.dataset, fold+1), write_images=True)

    print('\n\nComplete.')
    print_metrics(metrics, [], args)
