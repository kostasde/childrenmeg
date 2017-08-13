import numpy as np
import keras
import keras.backend as K
import re
from tensorflow import multiply as tensormult
from keras.callbacks import Callback
from keras.models import Sequential
from hyperopt import hp

from MEGDataset import BaseDatasetAgeRanges as AgeRanges

TYPE_REGRESSION = 0
TYPE_CLASSIFICATION = 1


class ConfusionMatrix(Callback):
    """
    Callback that reports the confusion matrix at the end of each epoch. This is a workaround to deal with the fact that
    the confusion matrix is hard to do batchwise.

    WARNING: I doubt this implementation would work for a multi-process implementation
    """
    metric_string = 'conf_mat_{0}_{1}'

    @staticmethod
    def _metric_maker(i, j):
        def f(y_true, y_pred, i, j):
            i = K.constant(i, dtype='int64')
            j = K.constant(j, dtype='int64')
            x = K.cast(K.equal(K.argmax(y_true), i), dtype=K.floatx())
            y = K.cast(K.equal(K.argmax(y_pred), j), dtype=K.floatx())
            return K.sum(tensormult(x, y))

        ret = lambda y_true, y_pred: f(y_true, y_pred, i, j)
        ret.__name__ = ConfusionMatrix.metric_string.format(i, j)
        return ret

    def __init__(self, num_classes, plot=False):
        super().__init__()
        self._num_classes = num_classes
        self.cmat = np.zeros((num_classes, num_classes))
        # self.reset()
        if plot:
            print('Have not completed plot implementation, falling back to text...')
        # Create all the metrics functions now
        self.metrics = [self._metric_maker(i, j) for i in range(num_classes) for j in range(num_classes)]

    def reset(self):
        self.cmat = np.zeros((self._num_classes, self._num_classes))

    def get_metrics(self):
        return self.metrics

    def print_matrix(self, newcmat=None):
        if newcmat is None:
            cmat = self.cmat
        else:
            cmat = newcmat
        print('Total True:')
        print(cmat.sum(axis=-1))
        print('Total Predicted:')
        print(cmat.sum(axis=0))
        print('Confusion Matrix:')
        print(cmat/cmat.sum(axis=-1)[:, np.newaxis], flush=True)

    def is_metric(self, name):
        """
        Check if name is a metric string
        :param name:
        :return:
        """
        return re.match(self.metric_string.format('\\d', '\\d'), name) is not None

    def on_epoch_begin(self, epoch, logs=None):
        self.reset()

    def on_batch_end(self, batch, logs=None):
        batch_cmat = np.zeros_like(self.cmat)
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                batch_cmat[i, j] = logs[self.metric_string.format(i, j)]
        self.cmat += batch_cmat

    def on_epoch_end(self, epoch, logs=None):
        self.print_matrix()


class ExpandLayer(keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        input_shape.insert(ax+1, 1)
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs, axis=self.axis)


class SqueezeLayer(ExpandLayer):

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        if input_shape[ax] == 1:
            input_shape.pop(ax)
        else:
            raise ValueError('Dimension ', ax, 'is not equal to 1!')
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return K.squeeze(inputs, axis=self.axis)


class Searchable:
    """
    Makes the model searchable if it implements this interface
    """

    PARAM_LR = 'learning_rate'
    PARAM_OPT = 'optimizer'
    PARAM_BATCH = 'batch_size'
    PARAM_MOMENTUM = 'momentum'
    PARAM_REG = 'regularization'
    PARAM_DROPOUT = 'dropout'
    PARAM_LAYERS = 'layers'

    NEEDS_FLAT = True

    @staticmethod
    def search_space():
        """
        Return a search space for the parameters that are expected
        :return: Dictionary of values and spaces
        """
        pass

    @staticmethod
    def parse_opt(params):
        if callable(params[Searchable.PARAM_OPT]):
            return params[Searchable.PARAM_OPT]
        elif isinstance(params[Searchable.PARAM_OPT], int):
            return [keras.optimizers.sgd, keras.optimizers.adam][params[Searchable.PARAM_OPT]]
        else:
            raise TypeError('Optimizer cannot be parsed from: ' + str(params))

    def opt_param(self):
        if self.optimizer is keras.optimizers.adam:
            return keras.optimizers.Adam(self.lr)
        elif self.optimizer is keras.optimizers.sgd:
            return keras.optimizers.SGD(self.lr, self.momentum, nesterov=True)
        elif callable(self.optimizer) and isinstance(self.optimizer(), keras.optimizers.Optimizer):
            return self.optimizer(self.lr)
        else:
            raise AttributeError('Optimizer not properly initialized, got: ' + str(self.optimizer))

    def __init__(self, params):
        if params is None:
            params = {
                Searchable.PARAM_LR: 1e-3, Searchable.PARAM_BATCH: 100, Searchable.PARAM_REG: 0.1,
                Searchable.PARAM_MOMENTUM: 0.1, Searchable.PARAM_OPT: keras.optimizers.adam
            }

        self.lr = params[Searchable.PARAM_LR]
        self.batchsize = params[Searchable.PARAM_BATCH]
        self.reg = params[Searchable.PARAM_REG]
        self.momentum = params[Searchable.PARAM_MOMENTUM]
        self.optimizer = self.parse_opt(params)


class LinearRegression(Sequential, Searchable):
    """
    Simple Linear Regression model
    """

    def __init__(self, inputlength, outputlength, params=None):
        Searchable.__init__(self, params=params)
        inputlength = np.prod(inputlength)
        super().__init__([
            keras.layers.Dense(outputlength, activation='linear', input_dim=inputlength,
                               kernel_regularizer=keras.regularizers.l2(self.reg))
        ], 'Linear Regression')

    def compile(self, **kwargs):
        extra_metrics = []
        if 'metrics' in kwargs.keys():
            extra_metrics = kwargs['metrics']
        super().compile(optimizer=self.opt_param(), loss=keras.losses.mean_squared_error,
                        metrics=[keras.metrics.mse, keras.metrics.mae, *extra_metrics], **kwargs)

    def summary(self, line_length=None, positions=None):
        super().summary(line_length, positions)
        print('Hyper-Parameters:')
        print('Batch Size:', self.batchsize)
        print('Weight Regularization:', self.reg)

    TYPE = TYPE_REGRESSION

    @staticmethod
    def search_space():
        return {
            Searchable.PARAM_LR: hp.loguniform(Searchable.PARAM_LR, -7, -2),
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, [keras.optimizers.sgd, keras.optimizers.adam]),
            Searchable.PARAM_MOMENTUM: hp.loguniform(Searchable.PARAM_MOMENTUM, -7, 0),
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 10, 200, 10),
            Searchable.PARAM_REG: hp.loguniform(Searchable.PARAM_REG, -4, -1)
        }


class LogisticRegression(Sequential, Searchable):
    """
        Simple Linear Regression model
        """

    TYPE = TYPE_CLASSIFICATION

    def __init__(self, inputlength, outputlength, params=None, activation='softmax'):
        Searchable.__init__(self, params=params)
        inputlength = np.prod(inputlength)
        super().__init__([
            keras.layers.Dense(outputlength, activation=activation, input_dim=inputlength,
                               kernel_regularizer=keras.regularizers.l2(self.reg),
                               # kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                               bias_initializer=keras.initializers.Constant(value=0.0001)
            )
        ], self.__class__.__name__)

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        super().compile(optimizer=self.opt_param(), loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy,
                                 *extra_metrics],
                        **kwargs)

    @staticmethod
    def search_space():
        return {
            Searchable.PARAM_LR: hp.loguniform(Searchable.PARAM_LR, -7, 0),
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, [keras.optimizers.sgd, keras.optimizers.adam]),
            Searchable.PARAM_MOMENTUM: hp.loguniform(Searchable.PARAM_MOMENTUM, -7, 0),
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 1, 1000, 10),
            Searchable.PARAM_REG: hp.loguniform(Searchable.PARAM_REG, -4, 0)
        }


class LinearSVM(LogisticRegression):

    def __init__(self, inputlength, outputlength, params=None):
        super().__init__(inputlength, outputlength, params=params, activation='linear')

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        Sequential.compile(self, optimizer=self.opt_param(), loss=keras.losses.categorical_hinge,
                           metrics=[keras.metrics.categorical_crossentropy, 'accuracy', *extra_metrics], **kwargs)


class SimpleMLP(Sequential, Searchable):
    """
    Simple Multi-Layer Perceptron with Dense connections
    """

    TYPE = TYPE_CLASSIFICATION

    @staticmethod
    def parse_layers(params):
        if isinstance(params[Searchable.PARAM_LAYERS], int):
            n = params[Searchable.PARAM_LAYERS]+1
            return [int(params['{0}layer{1}'.format(n, i)]) for i in range(1, n+1)]
        elif isinstance(params[Searchable.PARAM_LAYERS], tuple):
            return [int(x) for x in params[Searchable.PARAM_LAYERS]]
        else:
            raise TypeError('Layers cannot be parsed from: ' + str(params))

    def __init__(self, inputlength, outputlength, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params=params)
        inputlength = np.prod(inputlength)

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
            # self.do = 0.4
        else:
            self.lunits = [64, 32]
            self.do = 0.4

        super().__init__(name=self.__class__.__name__)

        # Build layers
        # self.add(keras.layers.Dense(self.lunits[0], activation=activation, input_dim=inputlength))
        self.add(keras.layers.Dense(self.lunits[0], activation=activation, input_dim=np.prod(inputlength, dtype=np.int), name='IN1'))
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Dropout(self.do, name='L1'))
        for i, l in enumerate(self.lunits[1:]):
            self.add(keras.layers.Dense(l, activation=activation, name='IN{0}'.format(i+2)))
            self.add(keras.layers.Dropout(self.do, name='L{0}'.format(i+2)))
        self.add(keras.layers.Dense(outputlength, activation='softmax', name='OUT'))
        # Consider using SVM output layer
        # self.add(keras.layers.Dense(outputlength, activation='linear',
        #                             kernel_regularizer=keras.regularizers.l2(self.reg)))

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        super().compile(optimizer=self.opt_param(), loss=keras.losses.categorical_hinge,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy,
                                 *extra_metrics], **kwargs)

    @staticmethod
    def search_space():
        return {
            Searchable.PARAM_LR: hp.loguniform(Searchable.PARAM_LR, -8, -2),
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, [keras.optimizers.sgd, keras.optimizers.adam]),
            Searchable.PARAM_MOMENTUM: hp.loguniform(Searchable.PARAM_MOMENTUM, -7, 0),
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 1, 100, 5),
            Searchable.PARAM_DROPOUT: hp.normal(Searchable.PARAM_DROPOUT, 0.5, 0.15),
            Searchable.PARAM_REG: hp.loguniform(Searchable.PARAM_REG, -4, 0),
            Searchable.PARAM_LAYERS: hp.choice(Searchable.PARAM_LAYERS, [
                [hp.quniform('1layer1', 50, 1000, 10)],
                [hp.quniform('2layer1', 50, 700, 10), hp.quniform('2layer2', 20, 100, 10)],
                [hp.quniform('3layer1', 50, 200, 10), hp.quniform('3layer2', 20, 100, 10), hp.quniform('3layer3', 20, 100, 10)]
            ])
        }


class StackedAutoEncoder(SimpleMLP):
    """
    This encoder trains layer-wise by using Stacked Autoencoders.
    
    Only uses the fit_generator interface.
    """

    class LayerWiseGenWrapper(object):

        def __init__(self, generator, layertransform=None):
            self.gen = generator
            self.transform = layertransform

        def reset(self):
            return self.gen.reset()

        def __iter__(self):
            return self.gen.__iter__()

        def __next__(self):
            x, y = self.gen.__next__()
            if self.transform is not None:
                x = self.transform([x, 1])
            return x, x

    # def __init__(self, inputlength, outputlength, activation=keras.activations.relu, params=None):
    #     """
    #     Create a stacked autoencoder
    #     :param inputlength:
    #     :param outputlength:
    #     :param activation:
    #     :param params:
    #     """
    #     super().__init__(inputlength, outputlength, activation, params)
        # pop the classifier layer, and build the decoding layers
        # self.pop()

    def fit_generator(self, generator,
                      steps_per_epoch,
                      layer_min_delta=1e-3,
                      patience=5,
                      fine_tune=True,
                      epochs=1,
                      verbose=1,
                      callbacks=[],
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_q_size=10,
                      workers=1,
                      pickle_safe=False,
                      initial_epoch=0):
        """
        Special implementation of fit_generator that trains layer-wise, 
        :param layer_min_delta: Minimum change that needs to occur to continue training the layer 
        :param fine_tune: If true, after the model is finished being trained in a layer-wise fashion, a final training
        step occurs where all the layers are trained via backpropagation.
        :return: 
        """
        if verbose:
            print('Beginning greedy training')

        greedycb = keras.callbacks.EarlyStopping(min_delta=layer_min_delta, patience=patience)

        def greedymodel(layername):
            l = self.get_layer(layername)
            # workaround, input shape always there
            conf = l.get_config()
            conf['batch_input_shape'] = l.input_shape
            return Sequential([
                keras.layers.Dense.from_config(conf),
                # keras.layers.BatchNormalization(),
                keras.layers.Dropout(self.do),
                keras.layers.Dense(l.input_shape[-1], activation='linear', name='OUT')
            ])

        def trainlayer(model, train, val):
            model.compile(optimizer=self.optimizer, loss=keras.losses.mse, metrics=[keras.metrics.mae])
            if verbose:
                model.summary()
            model.fit_generator(train, steps_per_epoch, epochs=epochs, verbose=verbose,
                                callbacks=[greedycb, *callbacks], validation_data=val,
                                validation_steps=validation_steps, class_weight=class_weight, max_q_size=max_q_size,
                                workers=workers, pickle_safe=pickle_safe, initial_epoch=initial_epoch)
            return model

        if verbose:
            print('\nLayer 1:\n')
        # layer 1 does not need its data transformed
        # l0 = greedymodel('IN')
        # l0 = trainlayer(l0, StackedAutoEncoder.LayerWiseGenWrapper(generator, lambda x: x),
        #                 StackedAutoEncoder.LayerWiseGenWrapper(validation_data, lambda x: x))
        # # Setting weights
        # self.get_layer('IN').set_weights(l0.get_layer('IN').get_weights())
        # self.get_layer('OUT').set_weights(l0.get_layer('OUT').get_weights())

        for i, layer in enumerate(self.lunits):
            if verbose:
                print('Pre-training layer {0}'.format(i+1))
            in_id = 'IN{0}'.format(i+1)
            # out_id = 'DEC{0}'.format(i)
            model = greedymodel(in_id)
            if i > 0:
                encode = keras.backend.function(inputs=[self.input, keras.backend.learning_phase()],
                                                outputs=[self.get_layer(in_id).input])
            else:
                encode = None
            gtrain = StackedAutoEncoder.LayerWiseGenWrapper(generator, encode)
            geval = StackedAutoEncoder.LayerWiseGenWrapper(validation_data, encode)

            model = trainlayer(model, gtrain, geval)

            self.get_layer(in_id).set_weights(model.get_layer(in_id).get_weights())
            # self.get_layer(out_id).set_weights(model.get_layer(out_id).get_weights())
            print('\nCompleted layer', i+1)
            print()

        if fine_tune:
            print('Fine Tuning...')
            super().fit_generator(generator, steps_per_epoch, epochs=epochs, verbose=verbose,
                                  callbacks=callbacks, validation_data=validation_data,
                                  validation_steps=validation_steps, class_weight=class_weight,
                                  max_q_size=max_q_size, workers=workers,
                                  pickle_safe=pickle_safe, initial_epoch=initial_epoch)


class ShallowTSCNN(SimpleMLP):
    """
    This model implements a fairly straightforward CNN that first filters temporally, and then performs spatial
    filtering across the temporal filters and channels.

    The input data should have the form (time x channels)
    """

    def setupcnnparams(self, params):
        Searchable.__init__(self, params=params)
        Sequential.__init__(self, name=self.__class__.__name__)

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
            self.temporal = [int(params[self.PARAM_FILTER_TEMPORAL])]
            self.spatial = [int(params[self.PARAM_FILTER_SPATIAL])]
        else:
            params = {}
            self.temporal = 128
            self.spatial = 16
            self.lunits = [16, 16, 256]
            self.do = 0.4
        return params

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        params = self.setupcnnparams(params)

        # Build layers
        # Temporal Filtering
        self.add(keras.layers.Conv1D(
            self.lunits[0], self.temporal, padding='causal', activation=activation, input_shape=inputshape,
            # activity_regularizer=keras.regularizers.l2(self.reg)
        ))
        self.add(keras.layers.MaxPool1D())
        # self.add(ExpandLayer(input_shape=inputshape))
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Dropout(self.do/2))

        # Spatial Filtering
        # self.add(keras.layers.Permute((2, 1)))
        # self.add(keras.layers.Conv1D(
        #     self.lunits[1], self.spatial, padding='valid', activation=activation,
            # activity_regularizer=keras.regularizers.l2(self.reg)
        # ))

        # Classifier
        # self.add(keras.layers.MaxPool1D())
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(self.lunits[2], activation=activation))
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Dropout(self.do))
        # self.add(keras.layers.Dense(
        #     self.lunits[2], activation=activation, activity_regularizer=keras.regularizers.l2(self.reg))
        # )
        # self.add(keras.layers.Dropout(self.do))
        # self.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))
        self.add(keras.layers.Dense(outputshape, activation='linear',
                                    kernel_regularizer=keras.regularizers.l2(self.reg)))

    NEEDS_FLAT = False

    PARAM_FILTER_SPATIAL = 'spatial_filter'
    PARAM_FILTER_TEMPORAL = 'temporal_filter'

    @staticmethod
    def search_space():
        cnn_space = SimpleMLP.search_space()
        cnn_space[Searchable.PARAM_BATCH] = hp.quniform(Searchable.PARAM_BATCH, 1, 50, 5)
        cnn_space[Searchable.PARAM_LAYERS] = [
            hp.quniform('3layer1', 16, 128, 2),
            hp.quniform('3layer2', 2, 64, 2),
            hp.quniform('3layer3', 20, 100, 10)
        ]
        cnn_space[ShallowTSCNN.PARAM_FILTER_TEMPORAL] = hp.quniform('_t', 1, 200, 10)
        cnn_space[ShallowTSCNN.PARAM_FILTER_SPATIAL] = hp.quniform('_x', 1, 50, 5)
        return cnn_space


class TCNN(ShallowTSCNN):

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        params = self.setupcnnparams(params)

        # Add dummy channel dimension
        self.add(ExpandLayer(axis=-1, input_shape=inputshape))
        # Temporal without using entire channels vector
        self.add(keras.layers.Conv2D(
            self.lunits[0], (self.temporal[0], 1),
            activation=activation, data_format='channels_last'
        ))
        self.add(keras.layers.SpatialDropout2D(0.2))
        self.add(keras.layers.MaxPool2D((2, 1)))

        # Classify after temporal filtering
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(self.lunits[1], activation=activation))
        self.add(keras.layers.Dropout(self.do))
        self.add(keras.layers.Dense(outputshape, activation='linear',
                                    kernel_regularizer=keras.regularizers.l2(self.reg)))


class Shallow2DSTCNN(ShallowTSCNN):

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        params = self.setupcnnparams(params)

        # Start with 2D Spatial filtering
        self.add(keras.layers.Conv2D(
            64, (16, 16),
            activation=activation, data_format='channels_first', input_shape=inputshape
        ))
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.MaxPool2D(pool_size=(4, 4), data_format='channels_first'))
        self.add(keras.layers.Dropout(self.do))

        # Another Level
        self.add(keras.layers.Conv2D(
            32, (4, 4), activation=activation,
            data_format='channels_first'
        ))

        # Classifier
        self.add(keras.layers.MaxPooling2D(data_format='channels_first'))
        # self.add(keras.layers.MaxPool1D())
        self.add(keras.layers.Flatten())
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Dropout(self.do))
        self.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))


class SimpleGRU(SimpleMLP):

    NEEDS_FLAT = False

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        Searchable.__init__(self, params=params)
        Sequential.__init__(self, name=self.__class__.__name__)

        self.add(keras.layers.GRU(16, activation=activation, dropout=0.4,
                                  recurrent_dropout=0.4, return_sequences=True, input_shape=inputshape))
        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))


MODELS = [
    # Basic Regression
    LinearRegression,
    # Basic Classification
    LogisticRegression, LinearSVM, SimpleMLP, StackedAutoEncoder,
    # CNN Based
    ShallowTSCNN, TCNN, Shallow2DSTCNN,
    # RNN Based
    SimpleGRU
]
