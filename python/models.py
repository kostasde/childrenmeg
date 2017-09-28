import numpy as np
import keras
import keras.backend as K
import re

from tensorflow import multiply as tensormult
from tensorflow import matmul
from keras.callbacks import Callback
from keras.models import Sequential, Model
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


class ProjectionLayer(keras.layers.Layer):

    def __init__(self, gridsize=100, **kwargs):
        super().__init__(**kwargs)
        self.gridsize = gridsize

    def build(self, input_shape):
        # Placeholder and constant tensors
        mgx, mgy = np.mgrid[0:self.gridsize, 0:self.gridsize]
        self.gridpoints = K.constant(np.reshape(np.array([mgx, mgy]), (2, -1)).T.astype(np.float32),
                                     dtype=keras.backend.floatx())
        self.input_spec = [keras.engine.InputSpec(ndim=3), keras.engine.InputSpec(shape=[None, None, 2])]
        super(ProjectionLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return tuple([*input_shape[:-1], self.gridsize, self.gridsize])

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        data = inputs[0]
        locs = inputs[1]

        distance = K.sqrt(K.sum(K.pow(K.expand_dims(locs, 1) - K.expand_dims(self.gridpoints, 1), 2), axis=-1))
        loc_mask = K.one_hot(K.argmin(distance), locs.get_shape()[1].value)
        mmul = matmul(data, K.permute_dimensions(loc_mask, [0, 2, 1]))

        return K.reshape(mmul, [-1, data.get_shape()[1].value, self.gridsize, self.gridsize])


class Searchable:
    """
    Provides a set of fairly general searchable hyper-parameters that will be used by most models
    """

    PARAM_LR = 'learning_rate'
    PARAM_OPT = 'optimizer'
    PARAM_BATCH = 'batch_size'
    PARAM_MOMENTUM = 'momentum'
    PARAM_REG = 'regularization'
    PARAM_DROPOUT = 'dropout'
    PARAM_LAYERS = 'layers'
    PARAM_ACTIVATION = 'activation'

    OPTIMIZERS = [keras.optimizers.sgd, keras.optimizers.adam]
    ACTIVATIONS = [keras.activations.relu, keras.activations.elu, keras.layers.LeakyReLU]

    NEEDS_FLAT = True

    @staticmethod
    def search_space():
        """
        Return a search space for the parameters that are expected
        :return: Dictionary of values and spaces
        """
        pass

    @staticmethod
    def parse_choice(choice, OPTIONS):
        if choice in OPTIONS:
            return choice
        elif isinstance(choice, int):
            return OPTIONS[choice]
        else:
            raise TypeError('Choice cannot be parsed...')

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
        self.params = params if params else {}

        self.lr = self.params.get(Searchable.PARAM_LR, 1e-3)
        self.momentum = self.params.get(Searchable.PARAM_MOMENTUM, 0.05)
        self.reg = self.params.get(Searchable.PARAM_REG, 0.05)
        self.batchsize = self.params.get(Searchable.PARAM_BATCH, 128)

        self.optimizer = Searchable.parse_choice(self.params.get(Searchable.PARAM_OPT, 0), Searchable.OPTIMIZERS)
        self.activation = Searchable.parse_choice(self.params.get(Searchable.PARAM_ACTIVATION, 1), Searchable.ACTIVATIONS)


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
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, Searchable.OPTIMIZERS),
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
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, Searchable.OPTIMIZERS),
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
        if Searchable.PARAM_LAYERS not in params.keys() or isinstance(params[Searchable.PARAM_LAYERS], int):
            layerkeys = sorted([key for key in params.keys() if re.match('\dlayer\d', key)])
            if len(layerkeys) > 0:
                return [int(params[key]) for key in layerkeys]
        elif isinstance(params[Searchable.PARAM_LAYERS], tuple):
            return [int(x) for x in params[Searchable.PARAM_LAYERS]]

        raise TypeError('Layers cannot be parsed from: ' + str(params))

    def __init__(self, inputlength, outputlength, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params=params)
        inputlength = np.prod(inputlength)

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
            # self.do = 0.4
        else:
            self.lunits = [512, 256, 32]
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
        super().compile(optimizer=self.opt_param(), loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy,
                                 *extra_metrics], **kwargs)

    @staticmethod
    def search_space():
        return {
            Searchable.PARAM_LR: hp.loguniform(Searchable.PARAM_LR, -12, 0),
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, Searchable.OPTIMIZERS),
            Searchable.PARAM_MOMENTUM: hp.loguniform(Searchable.PARAM_MOMENTUM, -7, 0),
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 1, 100, 5),
            Searchable.PARAM_DROPOUT: hp.normal(Searchable.PARAM_DROPOUT, 0.5, 0.15),
            Searchable.PARAM_REG: hp.loguniform(Searchable.PARAM_REG, -8, 0),
            Searchable.PARAM_LAYERS: hp.choice(Searchable.PARAM_LAYERS, [
                [hp.qloguniform('1layer1', 1.5, 9, 2)],
                [hp.qloguniform('2layer1', 1.5, 9, 2), hp.qloguniform('2layer2', 1.5, 8, 2)],
                [hp.qloguniform('3layer1', 1.5, 8, 2), hp.qloguniform('3layer2', 1.5, 7, 2),
                 hp.qloguniform('3layer3', 1.5, 5.5, 2)],
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


class ShallowFBCSP(SimpleMLP):
    """
    This model creates a learned convolution based FBCSP, this is a reproduction using description and source that uses
     different libraries.

       Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       arXiv preprint arXiv:1703.05051.
    """

    def setupcnnparams(self, params):
        Searchable.__init__(self, params=params)
        Sequential.__init__(self, name=self.__class__.__name__)

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
            self.temporal = int(params[self.PARAM_FILTER_TEMPORAL])
            self.spatial = int(params[self.PARAM_FILTER_SPATIAL])
        else:
            params = {}
            self.temporal = 25
            self.spatial = 25
            self.lunits = [40, 40]
            self.do = 0.5
        self.filters = int(params.get(self.PARAM_FILTER_LAYERS, 1))
        return params

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        params = self.setupcnnparams(params)

        # Build layers
        # Temporal Filtering
        self.add(ExpandLayer(input_shape=inputshape))
        self.add(keras.layers.Conv2D(
            self.lunits[0], (self.temporal, 1), activation='linear', data_format='channels_last',
            # activity_regularizer=keras.regularizers.l2(self.reg)
        ))
        # Spatial Filtering
        self.add(keras.layers.Conv2D(
            self.lunits[1], (1, self.spatial), activation='linear', use_bias=False, data_format='channels_last',
            # activity_regularizer=keras.regularizers.l2(self.reg)
        ))

        self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.Activation(K.square))
        self.add(keras.layers.AveragePooling2D((75, 1), 15))
        self.add(keras.layers.Activation(lambda x: K.log(K.maximum(x, K.constant(1e-6)))))
        self.add(keras.layers.Dropout(self.do))

        # Output convolution
        self.add(keras.layers.Conv2D(
            self.lunits[1], (10, 1), activation='linear', data_format='channels_last',
        ))

        # Classifier
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))

        # self.add(keras.layers.Dense(outputshape, activation='linear',
        #                             kernel_regularizer=keras.regularizers.l2(self.reg)))

    NEEDS_FLAT = False

    PARAM_FILTER_SPATIAL = '_x'
    PARAM_FILTER_TEMPORAL = '_t'
    PARAM_FILTER_LAYERS = 'f_layers'

    @staticmethod
    def search_space():
        cnn_space = SimpleMLP.search_space()
        cnn_space[Searchable.PARAM_BATCH] = hp.quniform(Searchable.PARAM_BATCH, 4, 128, 4)
        cnn_space[ShallowFBCSP.PARAM_FILTER_LAYERS] = hp.quniform(At2DSCNN.PARAM_FILTER_LAYERS, 2, 5, 1)
        cnn_space[Searchable.PARAM_LAYERS] = hp.choice(Searchable.PARAM_LAYERS, [
            [hp.qloguniform('2layer1', 1.5, 5.5, 2), hp.qloguniform('2layer2', 1.5, 5.5, 2)],
            [hp.qloguniform('3layer1', 1.5, 5.5, 2), hp.qloguniform('3layer2', 1.5, 5.5, 2),
             hp.qloguniform('3layer3', 1.5, 5.5, 2)],
            [hp.qloguniform('4layer1', 1.5, 5.5, 2), hp.qloguniform('4layer2', 1.5, 5.5, 2),
             hp.qloguniform('4layer3', 1.5, 5.5, 2), hp.qloguniform('4layer4', 4, 6, 10)],
            [hp.qloguniform('5layer1', 1.5, 5.5, 2), hp.qloguniform('5layer2', 1.5, 5.5, 2),
             hp.qloguniform('5layer3', 1.5, 5.5, 2), hp.qloguniform('5layer4', 4, 6, 10),
             hp.qloguniform('5layer5', 4, 6, 10)]
        ])
        cnn_space[ShallowFBCSP.PARAM_FILTER_TEMPORAL] = hp.quniform('_t', 2, 256, 2)
        cnn_space[ShallowFBCSP.PARAM_FILTER_SPATIAL] = hp.quniform('_x', 2, 64, 2)
        return cnn_space


# Probably not used for testing
class TCNN(ShallowFBCSP):

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        params = self.setupcnnparams(params)

        # Add dummy channel dimension
        self.add(ExpandLayer(axis=-1, input_shape=inputshape))
        # Temporal without using entire channels vector
        self.add(keras.layers.Conv2D(
            self.lunits[0], (self.temporal, 1),
            activation=activation, data_format='channels_last'
        ))
        self.add(keras.layers.SpatialDropout2D(0.2))
        self.add(keras.layers.MaxPool2D((2, 1)))
        self.add(keras.layers.BatchNormalization())

        # Classify after temporal filtering
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(self.lunits[1], activation=activation))
        self.add(keras.layers.Dropout(self.do))
        # self.add(keras.layers.Dense(outputshape, activation='linear',
        #                             kernel_regularizer=keras.regularizers.l2(self.reg)))
        self.add(keras.layers.Dense(outputshape, activation='softmax'))


# Probably not used for testing
class LinearSCNN(ShallowFBCSP):

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        params = self.setupcnnparams(params)

        # Add dummy channel dimension
        self.add(ExpandLayer(axis=-1, input_shape=inputshape))
        # Temporal without using entire channels vector
        self.add(keras.layers.Conv2D(
            self.lunits[0], (1, self.spatial),
            activation=activation, data_format='channels_last'
        ))
        self.add(keras.layers.SpatialDropout2D(0.2))
        self.add(keras.layers.BatchNormalization())
        # self.add(keras.layers.MaxPool2D((1, 2)))

        # Classify after temporal filtering
        self.add(keras.layers.Flatten())
        # self.add(keras.layers.Dense(self.lunits[1], activation=activation))
        # self.add(keras.layers.Dropout(self.do))
        self.add(keras.layers.Dense(outputshape, activation='linear',
                                    kernel_regularizer=keras.regularizers.l2(self.reg)))
        # self.add(keras.layers.Dense(outputshape, activation='softmax'))


# Inspired by PAPER
class FBCSP(ShallowFBCSP):

    def __init__(self, inputshape, outputshape, activation=keras.activations.elu, params=None):
        params = self.setupcnnparams(params)

        # Add dummy channel dimension
        self.add(ExpandLayer(axis=-1, input_shape=inputshape))
        # Temporal without using entire channels vector
        self.add(keras.layers.Conv2D(
            self.lunits[0], (self.temporal, 1),
            activation=activation, data_format='channels_last'
        ))
        self.add(keras.layers.SpatialDropout2D(self.do/2))
        self.add(keras.layers.MaxPool2D((2, 1)))
        self.add(keras.layers.BatchNormalization())

        # Temporal without using entire channels vector
        self.add(keras.layers.Conv2D(
            self.lunits[1], (1, self.spatial),
            activation=activation, data_format='channels_last'
        ))
        self.add(keras.layers.SpatialDropout2D(self.do/2))
        # self.add(keras.layers.MaxPool2D((1, 2)))
        self.add(keras.layers.BatchNormalization())

        # Classify after temporal filtering
        self.add(keras.layers.Flatten())
        # self.add(keras.layers.Dense(self.lunits[2], activation=activation))
        # self.add(keras.layers.Dropout(self.do))
        self.add(keras.layers.Dense(outputshape, activation='linear',
                                    kernel_regularizer=keras.regularizers.l2(self.reg)))
        # self.add(keras.layers.Dense(outputshape, activation='softmax'))


class SpatialCNN(ShallowFBCSP):
    PARAM_GRIDLEN = 'grid_length'

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        params = self.setupcnnparams(params)

        # Make the interpolated image
        signal_in = keras.layers.Input(shape=inputshape)
        # 2 comes from 2D locations
        locs_in = keras.layers.Input(shape=(inputshape[-1], 2))
        tempconv = ExpandLayer()(signal_in)

        # Temporal Convolution
        tempconv = keras.layers.Conv2D(1, (self.temporal, 1), activation=activation, data_format='channels_last', )(tempconv)
        tempconv = keras.layers.SpatialDropout2D(self.do/2, data_format='channels_last')(tempconv)
        tempconv = keras.layers.MaxPool2D(pool_size=(4, 1), data_format='channels_last')(tempconv)
        tempconv = keras.layers.normalization.BatchNormalization()(tempconv)
        tempconv = SqueezeLayer()(tempconv)

        # Start with 2D Spatial filtering, up to a possible 3 layers
        conv = ExpandLayer()(ProjectionLayer(gridsize=50)([tempconv, locs_in]))
        for units in self.lunits[:self.filters]:
            conv = keras.layers.Conv3D(units, (1, self.spatial, self.spatial), activation=activation, data_format='channels_last',)(conv)
            conv = keras.layers.SpatialDropout3D(self.do, data_format='channels_last')(conv)
            conv = keras.layers.MaxPool3D(pool_size=(1, 2, 2), data_format='channels_last')(conv)
            conv = keras.layers.normalization.BatchNormalization()(conv)

        # Classifier
        output = keras.layers.Flatten()(conv)
        for units in self.lunits[self.filters:]:
            output = keras.layers.Dense(units, activation=activation)(output)
            output = keras.layers.Dropout(self.do)(output)
            output = keras.layers.BatchNormalization()(output)

        output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(output)
        # output = keras.layers.Dense(outputshape, activation='linear',
        #                              kernel_regularizer=keras.regularizers.l2(self.reg))(output)

        super(Model, self).__init__([signal_in, locs_in], [output])


class At2DSCNN(ShallowFBCSP):
    """
    This class employs attention to weight a relatively deep set of features from the 2D sensor interpolation
    """

    PARAM_ATTENTION = 'attention'

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        params = self.setupcnnparams(params)
        attention = int(params.get(self.PARAM_ATTENTION, 96))

        # Make the interpolated image
        signal_in = keras.layers.Input(shape=inputshape)
        # 2 comes from 2D locations
        locs_in = keras.layers.Input(shape=(inputshape[-1], 2))
        tempconv = ExpandLayer()(signal_in)

        # Temporal Convolution
        tempconv = keras.layers.Conv2D(1, (self.temporal, 1), activation=activation, data_format='channels_last', )(tempconv)
        tempconv = keras.layers.SpatialDropout2D(self.do/2, data_format='channels_last')(tempconv)
        tempconv = keras.layers.MaxPool2D(pool_size=(4, 1), data_format='channels_last')(tempconv)
        tempconv = keras.layers.normalization.BatchNormalization()(tempconv)
        tempconv = SqueezeLayer()(tempconv)

        # Start with 2D Spatial filtering, up to a possible 3 layers
        conv = ExpandLayer()(ProjectionLayer(gridsize=50)([tempconv, locs_in]))
        weight_attention = 0
        for units in self.lunits[:self.filters]:
            conv = keras.layers.Conv3D(units, (1, self.spatial, self.spatial), activation=activation, data_format='channels_last',)(conv)
            conv = keras.layers.SpatialDropout3D(self.do, data_format='channels_last')(conv)
            conv = keras.layers.MaxPool3D(pool_size=(1, 2, 2), data_format='channels_last')(conv)
            conv = keras.layers.normalization.BatchNormalization()(conv)
            weight_attention = units

        # Attention
        rnn = keras.layers.Bidirectional(
            keras.layers.LSTM(attention, return_sequences=True, dropout=self.do / 2, recurrent_dropout=self.do / 2,)
        )(tempconv)
        rnn = keras.layers.BatchNormalization()(rnn)
        attn = keras.layers.TimeDistributed(keras.layers.Dense(weight_attention, activation='softmax'))(rnn)
        attn = ExpandLayer(-2)(attn)
        attn = ExpandLayer(-2)(attn)

        conv = keras.layers.Multiply()([conv, attn])

        # Classifier
        output = keras.layers.Flatten()(conv)
        for units in self.lunits[self.filters:]:
            output = keras.layers.Dense(units, activation=activation)(output)
            output = keras.layers.Dropout(self.do)(output)
            output = keras.layers.BatchNormalization()(output)

        output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(output)
        # output = keras.layers.Dense(outputshape, activation='linear',
        #                              kernel_regularizer=keras.regularizers.l2(self.reg))(output)

        super(Model, self).__init__([signal_in, locs_in], [output])

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        def next_2(y_true, y_pred):
            return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_pred)) <= 1)
        super(SimpleMLP, self).compile(optimizer=self.opt_param(), loss=keras.losses.categorical_crossentropy,
                                       metrics=[keras.metrics.categorical_crossentropy,
                                                keras.metrics.categorical_accuracy,
                                                next_2,
                                                *extra_metrics], **kwargs)


    @staticmethod
    def search_space():
        space = ShallowFBCSP.search_space()
        space[At2DSCNN.PARAM_ATTENTION] = hp.qloguniform(At2DSCNN.PARAM_ATTENTION, 3, 6, 5)
        space[ShallowFBCSP.PARAM_FILTER_SPATIAL] = hp.quniform('_x', 1, 16, 1)
        return space


class SimpleLSTM(SimpleMLP):
    """
    Basic Bidirectional LSTM, with optional fully connected output layers
    """

    NEEDS_FLAT = False
    PARAM_SEQ = 'return_sequences'

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params=params)
        Sequential.__init__(self, name=self.__class__.__name__)

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
            self.return_seq = bool(params[SimpleLSTM.PARAM_SEQ])
        else:
            self.return_seq = False
            self.lunits = [256, 512]
            self.do = 0.4

        self.add(keras.layers.wrappers.Bidirectional(
            keras.layers.LSTM(self.lunits[0], dropout=self.do/2, return_sequences=self.return_seq,
                              recurrent_dropout=self.do/2,), input_shape=inputshape))
        self.add(keras.layers.BatchNormalization())

        if self.return_seq:
            self.add(keras.layers.Flatten())

        for i, layer in enumerate(self.lunits[1:]):
            self.add(keras.layers.Dense(layer, activation=activation))
            self.add(keras.layers.Dropout(self.do))
            self.add(keras.layers.BatchNormalization())

        # self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))
        # self.add(keras.layers.Dense(outputshape, activation='linear',
        #                             kernel_regularizer=keras.regularizers.l2(self.reg),
        #                             bias_regularizer=keras.regularizers.l2(self.reg)))

    @staticmethod
    def search_space():
        rnn_space = SimpleMLP.search_space()
        rnn_space[SimpleLSTM.PARAM_SEQ] = hp.choice(SimpleLSTM.PARAM_SEQ, [0, 1])
        rnn_space[Searchable.PARAM_BATCH] = hp.quniform(Searchable.PARAM_BATCH, 1, 512, 2)
        rnn_space[Searchable.PARAM_LAYERS] = hp.choice(Searchable.PARAM_LAYERS, [
            [hp.quniform('1layer1', 1, 512, 8)],
            [hp.quniform('2layer1', 1, 512, 8), hp.qloguniform('2layer2', 3, 7, 5)]
        ])
        return rnn_space


# class DeepLSTM(ShallowLSTM)

# Feed-forward LSTM Based attention from FEED-FORWARD NETWORKS WITH ATTENTION CAN SOLVE SOME LONG-TERM MEMORY PROBLEMS
class AttentionLSTM(Model, Searchable):

    TYPE = TYPE_CLASSIFICATION
    NEEDS_FLAT = False

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params=params)

        if params is not None:
            if isinstance(params[SimpleMLP.PARAM_LAYERS], int):
                params[SimpleMLP.PARAM_LAYERS] += 1
            self.lunits = SimpleMLP.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
        else:
            self.lunits = [32, 128]
            self.do = 0.4

        _input = keras.layers.Input(inputshape)

        rnn = keras.layers.Bidirectional(
            keras.layers.LSTM(self.lunits[0], return_sequences=True, dropout=self.do/2, recurrent_dropout=self.do/2,
                              implementation=2, kernel_regularizer=keras.regularizers.l2(self.reg))
        )(_input)
        rnn = keras.layers.BatchNormalization()(rnn)

        a = rnn
        for i in range(2):
            a = keras.layers.TimeDistributed(keras.layers.Dense(2, activation=activation,
                                                                kernel_regularizer=keras.regularizers.l2(self.reg)))(a)
        e = SqueezeLayer()(keras.layers.TimeDistributed(keras.layers.Dense(1, activation=activation))(a))
        alpha = keras.layers.Dense(inputshape[0], activation='softmax')(e)

        fcnn = keras.layers.Dot(-1)([keras.layers.Permute((2, 1))(rnn), alpha])

        for i, layer in enumerate(self.lunits[2:]):
            fcnn = keras.layers.Dense(layer, activation=activation)(fcnn)
            fcnn = keras.layers.Dropout(self.do)(fcnn)
            fcnn = keras.layers.BatchNormalization()(fcnn)

        # self.add(keras.layers.Flatten())
        _output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(fcnn)
        # _output = keras.layers.Dense(outputshape, activation='linear',
        #                              kernel_regularizer=keras.regularizers.l2(self.reg))(fcnn)

        super(Model, self).__init__(_input, _output, name=self.__class__.__name__)

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        super().compile(optimizer=self.opt_param(), loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy,
                                 *extra_metrics], **kwargs)

    @staticmethod
    def search_space():
        space = SimpleLSTM.search_space()
        space[Searchable.PARAM_LAYERS] = hp.choice(Searchable.PARAM_LAYERS, [
            [hp.quniform('1layer1', 1, 128, 2)],
            [hp.quniform('2layer1', 1, 256, 2), hp.quniform('2layer2', 16, 512, 2)],
            [hp.quniform('3layer1', 1, 256, 2), hp.quniform('3layer2', 16, 256, 2), hp.quniform('3layer3', 16, 512, 2)]
        ])
        return space


class FACNN(ShallowFBCSP):

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        params = self.setupcnnparams(params)

        _input = keras.layers.Input(inputshape)

        # Develop CNN Features
        # Add dummy channel dimension
        features = ExpandLayer(axis=-1)(_input)
        features = keras.layers.Conv2D(
            self.lunits[0], (1, self.spatial),
            activation=activation, data_format='channels_last'
        )(features)
        features = keras.layers.SpatialDropout2D(self.do/2)(features)
        features = keras.layers.MaxPool2D((1, 4))(features)
        features = keras.layers.BatchNormalization()(features)

        rnn = keras.layers.Bidirectional(
            keras.layers.LSTM(self.temporal, return_sequences=True, dropout=self.do/2, recurrent_dropout=self.do/2,
                              implementation=2)
        )(_input)
        rnn = keras.layers.BatchNormalization()(rnn)

        # Apply single layer softmax to weight input features
        attn = keras.layers.TimeDistributed(keras.layers.Dense(self.lunits[0], activation='softmax'))(rnn)
        attn = ExpandLayer(-2)(attn)

        new_in = keras.layers.Multiply()([features, attn])
        fcnn = keras.layers.Flatten()(new_in)

        for i, layer in enumerate(self.lunits[1:]):
            fcnn = keras.layers.Dense(layer, activation=activation)(fcnn)
            fcnn = keras.layers.Dropout(self.do)(fcnn)
            fcnn = keras.layers.BatchNormalization()(fcnn)

        # self.add(keras.layers.Flatten())
        _output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(fcnn)
        # _output = keras.layers.Dense(outputshape, activation='linear',
        #                              kernel_regularizer=keras.regularizers.l2(0.1))(fcnn)

        super(Model, self).__init__(_input, _output, name=self.__class__.__name__)

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        super(SimpleMLP, self).compile(optimizer=self.opt_param(), loss=keras.losses.categorical_crossentropy,
                                       metrics=[keras.metrics.categorical_crossentropy,
                                                keras.metrics.categorical_accuracy, *extra_metrics], **kwargs)

    @staticmethod
    def search_space():
        space = FBCSP.search_space()
        space[Searchable.PARAM_BATCH] = hp.quniform(Searchable.PARAM_BATCH, 2, 256, 2)
        space[Searchable.PARAM_LAYERS] = hp.choice(Searchable.PARAM_LAYERS, [
            [hp.quniform('1layer1', 1, 64, 2)],
            [hp.quniform('2layer1', 1, 64, 2), hp.quniform('2layer2', 16, 512, 2)],
            [hp.quniform('3layer1', 1, 32, 2), hp.quniform('3layer2', 16, 512, 2), hp.quniform('3layer3', 16, 128, 2)]
        ])
        return space


class FACNN2(At2DSCNN):

    PARAM_TEMP_POOL = 'temp_pool'
    PARAM_SPAT_POOL = 'spat_pool'
    PARAM_ACTIVATION = 'activation'

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        params = self.setupcnnparams(params)
        attention = int(params.get(self.PARAM_ATTENTION, 55))
        temp_pool = int(params.get(self.PARAM_TEMP_POOL, 4))
        spat_pool = int(params.get(self.PARAM_SPAT_POOL, 1))

        _input = keras.layers.Input(inputshape)

        # Temporal Convolution
        tempconv = ExpandLayer()(_input)
        tempconv = keras.layers.Conv2D(1, (self.temporal, 1), activation=self.activation, data_format='channels_last',
                                       kernel_regularizer=keras.regularizers.l2(self.reg))(tempconv)
        tempconv = keras.layers.SpatialDropout2D(self.do / 2, data_format='channels_last')(tempconv)
        tempconv = keras.layers.MaxPooling2D(pool_size=(temp_pool, 1), data_format='channels_last')(tempconv)

        # features -> spatial convolution
        features = keras.layers.normalization.BatchNormalization()(tempconv)
        # Temporal convolution is squeezed to be used by attention mechanism
        tempconv = SqueezeLayer()(tempconv)
        # tempconv = keras.layers.Reshape([76, -1])(features)

        # Develop CNN Features
        # Add dummy channel dimension
        weight_attention = 0
        for units in self.lunits[:self.filters]:
            features = keras.layers.Conv2D(
                units, (1, self.spatial),
                activation=self.activation, data_format='channels_last',
                kernel_regularizer=keras.regularizers.l2(self.reg)
            )(features)
            features = keras.layers.SpatialDropout2D(self.do)(features)
            features = keras.layers.MaxPooling2D((1, spat_pool))(features)
            features = keras.layers.BatchNormalization()(features)
            weight_attention = units

        rnn = keras.layers.Bidirectional(
            keras.layers.LSTM(attention, return_sequences=True, dropout=self.do/2, recurrent_dropout=self.do/2,
                              implementation=2)
        )(tempconv)
        rnn = keras.layers.BatchNormalization()(rnn)
        # rnn = keras.layers.TimeDistributed(keras.layers.Dense(128, activation=activation))(tempconv)

        # Apply single layer softmax to weight input features
        attn = keras.layers.TimeDistributed(keras.layers.Dense(weight_attention, activation='softmax'))(rnn)
        attn = ExpandLayer(-2)(attn)

        new_in = keras.layers.Multiply()([features, attn])
        # Can try average pooling instead of
        # fcnn = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(new_in)
        fcnn = keras.layers.Flatten()(new_in)

        for units in self.lunits[self.filters:]:
            fcnn = keras.layers.Dense(units, activation=self.activation,
                                      kernel_initializer=keras.initializers.lecun_normal(),
                                      kernel_regularizer=keras.regularizers.l2(self.reg))(fcnn)
            fcnn = keras.layers.Dropout(self.do)(fcnn)
            fcnn = keras.layers.BatchNormalization()(fcnn)

        # self.add(keras.layers.Flatten())
        _output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(fcnn)
        # _output = keras.layers.Dense(outputshape, activation='linear',
        #                              kernel_regularizer=keras.regularizers.l2(self.reg))(fcnn)

        super(Model, self).__init__(_input, _output, name=self.__class__.__name__)

    @staticmethod
    def search_space():
        space = At2DSCNN.search_space()
        space[FACNN2.PARAM_SPAT_POOL] = hp.quniform(FACNN2.PARAM_SPAT_POOL, 1, 5, 1)
        space[FACNN2.PARAM_TEMP_POOL] = hp.quniform(FACNN2.PARAM_TEMP_POOL, 1, 5, 1)
        return space


class FACNN3(FACNN2):

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        params = self.setupcnnparams(params)
        attention = int(params.get(self.PARAM_ATTENTION, 55))
        temp_pool = int(params.get(self.PARAM_TEMP_POOL, 4))
        spat_pool = int(params.get(self.PARAM_SPAT_POOL, 2))
        activation = params.get(self.PARAM_ACTIVATION, activation)

        _input = keras.layers.Input(inputshape)

        # Temporal Convolution
        tempconv = ExpandLayer()(_input)
        tempconv = keras.layers.Conv2D(self.lunits[0], (self.temporal, 1), activation=activation, data_format='channels_last',
                                       kernel_regularizer=keras.regularizers.l2(self.reg))(tempconv)
        tempconv = keras.layers.SpatialDropout2D(self.do / 2, data_format='channels_last')(tempconv)
        tempconv = keras.layers.MaxPooling2D(pool_size=(temp_pool, 1), data_format='channels_last')(tempconv)

        # features -> spatial convolution
        features = keras.layers.normalization.BatchNormalization()(tempconv)
        # Temporal convolution is squeezed to be used by attention mechanism
        # tempconv = SqueezeLayer()(tempconv)
        tempconv = keras.layers.Reshape([inputshape[0], -1])(features)

        # Develop CNN Features
        # Add dummy channel dimension
        weight_attention = 0
        for units in self.lunits[1:self.filters]:
            features = keras.layers.Conv2D(
                units, (1, self.spatial),
                activation=activation, data_format='channels_last', kernel_regularizer=keras.regularizers.l2(self.reg)
            )(features)
            features = keras.layers.SpatialDropout2D(self.do)(features)
            features = keras.layers.MaxPooling2D((1, spat_pool))(features)
            features = keras.layers.BatchNormalization()(features)
            weight_attention = units

        rnn = keras.layers.Bidirectional(
            keras.layers.LSTM(attention, return_sequences=True, dropout=self.do/2, recurrent_dropout=self.do/2,
                              implementation=2)
        )(tempconv)
        rnn = keras.layers.BatchNormalization()(rnn)
        # rnn = keras.layers.TimeDistributed(keras.layers.Dense(128, activation=activation))(tempconv)

        # Apply single layer softmax to weight input features
        attn = keras.layers.TimeDistributed(keras.layers.Dense(weight_attention, activation='softmax'))(rnn)
        attn = ExpandLayer(-2)(attn)

        new_in = keras.layers.Multiply()([features, attn])
        # Can try average pooling instead of
        # fcnn = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(new_in)
        fcnn = keras.layers.Flatten()(new_in)

        for units in self.lunits[self.filters:]:
            fcnn = keras.layers.Dense(units, activation=activation, kernel_initializer=keras.initializers.lecun_normal())(fcnn)
            fcnn = keras.layers.Dropout(self.do)(fcnn)
            fcnn = keras.layers.BatchNormalization()(fcnn)

        # self.add(keras.layers.Flatten())
        _output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(fcnn)
        # _output = keras.layers.Dense(outputshape, activation='linear',
        #                              kernel_regularizer=keras.regularizers.l2(self.reg))(fcnn)

        super(Model, self).__init__(_input, _output, name=self.__class__.__name__)


class ASVM(FACNN):

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params)
        Sequential.__init__(self)
        self.do = params.get(SimpleMLP.PARAM_DROPOUT, 0.4)
        attention = int(params.get(At2DSCNN.PARAM_ATTENTION, 96))

        _input = keras.layers.Input(inputshape)

        rnn = keras.layers.Bidirectional(
            keras.layers.LSTM(attention, return_sequences=True, dropout=self.do, recurrent_dropout=self.do,
                              implementation=2)
        )(_input)
        rnn = keras.layers.BatchNormalization()(rnn)

        # Apply single layer softmax to weight input features
        attn = keras.layers.TimeDistributed(keras.layers.Dense(inputshape[-1], activation='softmax'))(rnn)

        new_in = keras.layers.Multiply()([_input, attn])
        fcnn = keras.layers.Flatten()(new_in)

        # _output = keras.layers.Dense(outputshape, activation='softmax', name='OUT')(fcnn)
        _output = keras.layers.Dense(outputshape, activation='linear',
                                     kernel_regularizer=keras.regularizers.l2(self.reg))(fcnn)

        super(Model, self).__init__(_input, _output, name=self.__class__.__name__)

    @staticmethod
    def search_space():
        space = LinearSVM.search_space()
        space[At2DSCNN.PARAM_ATTENTION] = hp.qloguniform(At2DSCNN.PARAM_ATTENTION, 1.5, 5.5, 5)
        space[SimpleMLP.PARAM_DROPOUT] = hp.normal(SimpleMLP.PARAM_DROPOUT, 0.4, 0.1)
        return space

    def compile(self, **kwargs):
        extra_metrics = kwargs.pop('metrics', [])
        super(SimpleMLP, self).compile(optimizer=self.opt_param(), loss=keras.losses.categorical_hinge,
                                       metrics=[keras.metrics.categorical_crossentropy,
                                                keras.metrics.categorical_accuracy, *extra_metrics], **kwargs)

MODELS = [
    # Basic Regression
    LinearRegression,
    # Basic Classification
    LogisticRegression, LinearSVM, SimpleMLP, StackedAutoEncoder,
    # CNN Based
    ShallowFBCSP, TCNN, LinearSCNN, FBCSP, SpatialCNN,
    # RNN Based
    SimpleLSTM,
    # Attention
    AttentionLSTM, FACNN, FACNN2, At2DSCNN, ASVM
]
