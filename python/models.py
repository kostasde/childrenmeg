import numpy as np
import sklearn
import keras
import keras.backend as K
from keras.models import Sequential
from hyperopt import hp

TYPE_REGRESSION = 0
TYPE_CLASSIFICATION = 1


def mean_pred(y_true, y_pred):
    return K.mean(K.max(y_pred))


def mean_class(y_true, y_pred):
    return K.max(y_true)


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

    @property
    def needsflatdata(self):
        return True

    def __init__(self, params):
        if params is None:
            params = {
                Searchable.PARAM_LR: 1e-3, Searchable.PARAM_BATCH: 200, Searchable.PARAM_REG: 0.001,
                Searchable.PARAM_MOMENTUM: 0.01, Searchable.PARAM_OPT: keras.optimizers.sgd
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
        super().__init__([
            keras.layers.Dense(outputlength, activation='linear', input_dim=inputlength,
                               kernel_regularizer=keras.regularizers.l2(self.reg))
        ], 'Linear Regression')

    def compile(self, **kwargs):
        super().compile(optimizer=self.opt_param(), loss=keras.losses.mean_squared_error,
                        metrics=[keras.metrics.mse, keras.metrics.mae], **kwargs)

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
        super().__init__([
            keras.layers.Dense(outputlength, activation=activation, input_dim=inputlength,
                               kernel_regularizer=keras.regularizers.l2(self.reg),
                               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                               bias_initializer=keras.initializers.Constant(value=0.0001))
        ], self.__class__.__name__)

    def compile(self, **kwargs):
        super().compile(optimizer=self.opt_param(), loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy], **kwargs)

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
        Sequential.compile(self, optimizer=self.opt_param(), loss=keras.losses.squared_hinge, metrics=['accuracy'])


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

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
        else:
            self.lunits = [1024, 128]
            self.do = 0

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
        # self.add(keras.layers.Dense(outputlength, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01)))

    def compile(self, **kwargs):
        super().compile(optimizer=self.opt_param(), loss='categorical_crossentropy',
                        metrics=[keras.metrics.categorical_accuracy], **kwargs)

    @staticmethod
    def search_space():
        return {
            Searchable.PARAM_LR: hp.loguniform(Searchable.PARAM_LR, -8, -2),
            Searchable.PARAM_OPT: hp.choice(Searchable.PARAM_OPT, [keras.optimizers.sgd, keras.optimizers.adam]),
            Searchable.PARAM_MOMENTUM: hp.loguniform(Searchable.PARAM_MOMENTUM, -7, 0),
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 1, 100, 5),
            Searchable.PARAM_DROPOUT: hp.normal(Searchable.PARAM_DROPOUT, 0.6, 0.05),
            Searchable.PARAM_REG: hp.uniform(Searchable.PARAM_REG, 0, 1e-4),
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

    def __init__(self, inputshape, outputshape, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params=params)
        Sequential.__init__(self, name=self.__class__.__name__)

        if params is not None:
            self.lunits = self.parse_layers(params)
            self.do = params[Searchable.PARAM_DROPOUT]
            params[self.PARAM_OPT] = keras.optimizers.sgd
        else:
            self.lunits = [32, 16]
            self.do = 0

        # Build layers
        # Temporal Filtering
        # self.add(keras.layers.Conv1D(
        #     40, 25, padding='causal', activation=activation, input_shape=inputshape)
        # )
        self.add(ExpandLayer(input_shape=inputshape))
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Conv2D(
            self.lunits[0], [int(x) for x in params[self.PARAM_FILTER_TEMPORAL]],
            activation=activation, data_format='channels_last'
        ))
        self.add(keras.layers.MaxPool2D(pool_size=(4, 1)))
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Dropout(self.do))

        # Spatial Filtering
        self.add(keras.layers.Conv2D(
            self.lunits[1], [int(x) for x in params[self.PARAM_FILTER_SPATIAL]], activation=activation, data_format='channels_last'
        ))

        # Classifier
        self.add(keras.layers.MaxPooling2D())
        # self.add(keras.layers.MaxPool1D())
        self.add(keras.layers.Flatten())
        self.add(keras.layers.normalization.BatchNormalization())
        self.add(keras.layers.Dropout(self.do))
        self.add(keras.layers.Dense(outputshape, activation='softmax', name='OUT'))

        # Consider using SVM output layer
        # self.add(keras.layers.Dense(outputlength, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01)))

    @property
    def needsflatdata(self):
        return False

    PARAM_FILTER_SPATIAL = 'spatial_filter'
    PARAM_FILTER_TEMPORAL = 'temporal_filter'

    @staticmethod
    def search_space():
        cnn_space = SimpleMLP.search_space()
        cnn_space[Searchable.PARAM_BATCH] = hp.quniform(Searchable.PARAM_BATCH, 1, 50, 5)
        cnn_space[Searchable.PARAM_LAYERS] = [hp.quniform('2layer1', 2, 64, 10), hp.quniform('2layer2', 2, 64, 10)]
        cnn_space[ShallowTSCNN.PARAM_FILTER_TEMPORAL] = [hp.quniform('_t', 1, 100, 10), 1]
        cnn_space[ShallowTSCNN.PARAM_FILTER_SPATIAL] = [hp.quniform('_x', 1, 100, 10), hp.quniform('_y', 1, 100, 10)]
        return cnn_space


MODELS = [LinearRegression, LogisticRegression, LinearSVM, SimpleMLP, StackedAutoEncoder, ShallowTSCNN]
