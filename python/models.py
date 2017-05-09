import numpy as np
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

    def __init__(self, params):
        if params is None:
            params = {Searchable.PARAM_LR: 1e-3, Searchable.PARAM_BATCH: 10, Searchable.PARAM_REG: 0}

        self.lr = params[Searchable.PARAM_LR]
        self.batchsize = params[Searchable.PARAM_BATCH]
        self.reg = params[Searchable.PARAM_REG]


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
        super().compile(optimizer=keras.optimizers.adam(), loss=keras.losses.mean_squared_error,
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
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 10, 200, 10),
            Searchable.PARAM_REG: hp.loguniform(Searchable.PARAM_REG, -4, -1)
        }


class LogisticRegression(Sequential, Searchable):
    """
        Simple Linear Regression model
        """

    TYPE = TYPE_CLASSIFICATION

    def __init__(self, inputlength, outputlength, params=None):
        Searchable.__init__(self, params=params)
        super().__init__([
            keras.layers.Dense(outputlength, activation='softmax', input_dim=inputlength,
                               kernel_regularizer=keras.regularizers.l2(self.reg),
                               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                               bias_initializer=keras.initializers.Constant(value=0.0001))
        ], 'Logistic Regression')

    def compile(self, **kwargs):
        super().compile(optimizer=keras.optimizers.adam(), loss=keras.losses.categorical_crossentropy,
                        metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy], **kwargs)

    @staticmethod
    def search_space():
        return {
            Searchable.PARAM_LR: hp.loguniform(Searchable.PARAM_LR, -7, 0),
            Searchable.PARAM_BATCH: hp.quniform(Searchable.PARAM_BATCH, 1, 1000, 10),
            Searchable.PARAM_REG: hp.loguniform(Searchable.PARAM_REG, -4, 0)
        }


class SimpleMLP(Sequential, Searchable):
    """
    Simple Multi-Layer Perceptron with Dense connections
    """

    TYPE = TYPE_CLASSIFICATION

    def __init__(self, inputlength, outputlength, activation=keras.activations.relu, params=None):
        Searchable.__init__(self, params=params)

        if params is not None:
            self.lunits = [int(x) for x in params[Searchable.PARAM_LAYERS]]
            self.do = params[Searchable.PARAM_DROPOUT]
            self.momentum = params[Searchable.PARAM_MOMENTUM]
            self.optimizer = params[Searchable.PARAM_OPT]
        else:
            self.lunits = [410, 10]
            self.do = 0

        super().__init__(name="Multi-Layer Perceptron")

        # Build layers
        self.add(keras.layers.Dense(self.lunits[0], activation=activation, input_dim=inputlength))
        self.add(keras.layers.Dropout(self.do))
        for l in self.lunits[1:]:
            self.add(keras.layers.Dense(l, activation=activation))
            self.add(keras.layers.Dropout(self.do))
        self.add(keras.layers.Dense(outputlength, activation='softmax'))

    def compile(self, **kwargs):
        if self.optimizer is keras.optimizers.adam:
            opt = keras.optimizers.adam(self.lr)
        elif self.optimizer is keras.optimizers.sgd:
            opt = keras.optimizers.sgd(self.lr, self.momentum, nesterov=True)
        super().compile(optimizer=opt, loss='categorical_crossentropy',
                        metrics=[keras.metrics.categorical_accuracy, mean_pred, mean_class], **kwargs)

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


MODELS = [LinearRegression, LogisticRegression, SimpleMLP]
