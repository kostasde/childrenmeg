import numpy as np
import keras
from keras.models import Sequential


class LinearRegression(Sequential):
    """
    Simple Linear Regression model
    """

    def __init__(self, inputlength, outputlength):
        super().__init__([
            keras.layers.Dense(outputlength, activation='linear', input_dim=inputlength)
        ], 'Linear Regression')

    def compile(self, **kwargs):
        super().compile(optimizer=keras.optimizers.sgd, loss=keras.losses.mean_squared_error,
                        metrics=[keras.metrics.mae], **kwargs)


class SimpleMLP(Sequential):
    """
    Simple Multi-Layer Perceptron with Dense connections
    """

    def __init__(self, inputlength, outputlength, activation=keras.activations.relu):
        super().__init__([
            keras.layers.Dense(inputlength, activation=activation, input_dim=inputlength),
            keras.layers.Dense(outputlength, activation='softmax')
        ], "Multi-Layer Perceptron")


MODELS = [LinearRegression, SimpleMLP]
