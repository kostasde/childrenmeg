import utils
import pickle
import keras
import numpy as np

from pathlib import Path
from sklearn.metrics import accuracy_score


# from models import MODELS
from MEGDataset import MEGRawTask

from models import BestSCNN

FOLDS = 5
BATCH_SIZE = 40
EPOCHS = 200
PATIENCE = 20

WEIGHTS_FILE = 'Fold-{0}-weights.hdf5'
PREDICT_FILE = 'Fold-{0}-predictions.pkl'

PRE_TRAINED_WEIGHTS = Path('/ais/clspace5/spoclab/children_megdata/eeglabdatasets/models/MEGraw/scnn/bin/')
# PRE_TRAINED_WEIGHTS = None
SAVE_MODEL = Path('/ais/clspace5/spoclab/children_megdata/eeglabdatasets/models/MEGTask/scnn/ageweights')

if __name__ == '__main__':

    dataset = MEGRawTask('/ais/clspace5/spoclab/children_megdata/biggerset', batchsize=BATCH_SIZE)
    testset = dataset.testset(batchsize=BATCH_SIZE, fnames=True, flatten=False)

    acc = []

    for fold in range(FOLDS):
        print('-' * 30)
        print('Training Fold:', fold)
        print('-' * 30)

        dataset.next_leaveout(force=fold)
        model = BestSCNN(dataset.inputshape(), dataset.outputshape(), output_classifier=[])
        # model = BestSCNN(dataset.inputshape(), dataset.outputshape())
        model.compile()
        model.summary()
        if PRE_TRAINED_WEIGHTS:
            dummy_model = BestSCNN(dataset.inputshape(), 2)
            weight_f = PRE_TRAINED_WEIGHTS / WEIGHTS_FILE.format(fold+1)
            print('Loading weights from: ', str(weight_f))
            dummy_model.load_weights(str(weight_f))
            for layer in model.layers:
                assert isinstance(layer, keras.layers.Layer)
                if 'conv' in layer.name:
                    print('Loading weights and Freezeing layer: ', layer.name)
                    layer.set_weights(dummy_model.get_layer(layer.name).get_weights())
                    layer.trainable = False
            model.summary()

        # Callbacks
        callbacks = [keras.callbacks.ReduceLROnPlateau(verbose=1, patience=PATIENCE // 5, factor=0.5, epsilon=0.05),
                     keras.callbacks.EarlyStopping(min_delta=0.005, verbose=1, mode='min', patience=PATIENCE // 2),
                     keras.callbacks.EarlyStopping(min_delta=0.05, verbose=1, mode='min', patience=PATIENCE),]

        if SAVE_MODEL is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                str(SAVE_MODEL / WEIGHTS_FILE.format(fold+1)), verbose=1, save_best_only=True))

        train_set = dataset.trainingset(batchsize=BATCH_SIZE, flatten=False)
        valid_set = dataset.evaluationset(batchsize=BATCH_SIZE, flatten=False)
        history = model.fit_generator(train_set, np.ceil(train_set.n / BATCH_SIZE),
                                      validation_data=valid_set, validation_steps=np.ceil(valid_set.n / BATCH_SIZE),
                                      epochs=EPOCHS, callbacks=callbacks, class_weight=[0.539, 0.552, 0.91])

        if SAVE_MODEL is not None:
            model.load_weights(str(SAVE_MODEL / WEIGHTS_FILE.format(fold+1)))

        predictions = []
        true_labels = []
        for i in range(int(np.ceil(testset.n / BATCH_SIZE))):
            fn, x, y = next(testset)
            true_labels.append(y)
            predictions.append(model.predict(x, batch_size=BATCH_SIZE, verbose=True))
        acc.append(accuracy_score(np.vstack(true_labels).argmax(axis=-1), np.vstack(predictions).argmax(axis=-1)))
        pickle.dump(predictions, (SAVE_MODEL / PREDICT_FILE.format(fold+1)).open('wb'))

    print('Final Test Accuracies: ', acc)
