import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from models import BestSCNN
from MEGDataset import MEGRawRanges, TEST_SUBJECTS, TEST_VG, TEST_PDK, TEST_PA

TESTS = [TEST_PA, TEST_PDK, TEST_VG]
SUBJECT_PART = 6
TEST_PART = 9

TRAINED_MODEL = Path('/ais/clspace5/spoclab/children_megdata/eeglabdatasets/models/MEGraw/scnn/bin'
                     '/Fold-2-weights.hdf5')
RESTING_DATA = '/ais/clspace5/spoclab/children_megdata/biggerset/resting/{0}/{1}.npy'


def noise_occlusion(point, occlusion, occluded_points, occlusion_weight=0.99):
    point[:, occluded_points, ...] = (1 - occlusion_weight) * point[:, occluded_points, ...]
    occlusion *= occlusion_weight
    return normalize(point + occlusion)


def normalize(data, ch_axis=-1, eps=1e-12):
    return (data - data.mean(ch_axis, keepdims=True)) / (data.std(ch_axis, keepdims=True) + eps)


def rest_points(fnames, x_shape):
    """
    Generate all the test points in one contiguous unit, as corresponding to the list of filenames.
    :param fnames: List/ndarray of "Path"s for x points
    :param x_shape: Shape for the resulting data to enable pre-allocation
    :return: All rest points as single ndarray with shape equal to x_shape
    """
    rest = np.zeros(x_shape)
    for i, fn in enumerate(fnames):
        rest[i, ...] = np.load(RESTING_DATA.format(fn.parts[SUBJECT_PART], fn.parts[TEST_PART]))
    return rest


def obscuring_profile(model, x):
    obs_event = np.zeros(700, 2)
    obs_ends = np.zeros(700, 2)
    event_inds = [100]
    end_inds = [[0], [699]]
    for i in range(700):
        if i % 6:
            event_inds.insert(0, event_inds[0]-1)
            end_inds[0].append(end_inds[0][-1] + 1)
        else:
            event_inds.append(event_inds[-1]+1)
            end_inds.insert(0, end_inds[1][0]-1)

        obscure_event = np.zeros_like(x)
        obscure_ends = np.zeros_like(x)
        obscure_event[:, event_inds, :] = np.random.rand(*obscure_event[:, event_inds, :].shape)
        obscure_ends[:, end_inds, :] = np.random.rand(*obscure_event[:, end_inds, :].shape)

        obs_event[i, :] = model.predict(noise_occlusion(x, obscure_event, event_inds))
        obs_ends[i, :] = model.predict(noise_occlusion(x, obscure_ends, end_inds))

    return obs_event, obs_ends


BATCH_SIZE = 256
POINTS_PER_SUBJECT = 20

if __name__ == '__main__':

    dataset = MEGRawRanges('/ais/clspace5/spoclab/children_megdata/biggerset')
    testset = dataset.testset(batchsize=BATCH_SIZE, fnames=True, flatten=False)

    model = BestSCNN(dataset.inputshape(), dataset.outputshape())
    model.compile()
    model.load_weights(TRAINED_MODEL)
    model.summary()

    x = []
    predictions = []
    true_labels = []
    filenames = []
    for i in range(int(np.ceil(testset.n / BATCH_SIZE))):
        fn, _x, y = next(testset)
        x.append(_x)
        true_labels.append(y)
        filenames.append(fn)
        predictions.append(model.predict(_x, batch_size=BATCH_SIZE, verbose=True))

    correct_filter = np.vstack(true_labels).argmax(axis=-1) == np.vstack(predictions).argmax(axis=-1)
    print('Un-modified Test Accuracy: ', np.mean(correct_filter))

    correct_pred = np.vstack(predictions)[correct_filter]
    best_confidence = np.argsort(np.max(correct_pred, axis=1))

    x = np.vstack(x)
    filenames = np.vstack(filenames).squeeze()
    rest = rest_points(filenames, x.shape)

    rest_corr_pred = model.predict(normalize(x - normalize(rest)), batch_size=BATCH_SIZE)
    print('Rest Corrected Test Accuracy: ',
          np.mean(np.vstack(true_labels).argmax(axis=-1) == rest_corr_pred.argmax(axis=-1)))

    x = x[correct_filter]
    filenames = filenames[correct_filter]

    writer = pd.ExcelWriter('event_results/results_obscured.xlsx')
    dest = {s: {t: dict() for t in TESTS} for s in TEST_SUBJECTS}

    # Consider high confidence points
    for conf in reversed(best_confidence):
        i = np.argmax(correct_pred[conf, :])

        rest_x = rest[[conf], ...]
        rest_pred = model.predict(normalize(rest_x))[0, i]

        test_x = x[[conf], ...]
        x_minus_rest = model.predict(normalize(test_x - normalize(rest_x)))[0, i]

        obs_event, obs_ends = obscuring_profile(model, test_x)

        dest[filenames[conf].parts[SUBJECT_PART]][filenames[conf].parts[TEST_PART]] = dict(test_x=correct_pred[conf, i],
                                                                                           rest=rest_pred,
                                                                                           x_minus_rest=x_minus_rest,
                                                                                           obscure_event=obs_event,
                                                                                           obscure_ends=obs_ends)

    for subject in TEST_SUBJECTS:
        pd.DataFrame.from_dict(dest[subject], orient='index').to_excel(writer, subject,
                                                                       columns=['test_x', 'rest', 'test_x_minus_rest'])
        pd.DataFrame.from_dict(dest[subject], orient='column')

    writer.save()
