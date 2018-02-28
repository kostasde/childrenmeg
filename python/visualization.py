import argparse
import utils
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import *
from pathlib import Path
from mne.viz import plot_topomap


def _parse_layers(model):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    conv_dict = dict([(layer, layer_dict[layer]) for layer in layer_dict if 'conv' in layer])
    spat_dict = dict([(layer, conv_dict[layer]) for layer in conv_dict if 'spatial' in layer])
    temp_dict = dict([(layer, conv_dict[layer]) for layer in conv_dict if 'temporal' in layer])

    # Last spatial layer should be the "largest" when sorted
    spatial_out = spat_dict[sorted(spat_dict.keys(), reverse=True)[0]]
    # Output layer
    out_layer = layer_dict.get('OUT', None)
    if out_layer is None:
        print('No output layer found..!')
        if input('Continue?(y/n)').lower() != 'y':
            print('Stopping here')
            exit(-1)

    return layer_dict, conv_dict, spat_dict, temp_dict, spatial_out, out_layer


def maximize_activation(input, output, lr=0.2, input_generator=utils.pink_noise, l2_reg=0.05, max_steps=1e4,
                        verbose=True):
    """
    Given input and output tensors, perform gradient ascent from data created by input generator until there is some
    semblance of convergence, or a maximum number of iterations have occurred

    Some of this code taken from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    :param input_generator: function with a single argument for size/shape, and returns a numpy array with said shape
    :return: The new input data achieved, as np.array
    """
    activations = list()
    for filter_index in range(output._keras_shape[-1]):
        if verbose:
            print('Filter: {0}/{1}'.format(filter_index+1, output._keras_shape[-1]))
        loss = K.mean(output[..., filter_index]) - l2_reg * K.mean(K.square(input))
        grads = K.gradients(loss, input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + K.constant(1e-5))

        # this function returns the loss and grads given the input picture
        iterate = K.function([input, K.learning_phase()], [loss, grads])

        # Create a noise input signal
        in_data = input_generator(input._keras_shape[1:])[np.newaxis, :]

        patience = 5
        best = -np.inf
        steps = 0.0
        while patience > 0 and steps < max_steps:
            loss_value, grads_value = iterate([in_data, 0])
            if verbose:
                print('Loss:', loss_value)
            in_data += grads_value * lr
            if loss_value > best:
                best = loss_value
                patience = 5
            else:
                patience -= 1
            steps += 1
        else:
            print('Best Loss:', best)
        activations.append(in_data.squeeze())

    return np.stack(activations, axis=-1)


def activations_scnn(args, model: SCNN):
    """
    Calculate the maximum activations for the scnn model: the spatial components, the spectral components of the
    temporal filters and the spectral behaviour of each component that maximally activates output classes.
    :param args: The args parsed by the argument parser
    :return:
    """
    if args.activ is not None and args.activ.exists():
        if args.maximize:
            return pickle.load(args.activ.open('rb'))
        else:
            in_data = np.load(str(args.activ))[np.newaxis, :]
    to_return = dict()

    model.load_weights(args.saved_model)
    model.summary()

    layer_dict, conv_dict, spat_dict, temp_dict, spatial_out, out_layer = _parse_layers(model)

    # Max activations from the input to each spatial filter
    for layer in spat_dict:
        if args.maximize:
            print('Calculating maximum for layer: ', layer)
            act = maximize_activation(model.input, spat_dict[layer].output)
            act = np.mean(act, axis=0)
        else:
            act = np.empty
        # Compress the redundant temporal dimension
        to_return[layer] = act

    # Max activations for spatial filters starting from the mixed channels
    for layer in temp_dict:
        if args.maximize:
            print('Calculating maximum for layer: ', layer)
            act = maximize_activation(spatial_out.output, temp_dict[layer].output)
        else:
            act = K.function([model.input, K.learning_phase()], [temp_dict[layer].output])([in_data, 0])[0].squeeze()
        to_return[layer] = act

    # output activations
    if out_layer is not None:
        if args.maximize:
            print('Calculating maximum for each output class after spatial mixing.')
            to_return['OUT'] = maximize_activation(spatial_out.output, out_layer.output)
        else:
            to_return['OUT'] = np.expand_dims(K.function([model.input, K.learning_phase()],
                                                         [spat_dict[sorted(spat_dict.keys())[-1]].output])
                                              ([in_data, 0])[0].squeeze(), -1)

    print('Completed Maximum Calculations.')
    if args.maximize:
        pickle.dump(to_return, args.activ.open('wb'))
    return to_return


def save_viz_scnn(args, activations, model):
    import matplotlib.pyplot as plt

    layer_dict, conv_dict, spat_dict, temp_dict, spatial_out, out_layer = _parse_layers(model)

    chans = np.load(args.chans.open('rb'))

    if args.maximize:
        topomaps = activations[spatial_out.name]
        for i in range(topomaps.shape[-1]):
            im, cn = plot_topomap(topomaps[..., i], chans, show=True)
            save_loc = args.save_viz / spatial_out.name
            save_loc.mkdir(parents=True, exist_ok=True)
            plt.title('Component {0}'.format(i))
            plt.savefig(str(save_loc / 'component_{0}.png'.format(i)))
            plt.clf()

        for t in temp_dict:
            fil_act = activations[t]
            for f in range(fil_act.shape[-1]):
                for c in range(fil_act.shape[-2]):
                    directory = args.save_viz / t / 'filter_{0}'.format(f)
                    directory.mkdir(parents=True, exist_ok=True)
                    plt.specgram(fil_act[..., c, f].squeeze(), Fs=200, cmap='bwr', NFFT=64, noverlap=50)
                    plt.colorbar()
                    plt.title('{0} {1} Component {2}'.format(t, f, c))
                    plt.savefig(str(directory / 'component_{0}'.format(c)))
                    plt.clf()

    if out_layer is not None:
        out_act = activations[out_layer.name]
        for c in range(out_act.shape[-1]):
            for f in range(out_act.shape[-2]):
                directory = args.save_viz / 'OUT' / 'class_{0}'.format(c)
                directory.mkdir(parents=True, exist_ok=True)
                plt.specgram(out_act[..., f, c].squeeze(), Fs=200, cmap='bwr', NFFT=64, noverlap=50)
                plt.colorbar()
                plt.title('{0} Output Class {1} Component {2}'.format(out_layer.name, c, f))
                plt.ylabel('Frequency (Hz)')
                plt.xlabel('Time (s)')
                plt.savefig(str(directory / 'component_{0}'.format(f)))
                plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce max-activations and associated visualizations for SCNN and '
                                                 'Attention LSTM models.')
    parser.add_argument('model', choices=['SCNN'], help='Which model to produce output for.')
    parser.add_argument('activ', help='Previously stored file with max activations to be calculated for '
                                                  'model, or datapoint to calculate outputs for.')
    parser.add_argument('--saved-model', help='Model weights that will be loaded to calculate activations.')
    parser.add_argument('--hyper-params', help='Model hyperparameters.', default=None)
    parser.add_argument('--save-viz', help='Directory to save all vizualizations to. If not provided, will not '
                                           'determine vizualizations.')
    parser.add_argument('--chans', default='../CTF151.npy', help='Location of the CTF151 channel locations file.')

    args = parser.parse_args()
    activations = None

    if args.hyper_params is not None:
        args.hyper_params = pickle.load(open(args.hyper_params, 'rb'))
        print('Loaded provided Hyper-parameters')
        print(args.hyper_params)

    if args.activ is not None:
        args.activ = Path(args.activ)
        if not args.activ.exists() and args.saved_model is None:
            print('Activation file is empty, need to provide model weights to calculate activations.')
            exit(-1)

        args.maximize = args.activ.suffix == '.pkl'
        if args.model == 'SCNN':
            print('Calculating SCNN activations...')
            model = SCNN((700, 151), 2, params=args.hyper_params)
            # FIXME Hardcoded
            activations = activations_scnn(args, model)
        else:
            raise NotImplementedError('Currently do not support: ' + args.model)

    if args.save_viz is not None:
        args.save_viz = Path(args.save_viz)
        args.chans = Path(args.chans)

        if not args.save_viz.exists():
            print('The directory ', args.save_viz, ' does not exist.')
            if input('Create directory?(y/n)').lower() == 'y':
                args.save_viz.mkdir(parents=True)
            else:
                print('Proceeding without saving activations...')
                args.save_viz = None
        if not args.save_viz.is_dir():
            print(args.save_viz, ' must be a directory. Exiting.')
            exit(-1)
        if not args.chans.exists():
            print('Channel locations file {0} not found'.format(str(args.chans)))

        if args.model == 'SCNN':
            print('Saving SCNN visualizations...')
            save_viz_scnn(args, activations, model)

    print('Finished.')

