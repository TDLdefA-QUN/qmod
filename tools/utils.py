import random as r
import json
import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

from cubes2npy import CubeFiles

# Add the architecture path for the autoencoder
sys.path.append("../architecture/")
from autoencoder import autoencoder

def construct_network(**architecture_config):
    return autoencoder(**architecture_config)

def evaluate_modes(save_prefix, train_data,
                            train_opts, network_config):

    # Gather the relevant training options
    epochs = train_opts['epochs']
    num_init_models = train_opts['num_init_models']
    loss_fn = train_opts['loss_fn']
    opt = train_opts['optimizer']
    optimizer_opts = train_opts['optimizer_opts']
    batch_size = train_opts['batch_size']

    # Set up results dictionary
    results = {'full_hist': [],
               'aec_hist': [],
               'lr': [],
               'best_loss': [],
               'model_path': []}

    # For loop for generating, training, and evaluating the initial model pool
    for i in range(num_init_models):
        # Randomly selected learning rate
        lr = 10**(-r.uniform(2, 5))

        # Create a model, initially only to train autoencoders!
        model = construct_network(**network_config)

        # Compile the model
        model.compile(loss=4*[loss_fn], optimizer=opt(lr=lr, **optimizer_opts))

        # Set up the Callback function
        checkpoint_path_aec = save_prefix + 'checkpoint_aec_{}'.format(i)
        cbs_aec = [keras.callbacks.ModelCheckpoint(checkpoint_path_aec,
                                                   save_weights_only=True,
                                                   monitor='val_loss',
                                                   save_best_only=True)]

        #  Fit autoencoder
        aec_hist = model.fit(x=train_data, y=train_data,
                             callbacks=cbs_aec, batch_size=batch_size,
                             epochs=epochs, verbose=2)
        # Re-load weights with best validation loss
        model.load_weights(checkpoint_path_aec)

        # Save the model
        model_path = save_prefix + "model_{}".format(i)
        model.save(model_path)
        # Evaluate model at checkpoint
        best_loss = model.evaluate(x=train_data, y=train_data, verbose=False)
        # Append the results to the model list
        results['aec_hist'].append(aec_hist.history.copy())
        results['lr'].append(lr)
        results['best_loss'].append(best_loss[0])
        results['model_path'].append(model_path)

        # Delete the model variable and clear_session to remove any graph
        del model
        tf.keras.backend.clear_session()

    # Select the best model from the loop
    best_model_idc = np.argmin(results['best_loss'])
    best_model_path = results['model_path'][best_model_idc]

    # Return the best model's path
    return results, best_model_path


def run_experiment(train_data, random_seed, expt_name: str, training_options: dict, network_config: dict):
    # Assign a random number generator seed for learning rates
    r.seed(random_seed)

    # Set the prefix for where to save the results/checkpointed models
    save_prefix = '../model_weights/{}/'.format(expt_name)
    # Step 1 -- Run the autoencoder
    hist, model_path = evaluate_modes(save_prefix,train_data,training_options,network_config)

