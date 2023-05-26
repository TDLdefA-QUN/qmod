# Import libraries that will be needed for the lab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import os, datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.activations import relu
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model

import pickle

import random


class Encoder2D(tf.keras.Model):
    """
    A deep autoencoder for finding modes, operating on 2D data+

    """

    def __init__(self, units_full=128,
                 units_hidden=128, num_layers=3,
                 actlay_config=dict(activation="elu",
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.units_full = units_full

        # Construct a list of the layers used in this block
        self.block_layers = []
        # The first (num_layers-1) layers will be dense, activated layers
        for i in range(num_layers - 1):
            self.block_layers.append(tf.keras.layers.Dense(units_hidden,
                                                           **actlay_config))
        # The final layer does not have activation
        self.block_layers.append(tf.keras.layers.Dense(units_full,
                                                       **linlay_config))
        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin
        # The encoder will consist of a number of dense layers that decrease in size
        # as we taper down towards the bottleneck of the network, the latent space
        input_data = Input(shape=(input_dim,), name='encoder_input')

        # hidden layers
        encoder = Dense(96, activation='tanh', name='encoder_1')(input_data)
        encoder = Dropout(.1)(encoder)
        encoder = Dense(64, activation='tanh', name='encoder_2')(encoder)
        encoder = Dropout(.1)(encoder)
        encoder = Dense(48, activation='tanh', name='encoder_3')(encoder)
        encoder = Dropout(.1)(encoder)
        encoder = Dense(16, activation='tanh', name='encoder_4')(encoder)
        encoder = Dropout(.1)(encoder)

        # bottleneck layer
        latent_encoding = Dense(latent_dim, activation='linear', name='latent_encoding')(encoder)

    def call(self, input_tensor):
        x = input_tensor
        x = tf.reshape(x, shape=(-1, self.units_full))
        for layer in self.block_layers:
            x = layer(x)
        if self.add_init_fin:
            x += tf.reshape(input_tensor, shape=(-1, self.units_full))
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_layers": self.num_layers,
                "actlay_config": self.actlay_config,
                "linlay_config": self.linlay_config,
                "units_full": self.units_full,
                "layers": self.layers,
                "add_layer": self.add_layer,
                "add_init_fin": self.add_init_fin}
