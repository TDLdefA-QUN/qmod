### Script for constructing model (this can be changed later.)

from tensorflow import keras
import tensorflow as tf
import numpy as np

from DenseEncoderDecoder import DenseEncoderDecoder


class autoencoder(keras.Model):
    def __init__(self,
                 encoder_block=DenseEncoderDecoder(),
                 decoder_block=DenseEncoderDecoder(),
                 **kwargs):
        super().__init__(**kwargs)  # handles standard args (e.g., name)


        # u autoencoder
        self.encoder = encoder_block
        self.decoder = decoder_block



    def call(self, input):

        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded
