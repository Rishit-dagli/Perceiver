import tensorflow as tf
from .geglu import GEGLU


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([tf.keras.layers.Dense(dim * mult, input_dim=dim),
                                        GEGLU(),
                                        tf.keras.layers.Dropout(dropout),
                                        tf.keras.layers.Dense(dim, input_dim=dim * mult)
                                        ])

    def call(self, inputs):
        return self.net(inputs)
