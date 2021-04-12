import tensorflow as tf


class GEGLU(tf.keras.layers.Layer):
    def call(self, x):
        x, gates = tf.split(x, 2, axis=-1)
        return x * tf.nn.gelu(gates)
