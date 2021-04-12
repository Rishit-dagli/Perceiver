import tensorflow as tf

class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, context_dim=None):
        super(PreNorm, self).__init__()
        self.fn = fn
        tf.print(dim)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, **kwargs):
        tf.print(x)
        x = self.norm(x)

        return self.fn(x)
