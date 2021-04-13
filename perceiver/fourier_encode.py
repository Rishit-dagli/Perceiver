import tensorflow as tf
import math


def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = tf.expand_dims(x, -1)
    x = tf.cast(x, dtype=tf.float32)
    orig_x = x
    scales = tf.experimental.numpy.logspace(
        1.0,
        math.log(max_freq / 2) / math.log(base),
        num=num_bands,
        base=base,
        dtype=tf.float32,
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
    x = tf.concat((x, orig_x), axis=-1)
    return x
