import tensorflow as tf
from einops import rearrange, repeat


class Attention(tf.keras.layers.Layer):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_queries = tf.keras.layers.Dense(
            inner_dim, input_dim=query_dim, use_bias=False
        )
        self.to_keys_values = tf.keras.layers.Dense(
            inner_dim * 2, input_dim=query_dim, use_bias=False
        )

        self.to_out = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(inner_dim, input_dim=query_dim),
                tf.keras.layers.Dropout(dropout),
            ]
        )

        def call(self, x, context=None, mask=None):
            h = self.heads
            queries = self.to_queries(x)

            if context is None:
                context = x

            kv = self.to_keys_values(context)
            keys, values = tf.split(kv, num_or_size_splits=2, axis=-1)
            queries, keys, values = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
                (queries, keys, values),
            )

            sim = tf.einsum("b i d, b j d -> b i j", queries, keys) * self.scale

            if mask is not None:
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -tf.experimental.numpy.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim = tf.where(tf.bitwise.invert(mask), max_neg_value, sim)

            attn = tf.nn.softmax(sim, axis=-1)
            out = tf.einsum("b i j, b j d -> b i d", attn, values)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
            out = self.to_out(out)

            return out
