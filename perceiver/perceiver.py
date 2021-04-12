import tensorflow as tf
from einops import rearrange, repeat
from .prenorm import PreNorm
from .feedforward import FeedForward
from .attention import Attention
from .fourier_encode import fourier_encode

class Perceiver(tf.keras.Model):
    def __init__(self,
                 num_freq_bands,
                 depth,
                 max_freq,
                 freq_base=2,
                 input_channels=3,
                 input_axis=2,
                 num_latents=512,
                 latent_dim=512,
                 cross_heads=1,
                 latent_heads=8,
                 cross_dim_head=64,
                 latent_dim_head=64,
                 num_classes=1000,
                 attn_dropout=0.,
                 ff_dropout=0.):

        super(Perceiver, self).__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels
        self.latents = tf.Variable(tf.random.normal([num_latents, latent_dim]))

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout),
                                         context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        self.existing_layers = list()
        for i in range(depth):

            self.existing_layers = get_cross_attn()(self.existing_layers)
            self.existing_layers = get_cross_ff()(self.existing_layers)
            self.existing_layers = get_latent_attn()(self.existing_layers)
            self.existing_layers = get_latent_ff()(self.existing_layers)

        self.to_logits = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Dense(num_classes, input_dim=latent_dim)
        ])

    def call(self, data, mask=None):
        b, *axis, _ = data.shape
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        axis_pos = list(map(lambda size: tf.linspace(-1., 1., num=size), axis))
        pos = tf.stack(tf.meshgrid(
            *axis_pos,
            indexing='ij'
        ), axis=-1)

        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)

        data = tf.concat((data, enc_pos), axis=-1)
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        x = tf.keras.Sequential(self.existing_layers)(x)

        x = tf.math.reduce_mean(x, axis=-2)
        return self.to_logits(x)
    