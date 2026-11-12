"""TransformerMulticlassModel model child class

Unquantised, unpruned transformer for event tagging. Float upper-limit reference for
DeepSetsMulticlassModel (not intended for hls4ml conversion).

Tokens per event: jets (collection0) + jet pairs (collection1) + 1 event token (event features).
Each pair token carries the rank embeddings of the two jets that formed it. The event token
(plus a mean over all tokens) is the readout. No masking: padded objects are treated as ordinary tokens.
"""

import numpy as np
import tensorflow as tf
from schema import Schema, And, Use, Optional

from tagger.model.common import initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel

from tensorflow.keras.layers import (
    Activation, BatchNormalization, Concatenate, Dense, Dropout,
    Input, LayerNormalization, MultiHeadAttention, Reshape,
)

# The base class may wrap the model with tfmot magnitude pruning (initial/final_sparsity in the config).
# tfmot refuses unknown custom layers unless they declare their prunable weights, so declare none:
# the custom layers below (incl. attention) are left dense; only the plain Dense layers can be pruned.
try:
    from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer as _PrunableBase
except ImportError:  # tfmot not installed -> nothing to satisfy
    class _PrunableBase:
        pass


class _NoPrune(_PrunableBase):
    def get_prunable_weights(self):
        return []


# Which (leading-jet) indices form each of the 6 pairs. CHECK against how jetpairs is built.
PAIR_IDX = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_I = np.array([p[0] for p in PAIR_IDX], dtype="int32")
PAIR_J = np.array([p[1] for p in PAIR_IDX], dtype="int32")
N_PAIRS = len(PAIR_IDX)


# ----------------------------------------------------------------------------- custom layers
@tf.keras.utils.register_keras_serializable(package="tagger")
class AddIdentity(_NoPrune, tf.keras.layers.Layer):
    """Jet-rank embedding on jet tokens; pair tokens get both constituents' rank embeddings + a pair-id embedding."""

    def __init__(self, n_jets, d, **kw):
        super().__init__(**kw)
        self.n_jets, self.d = n_jets, d

    def build(self, _):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.rank = self.add_weight(name="rank", shape=(self.n_jets, self.d), initializer=init)
        self.pair = self.add_weight(name="pair", shape=(N_PAIRS, self.d), initializer=init)

    def call(self, inputs):
        jet_h, pair_h = inputs
        jet_h = jet_h + tf.expand_dims(self.rank, 0)
        pair_h = (pair_h + tf.expand_dims(tf.gather(self.rank, PAIR_I), 0)
                  + tf.expand_dims(tf.gather(self.rank, PAIR_J), 0)
                  + tf.expand_dims(self.pair, 0))
        return jet_h, pair_h

    def get_config(self):
        return {**super().get_config(), "n_jets": self.n_jets, "d": self.d}


@tf.keras.utils.register_keras_serializable(package="tagger")
class TransformerBlock(_NoPrune, tf.keras.layers.Layer):
    """Pre-LN transformer encoder block with stochastic depth."""

    def __init__(self, d, heads, ff_mult=4, dropout=0.1, drop_path=0.0,
                 kernel_initializer="glorot_uniform", **kw):
        super().__init__(**kw)
        self.cfg = dict(d=d, heads=heads, ff_mult=ff_mult, dropout=dropout,
                        drop_path=drop_path, kernel_initializer=kernel_initializer)
        self.drop_path = drop_path
        self.ls_init = 0.1
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(heads, d // heads, dropout=dropout,
                                      kernel_initializer=kernel_initializer)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.ff = tf.keras.Sequential([
            Dense(d * ff_mult, activation="gelu", kernel_initializer=kernel_initializer),
            Dropout(dropout),
            Dense(d, kernel_initializer=kernel_initializer),
            Dropout(dropout),
        ])

    def build(self, input_shape):
        # LayerScale: learnable per-channel scale on each residual branch, starts small so a
        # deep stack trains stably at a flat lr of 1e-3 without warmup.
        d = self.cfg["d"]
        init = tf.keras.initializers.Constant(self.ls_init)
        self.g1 = self.add_weight(name="ls_attn", shape=(d,), initializer=init)
        self.g2 = self.add_weight(name="ls_ff", shape=(d,), initializer=init)
        super().build(input_shape)

    def _dp(self, h, training):
        if not training or self.drop_path == 0:
            return h
        keep = 1.0 - self.drop_path
        m = tf.floor(keep + tf.random.uniform([tf.shape(h)[0], 1, 1])) / keep
        return h * m

    def call(self, x, training=False):
        h = self.ln1(x)
        x = x + self._dp(self.g1 * self.mha(h, h, training=training), training)
        x = x + self._dp(self.g2 * self.ff(self.ln2(x), training=training), training)
        return x

    def get_config(self):
        return {**super().get_config(), **self.cfg}


@tf.keras.utils.register_keras_serializable(package="tagger")
class Readout(_NoPrune, tf.keras.layers.Layer):
    """concat(event token, mean over all tokens)."""

    def call(self, x):
        return tf.concat([x[:, -1], tf.reduce_mean(x, axis=1)], -1)


# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('TransformerMulticlassModel')
class TransformerMulticlassModel(QKerasModel):

    """TransformerMulticlassModel class

    Args:
        QKerasModel (_type_): Base class (used for its training/eval/saving machinery; no quantisation applied)
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "kernel_initializer" : str,
                                  # transformer hyperparameters; defaults are applied in build_model
                                  Optional("d_model") : And(int, lambda s: s > 0),
                                  Optional("n_heads") : And(int, lambda s: s > 0),
                                  Optional("n_layers") : And(int, lambda s: s > 0),
                                  Optional("ff_mult") : And(int, lambda s: s > 0),
                                  Optional("dropout") : And(Use(float), lambda s: 0.0 <= s < 1.0),
                                  Optional("drop_path") : And(Use(float), lambda s: 0.0 <= s < 1.0),
                                  # legacy DeepSets keys: accepted so the old yaml still validates, ignored here
                                  Optional("conv1d_layers") : list,
                                  Optional("aggregator") : str},
                # kept so the QKerasModel base class finds what it expects; not used to build layers here
                "quantization_config" : QKerasModel.quantization_schema,
                "training_config" : QKerasModel.training_config_schema,
                "inputs_config": QKerasModel.inputs_config_schema,
                # no firmware conversion for this model
                Optional("firmware_config") : dict,
            }
    )

    @staticmethod
    def _embed(x, d, init, name):
        """Per-token MLP embedding."""
        x = Dense(d, activation="gelu", kernel_initializer=init, name=f"{name}_1")(x)
        return Dense(d, kernel_initializer=init, name=f"{name}_2")(x)

    def build_model(self, input_shapes: tuple, output_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            input_shapes (tuple): Shape of the input
            output_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            d_model, n_heads, n_layers, ff_mult: transformer size.
            dropout, drop_path: regularisation (drop_path ramps linearly over depth).
        """

        initialise_tensorflow(self.run_config['num_threads'])

        mc = self.model_config
        init = mc['kernel_initializer']
        d = mc.get('d_model', 128)
        n_heads = mc.get('n_heads', 8)
        n_layers = mc.get('n_layers', 6)
        ff_mult = mc.get('ff_mult', 4)
        dropout = float(mc.get('dropout', 0.1))
        drop_path = float(mc.get('drop_path', 0.1))
        n_jets = input_shapes["jets_input"][0]
        n_pairs = input_shapes["pairs_input"][0]

        if n_pairs != N_PAIRS:
            raise ValueError(f"pairs collection has {n_pairs} entries, expected {N_PAIRS} (see PAIR_IDX)")

        # Inputs
        inputs_jets = Input(shape=input_shapes["jets_input"], name="jets_input")
        inputs_pairs = Input(shape=input_shapes["pairs_input"], name="pairs_input")
        inputs_event_features = Input(shape=input_shapes["event_features_input"], name="event_features_input")

        # Batch Norm
        jets = BatchNormalization(name="norm_input")(inputs_jets)
        pairs = BatchNormalization(name="norm_pairs")(inputs_pairs)
        feats = BatchNormalization(name="norm_features")(inputs_event_features)

        # Token embeddings
        jet_h = self._embed(jets, d, init, "jet_embed")
        pair_h = self._embed(pairs, d, init, "pair_embed")
        evt_h = self._embed(feats, d, init, "event_embed")
        jet_h, pair_h = AddIdentity(n_jets, d, name="identity")([jet_h, pair_h])

        x = Concatenate(axis=1, name="tokens")([jet_h, pair_h, Reshape((1, d))(evt_h)])

        # Transformer encoder
        for i in range(n_layers):
            dp = drop_path * i / max(n_layers - 1, 1)
            x = TransformerBlock(d, n_heads, ff_mult, dropout, dp,
                                 kernel_initializer=init, name=f"block_{i}")(x)
        x = LayerNormalization(epsilon=1e-6, name="final_ln")(x)

        # Readout + classifier head
        event_id = Readout(name="readout")(x)
        event_id = Dense(256, activation="gelu", kernel_initializer=init, name="Dense_1_jetID")(event_id)
        event_id = Dropout(dropout)(event_id)
        event_id = Dense(64, activation="gelu", kernel_initializer=init, name="Dense_3_jetID")(event_id)
        event_id = Dense(output_shape, kernel_initializer=init, name="Dense_4_jetID")(event_id)

        output = Activation("softmax", name="event_id_output", dtype="float32")(event_id)

        # Define the model
        self.event_model = tf.keras.Model(
            inputs=[inputs_jets, inputs_pairs, inputs_event_features],
            outputs=output,
        )

        print(self.event_model.summary())

    @staticmethod
    def make_optimizer(steps_per_epoch: int, epochs: int, peak_lr: float = 5e-4,
                       weight_decay: float = 1e-4, warmup_epochs: int = 3):
        """AdamW + linear warmup / cosine decay + EMA weights. Transformers train noticeably worse
        with a flat Adam lr, so hook this into the base class compile step if it lets you.
        Needs TF >= 2.13 (CosineDecay warmup_target, AdamW, use_ema)."""
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-6,
            warmup_target=peak_lr,
            warmup_steps=warmup_epochs * steps_per_epoch,
            decay_steps=(epochs - warmup_epochs) * steps_per_epoch,
            alpha=1e-3,
        )
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay,
                                         clipnorm=1.0, use_ema=True, ema_momentum=0.999)

    def firmware_convert(self, firmware_dir: str, build: bool = False):
        raise NotImplementedError(
            "TransformerMulticlassModel is an unquantised reference model and has no hls4ml conversion."
        )
