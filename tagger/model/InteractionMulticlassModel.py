"""DeepSetsBinary model child class

Written 28/05/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from schema import Schema, And, Use, Optional

from tagger.model.common import choose_aggregator, initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel

from qkeras import QConv1D
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.layers import Activation, BatchNormalization, Multiply, GlobalAveragePooling1D, Concatenate, Flatten, Reshape

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('InteractionMulticlassModel')
class InteractionMulticlassModel(QKerasModel):

    """InteractionMulticlassModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "kernel_initializer" : str,
                                  "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])},
                "quantization_config" : QKerasModel.quantization_schema,
                "training_config" : QKerasModel.training_config_schema,
                "inputs_config": QKerasModel.inputs_config_schema,
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : str,
                                    "output_precision" : str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str}
            }
    )

    # (i, j) leading-jet indices for each row of the relations tensor, must match the dataloader
    PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    N_LEAD = 4

    def _incidence_matrix(self, n_effect: int) -> np.ndarray:
        """Fixed matrix that sums, for every leading jet, the effect vectors of the pairs it belongs to.

        Acts on the flattened effects (n_pairs * n_effect) -> (N_LEAD * n_effect).
        Flat index of effect k, feature d is k * n_effect + d; output index of jet i, feature d is i * n_effect + d.
        """
        n_pairs = len(self.PAIRS)
        A = np.zeros((n_pairs * n_effect, self.N_LEAD * n_effect), dtype=np.float32)
        for k, pair in enumerate(self.PAIRS):
            for i in pair:
                for d in range(n_effect):
                    A[k * n_effect + d, i * n_effect + d] = 1.0
        return A

    def build_model(self, input_shapes: tuple, output_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            input_shapes (tuple): Shape of the input
            output_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers (all-jets branch).
            relation_layers: List of number of nodes for each layer of the relation MLP f_R
                (acts on [pair, jet_i, jet_j]). Last entry is the size of the effect vector.
            object_layers: List of number of nodes for each layer of the object MLP f_O
                (acts on [jet_i, summed effects of the pairs containing jet_i]).
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
        """

        initialise_tensorflow(self.run_config['num_threads'])

        quantizer_bits = self.quantization_config['quantizer_bits']
        quantizer_bits_int = self.quantization_config['quantizer_bits_int']
        quantizer_alpha = self.quantization_config['quantizer_alpha_val']

        self.common_args = {
            'kernel_quantizer': quantized_bits(quantizer_bits, quantizer_bits_int, alpha=quantizer_alpha),
            'bias_quantizer': quantized_bits(quantizer_bits, quantizer_bits_int, alpha=quantizer_alpha),
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        relation_layers = self.model_config['conv1d_layers']
        object_layers = self.model_config['conv1d_layers']
        n_pairs = 6
        n_effect = relation_layers[-1]

        # Inputs
        inputs_coll1 = tf.keras.layers.Input(shape=input_shapes["jets_input"], name="jets_input")
        inputs_coll1_mask = tf.keras.layers.Input(shape=input_shapes["jets_mask"], name="jets_mask")
        inputs_lead = tf.keras.layers.Input(shape=input_shapes["lead_jets_input"], name="lead_jets_input")
        inputs_lead_mask = tf.keras.layers.Input(shape=input_shapes["lead_jets_mask"], name="lead_jets_mask")
        inputs_rel = tf.keras.layers.Input(shape=input_shapes["pairs_input"], name="pairs_input")
        inputs_rel_mask = tf.keras.layers.Input(shape=input_shapes["pairs_mask"], name="pairs_mask")
        inputs_event_features = tf.keras.layers.Input(shape=input_shapes["event_features_input"], name="event_features_input")

        # Batch Norm
        main_1 = BatchNormalization(name="norm_input")(inputs_coll1)
        lead = BatchNormalization(name="norm_leadjets")(inputs_lead)
        rel = BatchNormalization(name="norm_relations")(inputs_rel)
        main_features = BatchNormalization(name="norm_features")(inputs_event_features)

        # ---------------- All-jets branch (unchanged) ----------------
        for i, n in enumerate(self.model_config.get('conv1d_layers', [10, 10])):
            main_1 = QConv1D(filters=n, kernel_size=1, name=f"Conv1D_{i}_coll1", **self.common_args)(main_1)
            main_1 = QActivation(activation=quantized_relu(9, 2), name=f"relu_{i}_coll1")(main_1)

        main_1 = QActivation(activation=quantized_bits(18, 8), name="act_pool_coll1")(main_1)
        main_1 = Multiply(name="multiply_mask_coll1")([main_1, inputs_coll1_mask])
        main_1 = GlobalAveragePooling1D(name="avgpool_coll1")(main_1)

        # ---------------- Interaction network branch ----------------
        # Relation step: f_R on [pair_k, jet_i, jet_j], shared across pairs
        effects = rel
        for i, n in enumerate(relation_layers):
            effects = QConv1D(filters=n, kernel_size=1, name=f"Conv1D_{i}_fR", **self.common_args)(effects)
            effects = QActivation(activation=quantized_relu(9, 2), name=f"relu_{i}_fR")(effects)

        effects = QActivation(activation=quantized_bits(18, 8), name="act_effects")(effects)
        effects = Multiply(name="multiply_mask_effects")([effects, inputs_rel_mask])  # (B, n_pairs, n_effect)

        # Aggregation: sum effects of the pairs containing each leading jet.
        # Fixed (non-trainable) incidence matrix on the flattened effects, then reshape back to jets.
        incidence = self._incidence_matrix(n_effect)
        aggregate = QDense(
            self.N_LEAD * n_effect,
            use_bias=False,
            kernel_quantizer=quantized_bits(8, 2, alpha=1),  # 2 integer bits so that 1.0 is exactly representable
            kernel_initializer=tf.keras.initializers.Constant(incidence),
            name="Dense_aggregate_effects",
        )
        aggregate.trainable = False

        agg = Flatten(name="flatten_effects")(effects)
        agg = aggregate(agg)
        agg = Reshape((self.N_LEAD, n_effect), name="reshape_aggregated")(agg)
        agg = QActivation(activation=quantized_bits(18, 8), name="act_aggregated")(agg)

        # Object step: f_O on [jet_i, summed effects], shared across leading jets
        objects = Concatenate(axis=-1, name="concat_objects")([lead, agg])  # (B, N_LEAD, Fj + n_effect)
        for i, n in enumerate(object_layers):
            objects = QConv1D(filters=n, kernel_size=1, name=f"Conv1D_{i}_fO", **self.common_args)(objects)
            objects = QActivation(activation=quantized_relu(9, 2), name=f"relu_{i}_fO")(objects)

        objects = QActivation(activation=quantized_bits(18, 8), name="act_pool_objects")(objects)
        objects = Multiply(name="multiply_mask_objects")([objects, inputs_lead_mask])
        objects = GlobalAveragePooling1D(name="avgpool_objects")(objects)

        # Pool the pair effects themselves as well (already masked)
        effects_pooled = GlobalAveragePooling1D(name="avgpool_effects")(effects)

        # FF + event features
        event_id = Concatenate(name="concat")([main_1, objects, effects_pooled, main_features])

        for i, n in enumerate(self.model_config.get('classifier_layers', [64, 32])):
            event_id = QDense(n, name=f"Dense_{i}_jetID", **self.common_args)(event_id)
            event_id = QActivation(activation=quantized_relu(9, 2), name=f"relu_{i}_jetID")(event_id)

        event_id = QDense(output_shape, name="Dense_out_jetID", **self.common_args)(event_id)

        output = Activation("softmax", name="event_id_output")(event_id)

        # Define the model
        self.event_model = tf.keras.Model(
            inputs=[
                inputs_coll1, inputs_lead, inputs_rel, inputs_event_features,
                inputs_coll1_mask, inputs_lead_mask, inputs_rel_mask,
            ],
            outputs=output,
        )

        print(self.event_model.summary())

    def firmware_convert(self, firmware_dir: str, build: bool = False):
        """Run the hls4ml model conversion

        Args:
            firmware_dir (str): Where to save the firmware
            build (bool, optional): Run the full hls4ml build? Or just create the project. Defaults to False.
        """

        # Remove the old directory if it exists
        hls4ml_outdir = firmware_dir + '/' + self.firmware_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        # Create default config
        config = hls4ml.utils.config_from_keras_model(self.event_model, granularity='name')
        config['IOType'] = 'io_parallel'
        config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']

        # Configuration for conv1d layers
        # hls4ml does not !!! automatically figure out the paralellization factor, this leads to csim, hdl sim errors
        config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
        config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16

        # Additional config
        for layer in self.event_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["event_id_output"]["Precision"]["result"] = self.firmware_config['output_precision']
        config["LayerName"]["event_id_output"]["Implementation"] = "latency"

        # Write HLS
        self.hls_event_model = hls4ml.converters.convert_from_keras_model(
            self.event_model,
            backend='Vitis',
            project_name=self.firmware_config['project_name'],
            clock_period=self.firmware_config['clock_period'],
            hls_config=config,
            output_dir=f'{hls4ml_outdir}',
            part= self.firmware_config['fpga_part'],
        )

        # Compile the project
        self.hls_event_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir + '/config.json', 'w') as fp:
            json.dump(config, fp)

        if build:
            # build the project
            self.hls_event_model.build(csim=False, reset=True)
