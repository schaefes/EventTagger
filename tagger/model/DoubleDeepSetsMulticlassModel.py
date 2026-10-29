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
from tensorflow.keras.layers import Activation, BatchNormalization, Multiply, GlobalAveragePooling1D, Concatenate

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('DoubleDeepSetsMulticlassModel')
class DoubleDeepSetsMulticlassModel(QKerasModel):

    """DoubleDeepSetsMulticlassModel class

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

    def build_model(self, input_shapes: tuple, output_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            input_shapes (tuple): Shape of the input
            output_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers.
            aggregator: String that specifies the type of aggregator to use after the conv1D net.
        """

        initialise_tensorflow(self.run_config['num_threads'])

        self.common_args = {
            'kernel_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'bias_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        # Inputs
        inputs_jets = tf.keras.layers.Input(shape=input_shapes["jets_input"], name="jets_input")
        inputs_pairs = tf.keras.layers.Input(shape=input_shapes["pairs_input"], name="pairs_input")
        inputs_jets_mask = tf.keras.layers.Input(shape=input_shapes["jets_mask"], name="jets_mask")
        inputs_pairs_mask = tf.keras.layers.Input(shape=input_shapes["pairs_mask"], name="pairs_mask")
        inputs_event_features = tf.keras.layers.Input(shape=input_shapes["event_features_input"], name="event_features_input")

        # Batch Norm
        main_1 = BatchNormalization(name="norm_input")(inputs_jets)
        main_2 = BatchNormalization(name="norm_input_2")(inputs_pairs)
        main_features = BatchNormalization(name="norm_features")(inputs_event_features)

        # Conv1D layers
        main_1 = QConv1D(filters=10, kernel_size=1, name="Conv1D_1_jets", **self.common_args)(main_1)
        main_1 = QActivation(activation=quantized_relu(9, 2), name="relu_1_jets")(main_1)

        main_1 = QConv1D(filters=10, kernel_size=1, name="Conv1D_3_jets", **self.common_args)(main_1)
        main_1 = QActivation(activation=quantized_relu(9, 2), name="relu_3_jets")(main_1)

        main_1 = QActivation(activation=quantized_bits(18,8), name="act_pool_jets")(main_1)

        # Aggregation
        main_1 = Multiply(name="multiply_mask_jets")([main_1, inputs_jets_mask])
        main_1 = GlobalAveragePooling1D(name="avgpool_jets")(main_1)

        # Conv1D layers
        main_2 = QConv1D(filters=10, kernel_size=1, name="Conv1D_1_pairs", **self.common_args)(main_2)
        main_2 = QActivation(activation=quantized_relu(9, 2), name="relu_1_pairs")(main_2)

        main_2 = QConv1D(filters=10, kernel_size=1, name="Conv1D_3_pairs", **self.common_args)(main_2)
        main_2 = QActivation(activation=quantized_relu(9, 2), name="relu_3_pairs")(main_2)

        main_2 = QActivation(activation=quantized_bits(18,8), name="act_pool_pairs")(main_2)

        # Aggregation
        main_2 = Multiply(name="multiply_mask_pairs")([main_2, inputs_pairs_mask])
        main_2 = GlobalAveragePooling1D(name="avgpool_pairs")(main_2)

        avgpools = Concatenate(name="concat_avgpools")([main_1, main_2])

        # FF + event features
        event_id = Concatenate(name="concat")([avgpools, main_features])

        event_id = QDense(64, name="Dense_1_jetID", **self.common_args)(event_id)
        event_id = QActivation( activation=quantized_relu(9, 2), name="relu_1_jetID")(event_id)

        event_id = QDense(32, name="Dense_3_jetID", **self.common_args)(event_id)
        event_id = QActivation( activation=quantized_relu(9, 2), name="relu_3_jetID")(event_id)

        event_id = QDense(output_shape, name="Dense_4_jetID", **self.common_args)(event_id)

        output = Activation("softmax", name="event_id_output")(event_id)

        # Define the model using both branches
        self.event_model = tf.keras.Model(inputs=[inputs_jets, inputs_pairs, inputs_event_features, inputs_jets_mask, inputs_pairs_mask], outputs=output)

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
        input_names = [inp.name for inp in self.event_model.inputs]
        for l in input_names:
            if 'input' in l:
                config['LayerName'][l]['Precision']['result'] = self.firmware_config['input_precision']
            elif 'mask' in l:
                config['LayerName'][l]['Precision']['result'] = "ap_uint<1>"
            else:
                raise ValueError(f"Unknown input name {l} in model inputs. Please check the model inputs.")

        # Configuration for conv1d layers
        # hls4ml does not !!! automatically figure out the paralellization factor, this leads to csim, hdl sim errors
        for layer in self.event_model.layers:
            if isinstance(layer, QConv1D):
                conv1d_type = layer.name.split('_')[-1]
                config['LayerName'][layer.name]['ParallelizationFactor'] = self.inputs_config['objects'][conv1d_type][1]

        # Additional config
        for layer in self.event_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add", "Multiply"]:
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
