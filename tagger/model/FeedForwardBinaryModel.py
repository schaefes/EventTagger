"""FeedForwardBinary model child class

Written 28/05/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from schema import Schema, And, Use, Optional

from tagger.model.common import initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel

from qkeras import QConv1D
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.layers import Activation, BatchNormalization

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('FeedForwardBinaryModel')
class FeedForwardBinaryModel(QKerasModel):

    """FeedForwardBinaryModel class

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
                                  "classification_layers" : list,
                                  "regression_layers" : list,
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
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
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

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=input_shapes["model_input"], name="model_input")

        main = BatchNormalization(name="norm_input")(inputs)

        event_id = QDense(32, name="Dense_1_jetID", **self.common_args)(main)
        event_id = QActivation(activation=quantized_relu(9, 2), name="relu_1")(event_id)

        # (kept identical to your original code)
        event_id = QDense(32, name="Dense_2_jetID", **self.common_args)(event_id)
        event_id = QActivation(activation=quantized_relu(9, 2), name="relu_2")(event_id)

        event_id = QDense(32, name="Dense_3_jetID", **self.common_args)(event_id)
        event_id = QActivation(activation=quantized_relu(9, 2), name="relu_3")(event_id)

        event_id = QDense(16, name="Dense_4_jetID", **self.common_args)(event_id)
        event_id = QActivation(activation=quantized_relu(9, 2), name="relu_4")(event_id)

        event_id = QDense(output_shape, name="Dense_5_jetID", **self.common_args)(event_id)

        outputs = Activation("sigmoid", name="event_id_output")(event_id)

        # Define the model
        self.event_model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
