"""QKeras model parent class

Written 29/09/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from schema import Schema, And, Use, Optional
from tagger.data.tools import get_input_mask

# Qkeras
from qkeras.quantizers import quantized_bits
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tagger.model.common import choose_aggregator
from tagger.model.JetTagModel import JetModelFactory, JetTagModel

class QKerasModel(JetTagModel):
    """QKerasModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    quantization_schema = {'quantizer_bits' : And(int, lambda s: 32 >= s >= 0),
                           'quantizer_bits_int' : And(int, lambda s: 32 >= s >= 0),
                           'quantizer_alpha_val' : And(float, lambda s: 1.0 >= s >= 0.0)
                           }

    training_config_schema =    {"weight_method" : list,
                                 "validation_split" : And(float, lambda s: s > 0.0),
                                 "epochs" : And(int, lambda s: s >= 1),
                                 "batch_size" : And(int, lambda s: s >= 1),
                                 "learning_rate" : And(float, lambda s: s > 0.0),
                                 "initial_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "final_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                 "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "ReduceLROnPlateau_patience" : int,
                                 "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0),
                                 "binary": bool}

    inputs_config_schema = {"collections" : dict,
                            "masks" : list,
                            "event_features" : list,
                            "combine" : bool}

    def _prune_model(self, num_samples: int):
        """Pruning setup for the model, internal model function called by compile

        Args:
            num_samples (int): number of samples in the training set used for scheduling
        """

        print("Begin pruning the model...")

        # Calculate the ending step for pruning
        end_step = (
            np.ceil(num_samples / self.training_config['batch_size']).astype(np.int32) * self.training_config['epochs']
        )

        # Define the pruned model
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.training_config['initial_sparsity'],
                final_sparsity=self.training_config['final_sparsity'],
                begin_step=0,
                end_step=end_step,
            )
        }
        self.event_model = tfmot.sparsity.keras.prune_low_magnitude(self.event_model, **pruning_params)

        # Add preface to loss name
        self.loss_name = 'prune_low_magnitude_'

        # Add pruning callback
        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]

        # Define the pruning
        if 'initial_sparsity' in self.training_config:
            self._prune_model(num_samples)

        # compile the tensorflow model setting the loss and metrics
        is_binary = self.training_config["binary"]
        self.event_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'binary_crossentropy' if is_binary else 'categorical_crossentropy',
            },
            metrics={
                self.loss_name + self.output_id_name: 'binary_accuracy' if is_binary else 'categorical_accuracy',
            },
            weighted_metrics={
                self.loss_name + self.output_id_name: 'binary_accuracy' if is_binary else 'categorical_accuracy',
            },
        )

    def prepare_inputs(self, raw_dict):

        data_dict = {}
        model_dict = {}
        for f in self.inputs_config['collections']:
            data_dict[f"{f}_input"] = raw_dict[f]

        for m in self.inputs_config['masks']:
            model_dict[f'{m}_mask'] = get_input_mask(data_dict[f"{m}_input"], 10)

        event_features = []
        for i in self.inputs_config['event_features']:
            event_features.append(raw_dict[i])
        event_features = np.stack(event_features, axis=1)

        data_dict['event_features_input'] = event_features

        # combine all into a single input if specified in the config
        if self.inputs_config['combine']:
            flattend_inps = [arr.reshape(arr.shape[0], -1) for arr in data_dict.values()]
            model_dict['model_input'] = np.concatenate(flattend_inps, axis=-1)
        else:
            model_dict.update(data_dict)

        input_shapes = {}
        for k, v in model_dict.items():
            input_shapes[k] = v.shape[1:]

        return model_dict, input_shapes

    def fit(
        self,
        X_train: dict,
        y_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """

        # Train the model using hyperparameters in yaml config
        self.history = self.event_model.fit(
            X_train,
            {self.loss_name + self.output_id_name: y_train},
            sample_weight=sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.event_model)

        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """

        # Additional custom objects for attention layers
        custom_objects_ = {
        }

        # Load the model
        self.event_model = load_qmodel(f"{out_dir}/model/saved_model.keras", custom_objects=custom_objects_)
