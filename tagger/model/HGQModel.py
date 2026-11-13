import json
import os
import pickle
from schema import Schema, And, Use, Optional
import scipy

import keras
import numpy as np
import numpy.typing as npt
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten, Multiply, Concatenate
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler, PieceWiseSchedule,EarlyStoppingWithEbopsThres

from keras.models import load_model
#import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML, get_input_mask
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import initialise_tensorflow,initialise_jax,log_beta_schedule,cosine_decay_restarts


@JetModelFactory.register('HGQModel')
class HGQModel(JetTagModel):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "conv1d_parallelisation_factor" : list,
                                  "classification_layers" : list,
                                  "classification_parallelisation_factor" : list,
                                  "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                                  },

                "training_config" :     {"weight_method" : list,
                                         "validation_split" : And(float, lambda s: s > 0.0),
                                         "epochs" : And(int, lambda s: s >= 1),
                                         "batch_size" : And(int, lambda s: s >= 1),
                                         "learning_rate": And(float, lambda s: s > 0.0),
                                         "target_ebops" : int,
                                         "binary" : bool,
                                         "loss": str
                                        },

                "firmware_config" : {"input_precision" : str,
                                    "output_precision" : str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str},

                "inputs_config" : {"collections" : list,
                                   "objects": dict,
                                   "masks" : list,
                                   "event_features" : list,
                                   "combine" : bool}
            }
    )

    # Redefine save and load for HGQ due to needing h5 format
    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        #model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")
        pickle_path = os.path.join(out_dir, "model/saved_model.pkl")
        with open(pickle_path, "wb") as file:
            pickle.dump(self.jet_model, file)
        print(f"Model saved to {pickle_path}")


    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")

    def predict(self, X_test: npt.NDArray[np.float64]) -> tuple:
        model_outputs = self.jet_model.predict(X_test)
        if self.training_config["binary"]:
            class_predictions = scipy.special.expit(model_outputs[0])
        else:
            class_predictions = scipy.special.softmax(model_outputs[0],axis=1)
        return class_predictions

    def compile_model(self, num_samples: int, ebops: int):

        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        scheduler = keras.callbacks.LearningRateScheduler(schedule = lambda epoch : cosine_decay_restarts(epoch,
                                                                                                          initial_learning_rate=self.training_config['learning_rate'],
                                                                                                          max_epochs=self.training_config['epochs']))

        es = EarlyStoppingWithEbopsThres(monitor="val_loss",
                                         patience=150,
                                         verbose=1,
                                         mode="min",
                                         restore_best_weights=True,
                                         start_from_epoch=75,
                                         ebops_threshold=ebops + 100000
                                        )
        terminate_on_nan = keras.callbacks.TerminateOnNaN()

        ebops_tracker = FreeEBOPs()
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            scheduler,
            ebops_tracker,
            terminate_on_nan,
            es

        ]
        # compile the tensorflow model setting the loss and metrics
        is_binary = self.training_config["binary"]
        output_name = self.loss_name + self.output_id_name
        accuracy = "binary_accuracy" if is_binary else "categorical_accuracy"

        # compile the tensorflow model setting the loss and metrics
        use_loss = self._get_loss()
        is_binary = self.training_config["binary"]
        output_name = self.loss_name + self.output_id_name

        self.jet_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss={output_name: use_loss},
            metrics={output_name: accuracy},
            weighted_metrics={output_name: accuracy},
        )

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
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        keras.config.disable_traceback_filtering()

        # Train the model using hyperparameters in yaml config
        history = self.jet_model.fit(
            X_train,
            y_train,
            sample_weight = sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

        self.history = history.history

    def prepare_inputs(self, raw_dict):

        data_dict = {}
        model_dict = {}
        for f in self.inputs_config['collections']:
            data_dict[f"{f}_input"] = raw_dict[f]

        for m in self.inputs_config['masks']:
            model_dict[f'{m}_mask'] = get_input_mask(raw_dict[m])

        event_features = []
        for i in self.inputs_config['event_features']:
            event_features.append(raw_dict[i].reshape(raw_dict[i].shape[0], -1))
        event_features = np.concatenate(event_features, axis=1)

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

    def _get_loss(self):
        """Build the loss from the training config.

        Config keys (under training_config):
            loss (str): 'cce' (default) or 'grouped'
            lambda (float): weight of the grouped BCE term, only for 'grouped'
            loss_bkg_indices (list[int]): background class indices, only for 'grouped'
        """
        loss_type = self.training_config['loss']

        if loss_type == 'cce':
            return 'categorical_crossentropy'   # keep the Keras built-in for the baseline
        if loss_type == 'bce':
            return 'binary_crossentropy'
        if loss_type == 'grouped':
            return make_grouped_loss(
                bkg_idx=self.class_labels['MinBias'],
                lam=self.training_config['lambda'],
            )
        if loss_type == 'hardneg':
            return make_hardneg_loss(
                bkg_idx=self.class_labels['MinBias'],
                boost = 2.0,
                gamma = 2.0,
            )
        if loss_type == 'penalty_cce':
            return penalty_cce()
        if loss_type == 'fixed_cut':
            return make_fixed_cut_loss(
                threshold=self.training_config.get('fixed_cut_threshold', 0.8),
                margin=self.training_config.get('fixed_cut_margin', 0.05),
                a_sig=self.training_config.get('fixed_cut_signal_weight', 1.0),
                b_bkg=self.training_config.get('fixed_cut_background_weight', 5.0),
                bce_weight=self.training_config.get('fixed_cut_bce_weight', 1.0),
            )
        if loss_type == 'hard_negative_pairwise':
            return make_loss_and_metrics(
                threshold=self.training_config.get('pairwise_threshold', 0.8),
                margin=self.training_config.get('pairwise_margin', 0.05),
                pair_weight=self.training_config.get('pairwise_weight', 50.0),
                bce_weight=self.training_config.get('pairwise_bce_weight', 1.0),
                hardness_scale=self.training_config.get('pairwise_hardness_scale', 0.1),
            )
        if loss_type == 'focal':
            return make_focal_loss(
                gamma=2.,
                alpha=None,   # not 0. -- 0 means "weight every class by zero"
                penalty_matrix=[[0, 17, 17], [0, 0, 0], [0, 0, 0]],
            )

        raise ValueError(f"Unknown loss '{loss_type}', expected 'cce' or 'grouped'")

    def get_scores(self, preds, signal):
        """Get the score of the model for a given index

        Args:
            index (_type_): Index of the sample to get the score for

        Returns:
            _type_: Score of the model for the given index
        """
        if not self.training_config["binary"]:
            signal_index = self.class_labels[signal]
            minbias_index = self.class_labels["MinBias"]
            ovo_score = preds[:, signal_index] / (preds[:, signal_index] + preds[:, minbias_index])
            total_signal = 1 - preds[:, minbias_index]
            return [ovo_score, total_signal], ["ovo_score", "total_signal"]
        else:
            return [preds.flatten()], ["raw"]
