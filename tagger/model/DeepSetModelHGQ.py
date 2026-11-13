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
from tagger.model.HGQModel import HGQModel
from tagger.model.common import initialise_tensorflow,initialise_jax,log_beta_schedule,cosine_decay_restarts


@JetModelFactory.register('DeepSetModelHGQ')
class DeepSetModelHGQ(HGQModel):

    def build_model(self, input_shapes, output_shape):

        initialise_tensorflow(self.run_config['num_threads'])
        #initialise_jax()
        scope0 = QuantizerConfigScope(default_q_type='kbi',
                                      b0=7,
                                      overflow_mode='wrap',
                                      i0=0,
                                      fr=MonoL1(1.e-8),
                                      ir=MonoL1(1.e-8),
                                    )

        scope1 = QuantizerConfigScope(default_q_type='kif',
                                      place='datalane',
                                      overflow_mode='wrap',
                                      f0=7,
                                      fr=MonoL1(1.e-8),
                                      ic=MinMax(0, 12),
                                    )
        heterogeneous_axis = None

        scope2 = LayerConfigScope(enable_ebops=True, heterogeneous_axis=heterogeneous_axis, beta0=1e-8)

        with scope0, scope1, scope2:

                # Inputs
                inputs_jets = Input(shape=input_shapes["jets_input"], name="jets_input")
                inputs_event_features = Input(shape=input_shapes["event_features_input"], name="event_features_input")
                inputs_jets_mask = Input(shape=input_shapes["jets_mask"], name="jets_mask")

                # Batch Norm
                main = QBatchNormalization(name="norm_input")(inputs_jets)
                main_features = QBatchNormalization(name="norm_features")(inputs_event_features)

                for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
                    main = QConv1D(filters=depthconv1d, parallelization_factor=self.model_config['conv1d_parallelisation_factor'][iconv1d], kernel_size=1, name='Conv1D_' + str(iconv1d + 1),activation='relu')(main)
                main = Multiply(name='masking')([main, inputs_jets_mask])
                main = GlobalAveragePooling1D(name='avgpool')(main)
                main = Concatenate(name="concat")([main, main_features])

                for iclass, depthclass in enumerate(self.model_config['classification_layers']):
                    if iclass == 0:
                        event_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(main)
                    else:
                        event_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(event_id)
                event_id = QDense(output_shape, parallelization_factor=output_shape, activation='relu', name='event_id_output')(event_id)

                #Define the model using both branches
                self.jet_model = keras.Model(inputs = [inputs_jets, inputs_event_features, inputs_jets_mask], outputs = event_id)

                print(self.jet_model.summary())

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
            config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
            config["Model"]["Strategy"]="distributed_arithmetic"
            config["Model"]["ReuseFactor"]=1
            config['IOType'] = 'io_parallel'

            # Configuration for conv1d layers
            # hls4ml automatically figures out the paralellization factor
            config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
            config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16

            #config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']
            #config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
            #config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']

            # Additional config

            # Write HLS
            self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
                self.jet_model,
                backend='Vitis',
                project_name=self.firmware_config['project_name'],
                clock_period=self.firmware_config['clock_period'],
                hls_config=config,
                output_dir=f'{hls4ml_outdir}',
                part= self.firmware_config['fpga_part'],
                # namespace='hls4ml_'+self.firmware_config['project_name'],
                # write_weights_txt=False,
                # write_emulation_constants=True,
            )

            # Compile the project
            self.hls_jet_model.compile()

            # Save config  as json file
            print("Saving default config as config.json ...")
            with open(hls4ml_outdir + '/config.json', 'w') as fp:
                json.dump(config, fp)

            if build:
                # build the project
                self.hls_jet_model.build(csim=False, reset=True)
