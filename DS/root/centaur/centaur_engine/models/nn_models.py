import time
import re
import scipy.stats
import os
import numpy as np
import copy
import centaur_engine.models.model as model
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from deephealth_utils.data.input_processing import ImageInputProcessor
from deephealth_utils.ml.detection_helpers import box_coords_to_original
from centaur_engine.helpers.helper_preprocessor import make_tmp_dir
from centaur_engine.helpers.helper_model import transform_input, get_crop_info
import centaur_engine.constants as const
from deephealth_utils.ml.dbt_synthetic_helpers import DetectionNmsSynthModel
from deephealth_utils.data.parse_logs import log_line
from keras_retinanet import models as kr_models


class NNModel(model.BaseModel):
    """NNModel class: Abstract class for Neural Network models.
    __init__() loads the configs for a neural network input.
    Methods: __call__(inputs): Take inputs and returns outputs.
    """

    def __init__(self, config, version='current', logger=None):
        super().__init__(config, logger=logger)
        self.params_config = config['params']
        if 'filter_params' in self.params_config:
            self.filter_params = self.params_config['filter_params']
        self.transforms_config = config['transforms']
        self.input_proc_config = config['input_proc']
        self.input_proc = ImageInputProcessor(**self.input_proc_config).create_input
        self.weight_path = const.MODEL_PATH + version + '/' + self.config['weight_path']
        self.model = None
        self._weights_loaded = False

    def __call__(self, inputs):
        self._weights_check()
        return self._predict(inputs)

    def _weights_check(self):
        """
        Load model weights if they have not been already loaded. Otherwise, do nothing
        """
        if not self._weights_loaded:
            self.logger.info(log_line(-1, "Loading model weights ({})...".format(type(self).__name__)))
            self.load_model()
            self._weights_loaded = True

    def _predict(self, df):
        raise NotImplementedError("This method must be implemented in a child class")

    def load_model(self):
        raise NotImplementedError("This method must be implemented in a child class")


class RetinaNetModel(NNModel):
    """RetinaNet class: Abstract class for RetinaNet models.
    load_model() loads the standard RetinaNet Keras model.
    """

    def __init__(self, config, version='current', logger=None):
        super().__init__(config, version=version, logger=logger)
        self.backbone_name = self.params_config['backbone_name']

    def load_model(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
        t1 = time.time()
        if self.params_config['convert']:
            self.model = kr_models.load_model(self.weight_path,
                                              convert=self.params_config['convert'],
                                              backbone_name=self.backbone_name,
                                              nms=self.filter_params['nms'],
                                              score_threshold=self.filter_params['score_threshold'],
                                              max_detections=self.filter_params['max_detections'],
                                              nms_threshold=self.filter_params['nms_threshold'],
                                              return_boxes=self.params_config['return_boxes'])
        else:
            assert 'filter_params' not in self.params_config, "'filter_params' should not be in self.params_config"
            self.model = kr_models.load_model(self.weight_path,
                                              convert=self.params_config['convert'],
                                              backbone_name=self.backbone_name)
        output_dim = self.model.layers[-1]._inbound_nodes[0].input_tensors[1]._keras_shape[2]
        assert output_dim == 2, \
            "Model classification output dimension is {} but should be 2".format(output_dim)
        self.log_time("Load_Model", time.time() - t1, additional_info=self.weight_path)


class DxmRetinaNetModel(RetinaNetModel):
    """DxmRetinaNetModel class: DXm RetinaNet model.
    Scores DXm DICOM files with a RetinaNet model.
    """

    def predict(self, X):
        """
        Simple wrapper for _predict() since the det_model in dh_util's DetectionNmsSynthModel calls predict()
        :param X:x
        :return:
        """
        self._weights_check()
        return self._predict(X)

    def _predict(self, df, repeat_input=False):
        """predict() takes in a df and produces score and bboxes for each DICOM file (SOPInstanceUID).
        It also produces preprocessing info for each image

        :param df: a Pandas dataframe containing rows of SOPInstanceUIDs
        :return: dictionary:
                - 'dicom_scores': scores and bboxes for each SOPInstanceUID
                - 'proc_info': image preprocessing info
        """
        #predict_start = time.time()
        df = df.copy()
        results = {}
        proc_info = {}

        # For L and R lateralities
        for index, row in df.iterrows():
            sop_uid = row['SOPInstanceUID']
            if self._pixel_data is not None:
                if 'synth' in self._pixel_data[index].keys():
                    X = self._pixel_data[index]['synth']
                else:
                    X = self._pixel_data[index][0]
            else:
                pattern = r'^frame_.+\.npy$'
                np_paths = [p for p in os.listdir(row['np_paths']) if re.search(pattern, p)]
                assert len(np_paths) == 1, 'Expected exactly one numpy array with name frame_*.npy, found the \
                                following instead: {}'.format(np_paths)
                np_path = os.path.join(row['np_paths'], np_paths[0])

                # Load the numpy-ized dicom
                X = np.load(np_path)

            assert int(row['Rows']) == X.shape[0] and int(row['Columns']) == X.shape[1], "Image shape does not match"
            t1 = time.time()
            results[sop_uid], proc_info_ = self._predict_transform(X, repeat_input=repeat_input)
            self.log_time("Image_DxmRetinaNetModel_Full_Image_Prediction", time.time() - t1,
                          "{};{}".format(row['StudyInstanceUID'], sop_uid))
            proc_info_['model_class'] = self.__class__.__name__
            proc_info[sop_uid] = proc_info_
        return {'dicom_results': results, 'proc_info': proc_info}

    def _predict_transform(self, X, repeat_input=False):
        # Evaluate the DICOM file for each transformation
        results = {}
        proc_info = {}
        for transform_type in self.transforms_config:
            transformed_input = transform_input(X, transform_type)
            X_, proc_funcs = self.input_proc(transformed_input, return_extra_info=True)
            # get only the trim info

            # for the un-transformed one, store dimension info
            if transform_type == "none":
                proc_info = {'proc_funcs': proc_funcs[0],
                             'original_shape': X.shape,
                             'transformed_shape': np.shape(X_[0][0, :, :, 0]),
                             'crop_info': get_crop_info(proc_funcs)}

            # TEMPORARY CODE FOR SUPPORTING TupleDxmRetinaNetModel
            if repeat_input:
                X_input = [X_[0]] * len(self.model.inputs)
            else:
                X_input = X_
            t1 = time.time()
            preds = self.model.predict(X_input)
            self.log_time("Image_DxmRetinaNetModel_Keras_Prediction", time.time() - t1,
                          additional_info="{} x {}".format(len(X_input), X_input[0].shape))
            result = self._parse_preds(preds, proc_info)

            results[transform_type] = result

        return results, proc_info

    def _parse_preds(self, preds, proc_info):
        # Thresholds the predictions
        pos_labels = preds[2][0] == 1  # assumes positive label is 1
        if sum(pos_labels) > 0:  # has positive (malignant) label
            for i in range(len(preds)):
                preds[i] = preds[i][0, pos_labels]
            # im_score = np.max(preds[1])
            det_idx = preds[1] > 0.0  # self.params_config['map_score_threshold']
            if np.sum(det_idx):
                box_coords = preds[0][det_idx]
                box_scores = preds[1][det_idx]
                if self.params_config['convert']:
                    box_coords = box_coords[:self.filter_params['max_detections']]
                    box_scores = box_scores[:self.filter_params['max_detections']]
            else:
                box_scores = []
                box_coords = []
        else:
            box_scores = []
            box_coords = []
        box_coords = box_coords_to_original(box_coords, proc_info)
        result = []
        for box in range(len(box_scores)):
            result.append({
                'coords': box_coords[box].tolist(),
                'score': box_scores[box].tolist(),
                #'category': get_fourway_category(box_scores[box], self.params_config['bbx_thresholds'])
            })
        return result


class TupleDxmRetinaNetModel(DxmRetinaNetModel):
    """TupleDxmRetinaNetModel class: Tuple DXm RetinaNet model.
    Scores DXm DICOM files with 1-N RetinaNet models.
    The only difference from DxmRetinaNetModel is the load_model() and _predict() methods.
    """

    def _predict(self, df):
        return super()._predict(df, repeat_input=True)


class SliceDxmRetinaNetModel(DxmRetinaNetModel):
    """SliceDxmRetinaNetModel class: Slice DXm RetinaNet model.
    Scores 2D slices from a BT image.
    The only difference from DxmRetinaNetModel is the load_model() method.
    """

    def _predict(self, X):
        """
        :param X:
        :return pred:
        """
        t1 = time.time()
        pred = self.model.predict(X)
        self.log_time("Image_SliceDxmRetinaNetModel_Keras_Prediction", time.time() - t1,
                      additional_info="{} x {}".format(len(X), X[0].shape))
        return pred


class ScoringDxmRetinaNetModel(TupleDxmRetinaNetModel):
    """SynthScoringDxmRetinaNetModel class: Synthetic Scoring DXm RetinaNet model.
    Scores the synthetic 2D images.
    The only difference from TupleDxmRetinaNetModel is the _predict() method.
    """

    def _predict(self, df):
        preds = super()._predict(df)
        for sop_uid in preds['dicom_results'].keys():
            sop_df = df[df['SOPInstanceUID'] == sop_uid]
            assert len(sop_df) == 1, 'Multiple rows found in the input dataframe for SOPInstanceUID {}'.format(sop_uid)
            if self._pixel_data is not None:
                slice_map = self._pixel_data[sop_df.index.values[0]]['slice_map']
            else:
                slice_map = np.load(os.path.join(sop_df['np_paths'].values[0], const.SYNTHETIC_SLICE_MAP_FILENAME),
                                    allow_pickle=True)
            for box_dict in preds['dicom_results'][sop_uid]['none']:
                slice_num = self.get_box_slice_num(slice_map, box_dict)
                box_dict['slice'] = int(slice_num)
        return preds

    def get_box_slice_num(self, slice_map, box_dict):
        x1, y1, x2, y2 = [int(coord) for coord in box_dict['coords']]
        return int(scipy.stats.mode(slice_map[y1:y2, x1:x2], axis=None)[0][0])


class BtSynthesizerModel(NNModel):
    """BtSynthesizerAggModel takes in a df.
    It takes the BT images and and runs each slice of a BT file through a DxmRetinaNetModel and synthesizes
    the outputs into a synthetic 2d image.
    Returns a dataframe containing a row for each synthetic 2d image.
    """

    def __init__(self, config, version='current', slice_model=None, logger=None):
        super().__init__(config, version=version, logger=logger)
        self.slice_model = slice_model
        filter_params = copy.deepcopy(self.filter_params)
        self.nms_synthesizer = DetectionNmsSynthModel(self.input_proc, self.slice_model,
                                                      self.params_config['slice_start_prop'],
                                                      self.params_config['slice_stride'],
                                                      filter_params)

    def __call__(self, inputs):
        # start_prediction = time.time()
        bt_df = inputs
        syn_df = bt_df.copy()

        all_proc_info = {}
        for index, row in bt_df.iterrows():
            t1 = time.time()
            if self._pixel_data is not None:
                bt_im = np.stack([v for k, v in self._pixel_data[index].items()], axis=0)
                tmp_dir = make_tmp_dir(const.TMP_NP_PATH)
                tmp_dir = os.path.join(tmp_dir, 'synth')
            else:
                np_paths = os.listdir(row['np_paths'])
                tmp_dir = os.path.join(os.path.dirname(row['np_paths']), 'synth')
                np_paths = sorted(np_paths, key=lambda s: int(s.split('frame_')[1].split('.npy')[0]))
                np_files = [os.path.join(row['np_paths'], np_path) for np_path in np_paths]
                bt_im = np.stack([np.load(np_file) for np_file in np_files], axis=0)

            # bt_synthetic is a 2-channel image - the zeroth channel is the synthetic image, and the first channel is
            # a pixel map that shows which slice each pixel in the synthetic image comes from
            assert not os.path.isdir(tmp_dir), 'Synth directory {} already exists'.format(tmp_dir)
            t1 = time.time()

            bt_synthetic, synthetic_tracking_im, proc_info = self.nms_synthesizer.create_tracked_center_synthetic \
                                                             (bt_im, return_all=False)

            self.log_time("Image_BtSynthesizerModel_Synthetic_Creation", time.time()-t1,
                          "{};{}".format(row['SOPInstanceUID'], row['StudyInstanceUID']))
            os.makedirs(tmp_dir)
            synthetic_fn = os.path.join(tmp_dir, const.SYNTHETIC_IM_FILENAME)
            synthetic_slice_map_fn = os.path.join(tmp_dir, const.SYNTHETIC_SLICE_MAP_FILENAME)

            np.save(synthetic_fn, bt_synthetic)
            np.save(synthetic_slice_map_fn, synthetic_tracking_im)

            if self._pixel_data is not None:
                self._pixel_data[index]['synth'] = bt_synthetic
                self._pixel_data[index]['slice_map'] = synthetic_tracking_im

            syn_row = syn_df.loc[index].copy()
            syn_row['SOPClassUID'] = DicomTypeMap.get_dxm_class_id()
            syn_row['pixel_array_shape'] = bt_synthetic.shape
            syn_row['np_paths'] = tmp_dir
            syn_df.loc[index] = syn_row
            all_proc_info[index] = proc_info
        return {'dicom_results': syn_df, 'proc_info': all_proc_info}
