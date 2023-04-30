#import time
import os
import numpy as np
import centaur_engine.constants as const
import centaur_engine.models.model as model
import centaur_engine.models.nn_models as nn_models
import centaur_engine.models.model_selector as model_selector
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from deephealth_utils.misc.config import load_config


class AggModel(model.BaseModel):
    """AggModel class: Abstract class for Aggregate models.
    AggModels aggregate NNModels, but do not contain model weights.
    Methods: __call__(inputs): Take inputs and returns outputs.
    """

    def __init__(self, config, version='current', required_keys=[], logger=None):
        super().__init__(config, logger=logger)

        # load model configs
        self._model_configs = {}
        for k, v in config['models'].items():
            if 'config_name' in v:  # loading a model, open the _config file
                self._model_configs[k] = load_config(os.path.join(const.MODEL_PATH, version, config['model_type'],
                                                                  v['config_name']))
            else:  # loading an agg model, just pass the dictionary
                self._model_configs[k] = v

        self._check_required_keys(required_keys)

        # load models
        self._models = None
        self._load_models(version=version)

    def __call__(self, inputs):
        raise NotImplementedError('The __call__() method is not implemented in the BaseModel child object.')

    def _load_models(self, version='current'):
        """
        Create all the models that compound this Aggregate model based on their config files.
        Save the value in self._models property
        """
        self._models = {}
        for key, config in self._model_configs.items():
            config['model_type'] = key
            self._models[key] = model_selector.ModelSelector.get_class(config['class'])(config, version=version, logger=self.logger)

    def _check_required_keys(self, keys):
        for key in keys:
            if key not in self._model_configs:
                raise NotImplementedError('The required key {} is not in the _config file.'.format(key))

    @property
    def models(self):
        return self._models

    def get_version(self):
        """
        Get the model version.
        If the model has a version directly assigned, return that version.
        Otherwise, return a concatenation of the model versions that compound the model in a "M1__M2__...__Mn" format
        :return: str
        """
        # Try to get the version directly from the config file
        v = super().get_version()
        if v is not None:
            return v
        # If not found, look recursively
        return "__".join(map(lambda m: m.get_version(), self.models.values()))

    def get_params_config(self):
        raise NotImplementedError()

class TupleDxmAggModel(AggModel):

    def __init__(self, config, version='current', logger=None):
        self.required_keys = ['tuple']
        super().__init__(config, version=version, required_keys=self.required_keys, logger=logger)
        self.model_config = self._models['tuple'].get_config()

    def __call__(self, inputs):
        """
        Return tuple that contains model_results + proc_indo
        :param inputs:
        :return:
        """
        results = self._models['tuple'](inputs)
        return results

    def get_input_proc_config(self):
        return self._models['tuple'].input_proc_config

    def get_params_config(self):
        return self._models['tuple'].params_config


class OptimizedSyntheticAggModel(AggModel):
    """BtSyntheticRetinaNetModel class: 2D Synthetic model for BT DICOM images.
        Runs a RetinaNet model for each 3D slice and constructs a 2D synthetic image, which is then scored by a second
        RetinaNet model.
        """

    def __init__(self, config, version='current', logger=None):
        self.required_keys = ['slice', 'scoring']
        super().__init__(config, version=version, required_keys=self.required_keys, logger=logger)
        self._models['synth'] = nn_models.BtSynthesizerModel(self._model_configs['slice'],
                                                             slice_model=self._models['slice'],
                                                             logger=logger)

    def __call__(self, inputs):
        results = self._models['synth'](inputs)
        # results['dicom_results'] dataframe contains a row for every synthetic 2D image produced from a BT image.
        results = self._models['scoring'](results['dicom_results'])
        return results

    def get_config(self):
        return self._models['scoring'].get_config()

    def get_input_proc_config(self):
        return self._models['scoring'].get_input_proc_config()

    def get_params_config(self):
        return self.get_config()['params']

class CombinedDxmDbtAggModel(AggModel):

    def __init__(self, config, version='current', logger=None):
        self.required_keys = ['dxm', 'dbt']
        super().__init__(config, version=version, required_keys=self.required_keys, logger=logger)

        # self._config['parallelize'] = True

    def __call__(self, df):
        """
        Return a dictionary with all the results:
            - dicom_results: one entry per dicom image with the bounding boxes
            - proc_info: one entry per dicom image with information about the image processing/info
            - study_results: synthesized results for the full study
        :param df: dataframe. Image info
        :return: dictionary.
        """
        # start_prediction = time.time()
        # if self._config['parallelize'] or True:
        #     import deephealth_utils.misc.parallel_helpers as dh_parallel
        #     from functools import partial
        #     self.queue = dh_parallel.ParallelQueue()
        results = {}

        for dicom_type in self.required_keys:
            dicom_df = DicomTypeMap.get_type_df(df, dicom_type)
            # dicom_df = df[df['SOPClassUID'] == uid]

            if len(dicom_df) > 0:
                results_ = self._models[dicom_type](dicom_df)
                results[dicom_type] = results_
        all_results = self.synthesize(results, df)
        return all_results

    def synthesize(self, results, df):
        """
        generates study results from dicom results

        iterate through each SOPUID
        get max score across all bboxes in each image-transform
        get mean score across all transforms in each image
        get mean score across all images in each laterality
        get max score across all lateralities in study

        :param results:
        :param df:
        :return: dictionary with the entries dicom_results, proc_info, study_results
        """

        study_results = {}
        bboxes = {}
        lats = {}
        proc_info_results_all = {}
        dicom_results_all = {}
        # dicom_results = results[0]
        # proc_info_results = results[1]
        for dicom_type, results_type in results.items():
            all_bboxes = []
            lat_scores = {'L': [], 'R': []}
            bboxes[dicom_type] = {}
            dicom_results = results_type['dicom_results']
            proc_info = results_type['proc_info']
            dicom_results_all.update(dicom_results)
            proc_info_results_all.update(proc_info)
            for sop_uid, result in dicom_results.items():  # for each image
                assert len(df[df['SOPInstanceUID'] == sop_uid]['ImageLaterality'].values) == 1, \
                    "Found more than one SOPInstanceUID {} in df".format(sop_uid)
                lat = df[df['SOPInstanceUID'] == sop_uid]['ImageLaterality'].values[0]
                transform_scores = []
                for transform, boxes in result.items():  # for each transform
                    if len(boxes) == 0:
                        raise ValueError("No bboxes found for image.")
                    if transform == "none":
                        all_bboxes.extend(boxes)
                    box_scores = []
                    for box in boxes:  # for each box in each image-transform
                        box_scores.append(box['score'])
                    transform_scores.append(np.max(box_scores))
                if len(transform_scores) > 0:
                    lat_scores[lat].append(np.mean(transform_scores))
                bboxes[dicom_type][lat] = all_bboxes
            lats[dicom_type] = lat_scores

        # is_dbt = len(df[df['SOPClassUID'] == DicomTypeMap.get_dbt_class_id()]) > 0

        has_dbt = 'dbt' in list(results.keys())

        if has_dbt:
            # case_thresholds = self.config['case_thresholds']['dbt']
            bboxes = bboxes['dbt']
        else:
            # case_thresholds = self.config['case_thresholds']['dxm']
            bboxes = bboxes['dxm']

        available_lats = list(bboxes.keys())
        lat_scores = []
        for lat in available_lats:
            # lat_class_scores contains [2D,3D] scores for a given laterality
            lat_class_scores = []
            for dicom_type, scores in lats.items():
                if len(scores[lat]) > 0:
                    score = np.mean(scores[lat])
                else:
                    raise ValueError("Laterality does not contain scores.")
                lat_class_scores.append(score)
            lat_score = np.mean(lat_class_scores)
            lat_scores.append(lat_score)
            study_results[lat] = {
                'score': lat_score,
                #'category': get_fourway_category(lat_score, case_thresholds)
            }

        study_score = np.max(lat_scores)
        study_results['total'] = {
            'score': study_score,
            #'category': get_fourway_category(study_score, case_thresholds)
        }

        return_results = {'dicom_results': dicom_results_all,
                          'proc_info': proc_info_results_all,
                          'study_results': study_results}

        return return_results

    def get_input_proc_config(self):
        return self._models['dxm'].get_input_proc_config()

    def get_params_config(self):
        return self._models['dxm'].get_params_config()