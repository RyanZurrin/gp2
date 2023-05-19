import centaur_engine.constants as const
from deephealth_utils.misc.config import load_config
from centaur_engine.models.agg_models import OptimizedSyntheticAggModel, CombinedDxmDbtAggModel, TupleDxmAggModel
from centaur_engine.models.nn_models import DxmRetinaNetModel, TupleDxmRetinaNetModel, SliceDxmRetinaNetModel, \
    ScoringDxmRetinaNetModel, BtSynthesizerModel
from centaur_deploy.deploys.config import Config


class ModelSelector(object):
    """
    ModelSelector only selects top-level models; intermediate models will raise a NotImplementedError.
    """
    model_dict = {
        'DxmRetinaNetModel': DxmRetinaNetModel,
        'TupleDxmRetinaNetModel': TupleDxmRetinaNetModel,
        'SliceDxmRetinaNetModel': SliceDxmRetinaNetModel,
        'ScoringDxmRetinaNetModel': ScoringDxmRetinaNetModel,
        'BtSynthesizerModel': BtSynthesizerModel,
        'TupleDxmAggModel': TupleDxmAggModel,
        'OptimizedSyntheticAggModel': OptimizedSyntheticAggModel,
        'CombinedDxmDbtAggModel': CombinedDxmDbtAggModel,
    }

    @staticmethod
    def get_class(model_str):
        if model_str in ModelSelector.model_dict:
            return ModelSelector.model_dict[model_str]
        else:
            raise NotImplementedError("Model class name is not implemented")

    @staticmethod
    def select(model_version, cadt_threshold_version, cadx_threshold_version,  logger=None):
        config = load_config(const.MODEL_PATH + str(model_version) + '/' + const.MODEL_CONFIG_JSON)

        if cadt_threshold_version is not None:
            cadt_threshold_config = load_config(const.CADT_THRESHOLD_PATH + '/' +"{}.json".format(cadt_threshold_version))
            config["cadt_thresholds"] = cadt_threshold_config
            for modality in config['cadt_thresholds']:
                assert all(
                    [x in config["cadt_thresholds"][modality].keys() for x in Config.CADT_OPERATING_POINT_KEYS]), \
                    "expected operating points are {} but got {} for cadt threshold versions for ".format(
                        config["cadt_thresholds"][modality].keys(), Config.CADT_OPERATING_POINT_KEYS)
        else:
            config["cadt_thresholds"] = None

        if cadx_threshold_version is not None:
            cadx_threshold_config = load_config(const.CADX_THRESHOLD_PATH + '/' +"{}.json".format(cadx_threshold_version))
            config["cadx_thresholds"] = cadx_threshold_config
        else:
            config["cadx_thresholds"] = None

        model_class = config['class']
        if model_class in ModelSelector.model_dict:
            model = ModelSelector.model_dict[model_class](config, version=model_version, logger=logger)
            return model
        else:
            raise NotImplementedError("Model class name is not implemented.")
