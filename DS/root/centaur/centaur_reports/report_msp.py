import os
import numpy as np
from centaur_reports.report import Report
from centaur_reports.helpers.sc_writer import SCWriter
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
import centaur_reports.constants as const

class MSPReport(Report):
    def __init__(self, algorithm_version, logger=None):
        super().__init__(algorithm_version, logger=logger)

        self._report_prefix = 'MSP'
        self._product = const.SAIGE_DX
        self._file_extension = 'dcm'

        self.sc_writer = SCWriter(algorithm_version)

    def generate(self, study_deploy_results, ram_images=None, **kwargs):
        """
        Generate an SC Dataset for each MSP image generated.
        """
        metadata = study_deploy_results.metadata.copy()
        synth_items = []
        if ram_images is not None:
            for key, value in ram_images.items():
                if 'synth' in value:
                    synth_items.append({'metadata': metadata.loc[key], 'synth_im': value['synth']})
        else:
            bt_metadata = metadata[metadata.apply(DicomTypeMap.get_type_row, axis=1) == DicomTypeMap.DBT].copy()
            bt_metadata['synth_path'] = bt_metadata['np_paths'].apply(lambda p: os.path.join(os.path.dirname(p), 'synth',
                                                                                             'frame_synth.npy'))
            for idx, row in bt_metadata.iterrows():
                assert os.path.isfile(row['synth_path']), f'Unable to find synthetic for {row["SOPInstanceUID"]} in ' \
                                                          f'{row["synth_path"]}'
                synth_items.append({'metadata': row, 'synth_im': np.load(row['synth_path'])})

        for item in synth_items:
            self.sc_writer.create_sc_dataset(item['metadata'], item['synth_im'])
            msp_name = self.get_output_file_name(self.sc_writer.sc.SOPInstanceUID)
            msp_path = os.path.join(study_deploy_results.output_dir, msp_name)
            self.sc_writer.save(msp_path)
