import datetime
import copy
import pydicom
from centaur_reports.helpers.helper_sr import CodedEntry, DataEntity, get_image_attrs, value_container, get_bbox_container
from centaur_reports.helpers.report_helpers import read_json
from centaur_reports import constants as CONST
#from centaur_engine.helpers.helper_results_processing import orient_coordinates
from centaur_engine.helpers.helper_category import CategoryHelper
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence



class SRWriter(object):


    def __init__(self, study_deploy_results, intended_workstation, algorithm_version, run_mode):
        """
        create Pydicom.Dataset that will represent the CAD SR report
        Args:
            study_deploy_results: StudyDeployResult object of a study successfully predicted by centaur
            intended_workstation: str that indicates which work station this CAD SR report will be viewed in
            algorithm_version: str, that indicates algorithm version
        """

        self.report_info = read_json(CONST.STRUCTURED_REPORT_CONFIG_PATH)
        self.report_info['intended_workstation'] = intended_workstation
        self.report_info['AlgorithmVersion'] = algorithm_version

        assert run_mode in [CONST.RUN_MODE_CADX, CONST.RUN_MODE_CADT], 'invalid run_mode {}'.format(run_mode)
        self.run_mode = run_mode
        self.dicom_results, self.overall_score, self.category, self.metadata = \
            self.parse_results(study_deploy_results)

        self.report_info['AlgorithmName'] = CONST.RUN_MODE_PRODUCT_MAP[run_mode]

        self.category_mappings = None

        self.sr = Dataset()
        self.sr.file_meta = self.file_meta()
        self.set_creation_datetime()

        if self.report_info['intended_workstation'] == 'eRad':
            self.apply_erad_modifications()


    def parse_results(self, study_deploy_results):
        """
        Parse the study_deploy_results for dicom_results, overall_score, category, metadata
        Args:
            study_deploy_results: StudyDeployResult for study that has been successfully predicted by centaur

        Returns:
            dicom_results: dictionary that maps DICOM SOPs to boxes
            overall_score: int that represent the overall study score
            category: total category of the study
            metadata: metadata of the study

        """
        metadata = study_deploy_results.metadata.copy()
        metadata = metadata.set_index('SOPInstanceUID', drop=False)

        model_results = study_deploy_results.results
        #orient_coordinates(model_results, metadata)

        dicom_results = copy.deepcopy(model_results['dicom_results'])
        dicom_results = {k: dicom_results[k]['none'] for k in dicom_results}

        study_results = copy.deepcopy(model_results['study_results'])
        overall_score = study_results['total']['score']
        category = study_results['total']['category']

        return dicom_results, overall_score, category, metadata

    def populate_sr(self):

        self.add_heading()  # creates root node for the SR and set additional attributes and specify a CAD SR template
        self.set_transfer_syntax()  # set transfer syntax EXPLICIT VR LITTLE ENDIAN
        self.set_creation_datetime()  # set the creation datatime to "now"
        self.add_study_info()  # add study information to the metadata of the SR
        self.add_evidence_container()  # add all the DICOM information that were referenced inside the report

        self.language_container = self.get_lang_container() # indicate language information for this report (English)
        self.image_library = self.get_image_library() # creates a list of images from on with get_finding_container will add annotation and other text
        self.findings_container = self.get_findings_container()  # box of text that has number of finding and Case Suspicion Level' asla add bounding boxes
        self.detection_summary = self.get_detection_summary()  # detection summary success? etc.
        self.analysis_summary = self.get_analysis_summary()  # state no analysis was attempted

        # adding the language container, image library, findings_container, detection_summary
        # and analysis_summary containers to the reports content sequence
        self.sr.ContentSequence = Sequence([self.language_container,
                                            self.image_library,
                                            self.findings_container,
                                            self.detection_summary,
                                            self.analysis_summary])

    def save(self, dir):
        """
        Save the CAD SR file
        Args:
            dir: str, path to save SR

        Returns: None

        """

        self.sr.save_as(dir, write_like_original=False)

    def get_uid(self):
        """
        generate a uid
        Returns: None

        """

        return pydicom.uid.generate_uid(self.report_info['dh_unique'])

    def file_meta(self):
        """
        set the file meta for the CAD SR
        Returns:

        """

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = DicomTypeMap.get_mammo_cad_sr_class_id()
        file_meta.MediaStorageInstanceUID = self.get_uid()
        file_meta.ImplementationClassUID = self.report_info['dh_unique'] + '1'
        file_meta.ImplementationVersionName = 'DeepHealth SR'
        file_meta.SourceApplicationEntityTitle = 'DeepHealth, Inc.'

        return file_meta

    def set_transfer_syntax(self):
        """
        Set the transfer syntax for this CAD File
        Returns: None

        """
        self.sr.is_little_endian = True
        self.sr.is_implicit_VR = False
        self.sr.file_meta.TransferSyntaxUID = CONST.EXPLICIT_VR_LITTLE_ENDIAN

    def set_creation_datetime(self):
        """
        set the report creation time
        Returns: None

        """

        self.report_info['ContentDate'] = datetime.datetime.now().strftime('%Y%m%d')
        self.report_info['ContentTime'] = datetime.datetime.now().strftime('%H%M%S.%f')

        self.sr.ContentDate = self.report_info['ContentDate']
        self.sr.ContentTime = self.report_info['ContentTime']

    def add_heading(self):
        """
        create document heading
        Returns: None

        """

        # create a new SOPInstanceUID and modality for this report. also sets the \
        # SOPClassUID to the one specifying Mammography CAD SR
        self.sr.SOPInstanceUID = self.get_uid()
        self.sr.SOPClassUID = DicomTypeMap.get_mammo_cad_sr_class_id()  # Mammography CAD SR
        self.sr.Modality = 'SR'

        # root node of the report, has type container and the continuity of content are separated
        self.sr.ConceptNameCodeSequence = CodedEntry(('111036', 'DCM', 'Mammography CAD Report')).seq
        self.sr.ValueType = 'CONTAINER'
        self.sr.ContinuityOfContent = 'SEPARATE'

        # set additional attributes.
        self.sr.Manufacturer = self.report_info['Manufacturer']
        self.sr.SeriesInstanceUID = self.get_uid()
        self.sr.SeriesNumber = '1'
        self.sr.InstanceNumber = '1'
        self.sr.ReferencedPerformedProcedureStepSequence = Sequence([])
        self.sr.PerformedProcedureCodeSequence = Sequence([])

        # getting and setting the template for CAD SR
        co = Dataset()
        co.MappingResource = 'DCMR'
        co.TemplateIdentifier = '4000'
        self.sr.ContentTemplateSequence = Sequence([co])

        self.sr.CompletionFlag = 'COMPLETE'
        self.sr.VerificationFlag = 'UNVERIFIED'

    def add_study_info(self):
        """
        adding study attributes to this CAD SR file
        Returns: None

        """

        STUDY_ATTRS = ['PatientName',
                       'PatientID',
                       'StudyInstanceUID',
                       'StudyDate',
                       'ReferringPhysicianName',
                       'StudyID',
                       'StudyTime',
                       'AccessionNumber',
                       'PatientBirthDate',
                       'PatientSex']

        # locate the metadata and set the attributes above
        assert len(self.metadata) > 0, "Missing metadata"
        study_row = self.metadata.iloc[0]
        for element in STUDY_ATTRS:
            setattr(self.sr, element, study_row[element])

    def add_evidence_container(self):
        """
        add a contains all the information DICOMs that are referenced in the CAD SR report
        Returns: None

        """
        # create a evidence container and sequence and set the StudyInstanceUID
        co = Dataset()
        seq = Sequence()
        co.StudyInstanceUID = self.metadata['StudyInstanceUID'].iloc[0]

        # for each Series Instance in the study that are used in the Centaur Model
        for series_instance_uid in self.metadata.SeriesInstanceUID.unique():

            # create Series specific container and sequence
            co2 = Dataset()
            seq2 = Sequence()

            # for each in DICOM in the Series
            for idx, row in self.metadata[self.metadata.SeriesInstanceUID == series_instance_uid].iterrows():
                # create a DICOM specific container
                co3 = Dataset()
                co3.ReferencedSOPClassUID = row['SOPClassUID']
                co3.ReferencedSOPInstanceUID = row['SOPInstanceUID']

                # add the DICOM container to the Series Sequence
                seq2.append(co3)
            # put the Series Sequence inside the Series Container and set the series_instance_uid for that container
            co2.ReferencedSOPSequence = seq2
            co2.SeriesInstanceUID = series_instance_uid
            # add the series container to the evidence container
            seq.append(co2)

        co.ReferencedSeriesSequence = seq

        self.sr.CurrentRequestedProcedureEvidenceSequence = Sequence([co])

        return co

    def get_lang_container(self):
        """
        create a container that specifies the language and language country for this report
        Returns: None

        """
        # make a language container
        co = DataEntity('HAS CONCEPT MOD', 'CODE')
        co.set_name_code(('121049', 'DCM', 'Language of Content Item and Descendants'))
        co.set_code((self.report_info['Language'][0], 'RFC3066', self.report_info['Language'][1]))

        # specify the language and language country
        seq = DataEntity('HAS CONCEPT MOD', 'CODE')
        seq.set_code((self.report_info['LanguageCountry'][0], 'ISO3166_1', self.report_info['LanguageCountry'][1]))
        seq.set_name_code(('121046', 'DCM', 'Country of Language'))

        co.set_content_seq(seq.val)

        return co.val

    def get_image_library(self):
        """
        get image related data.
        Returns: None

        """

        # image_library container
        co = DataEntity('CONTAINS', 'CONTAINER', 'SEPARATE')
        co.set_name_code(('111028', 'DCM', 'Image Library'))

        # list of containers that will be sequence inside the image_library container
        co_list = []

        # keep track of the index in the co_list
        idx = 1
        for row_num, row in self.metadata.iterrows():

            sop_instance_uid = row.SOPInstanceUID

            if len(self.dicom_results[sop_instance_uid]) == 0:
                # add to Image Library even if no boxes

                # make image container for this DICOM
                image_co = DataEntity('CONTAINS', 'IMAGE')
                sop_co = Dataset()
                sop_co.ReferencedSOPClassUID = row['SOPClassUID']
                sop_co.ReferencedSOPInstanceUID = sop_instance_uid
                image_co.co.ReferencedSOPSequence = Sequence([sop_co])

                # get and set image_attributes
                image_attrs = get_image_attrs(row, self.report_info)
                image_co.set_content_seq(image_attrs)
                co_list.append(image_co.val)
                idx += 1
            else:

                for box_idx in range(len(self.dicom_results[sop_instance_uid])):

                    # keep tract which image in the image library corresponding to this annotation
                    self.dicom_results[sop_instance_uid][box_idx]['library_idx'] = idx
                    image_co = DataEntity('CONTAINS', 'IMAGE')
                    sop_co = Dataset()
                    sop_co.ReferencedSOPClassUID = row['SOPClassUID']
                    sop_co.ReferencedSOPInstanceUID = sop_instance_uid

                    # add slice number to annotation if its a 3D DICOM
                    if DicomTypeMap.get_type_row(row) == DicomTypeMap.DBT:
                        sop_co.ReferencedFrameNumber = self.dicom_results[sop_instance_uid][box_idx]['slice']
                    image_co.co.ReferencedSOPSequence = Sequence([sop_co])
                    image_attrs = get_image_attrs(row, self.report_info)
                    image_co.set_content_seq(image_attrs)
                    co_list.append(image_co.val)

                    idx += 1

        co.set_content_seq(co_list)

        return co.val

    def apply_erad_modifications(self):
        """
        only keep bounding boxes for 3D DICOMS
        Returns:

        """
        dicom_results_copy = copy.deepcopy(self.dicom_results)
        for foruid in self.metadata.FrameOfReferenceUID.unique():
            foruid_rows = self.metadata[self.metadata["FrameOfReferenceUID"] == foruid]
            dbts = []
            for _, row in foruid_rows.iterrows():
                if DicomTypeMap.get_type_row(row) == DicomTypeMap.DBT:
                    dbts.append(row.SOPInstanceUID)

            # no dbt don't delete any boxes
            if dbts == []:
                continue

            assert len(dbts) == 1, "only expect one dbt per FORUID, however got these dbts {} for FORUID {}".format(
                dbts, foruid)

            dbt = dbts[0]
            for dicom in dicom_results_copy:
                if dicom not in foruid_rows.SOPInstanceUID.values:
                    continue
                if dicom != dbt:
                    dicom_results_copy[dicom] = []
        self.dicom_results = dicom_results_copy


    def get_findings_container(self):
        """
        Create a box of text for number o findings and case assessment
        """

        if self.run_mode ==  CONST.RUN_MODE_CADX and sum([len(self.dicom_results[a]) for a in self.dicom_results]) > 0:
            with_findings = True
        elif self.run_mode == CONST.RUN_MODE_CADT and self.category>=1:
            with_findings = True
        else:
            with_findings = False

        # initiate findings container
        co = DataEntity('CONTAINS', 'CODE')
        co.set_name_code(('111017', 'DCM', 'CAD Processing and Findings Summary'))
        if with_findings:
            co.set_code(('111242', 'DCM', 'All algorithms succeeded; with findings'))
            lesion_detected = 'Finding(s) detected'
        else:
            co.set_code(('111241', 'DCM', 'All algorithms succeeded; without findings'))
            lesion_detected = 'No finding(s) detected'

        # get a list of annotations
        annotations_list = []


        # add the lesion_detected text to the annotations_list
        annotations_list.append(
            value_container('TEXT', ('111033', 'DCM', 'Impression Description'), lesion_detected).val)

        if self.report_info['intended_workstation'] in ['ThreePalm', 'eRad']:

            # created a certainty container for case suspicious level
            certainty_co = DataEntity('HAS PROPERTIES', 'TEXT')
            certainty_co.set_name_code(('DH0401', 'DHCODE', 'Case Assessment Type'))
            if self.run_mode == CONST.RUN_MODE_CADX:
                certainty_co.val.TextValue = 'Case Suspicion Level'
            else:
                certainty_co.val.TextValue = ''

            annotations_list.append(certainty_co.val)

            certainty_co_val = DataEntity('HAS PROPERTIES', 'TEXT')
            certainty_co_val.set_name_code(('DH0402', 'DHCODE', 'Case Assessment Value'))
            certainty_co_val.val.TextValue = CategoryHelper.get_category_abbreviation(self.category, self.run_mode)

            annotations_list.append(certainty_co_val.val)
        else:
            raise ValueError('Config \'intended_workstation\': {} not supported.'
                             .format(self.report_info['intended_workstation']))


        annotations_list.append(
            value_container('TEXT', ('111001', 'DCM', 'Algorithm Name'), self.report_info['AlgorithmName']).val)
        annotations_list.append(
            value_container('TEXT', ('111003', 'DCM', 'Algorithm Version'), self.report_info['AlgorithmVersion']).val)

        # add annotation to each DICOM
        for uid in self.dicom_results:

            co1 = DataEntity('INFERRED FROM', 'CONTAINER', 'SEPARATE')
            co1.set_name_code(('111034', 'DCM', 'Individual Impression/Recommendation'))

            co1_list = []

            co2 = DataEntity('HAS CONCEPT MOD', 'CODE')
            co2.set_name_code(('111056', 'DCM', 'Rendering Intent'))
            co2.set_code(('111150', 'DCM', 'Presentation Required: Rendering device is expected to present'))
            co1_list.append(co2.val)

            box_list = self.dicom_results[uid]

            # get each annotation
            if self.run_mode == CONST.RUN_MODE_CADX:
                for box_info in box_list:

                    co3 = DataEntity('CONTAINS', 'CODE')
                    co3.set_name_code(('111059', 'DCM', 'Single Image Finding'))
                    co3.set_code(('F-01796', 'SRT', 'Mammography breast density'))

                    co3.set_content_seq(get_bbox_container(self.report_info, box_info,
                                                           CategoryHelper.get_category_abbreviations(self.run_mode)))

                    co1_list.append(co3.val)

            co1.set_content_seq(co1_list)

            annotations_list.append(co1.val)

        co.set_content_seq(annotations_list)

        return co.val

    def get_detection_summary(self):

        # summary container
        co = DataEntity('CONTAINS', 'CODE')
        co.set_name_code(('111064', 'DCM', 'Summary of Detections'))
        co.set_code(('111222', 'DCM', 'Succeeded'))

        # container for content continuity
        co1 = DataEntity('INFERRED FROM', 'CONTAINER', 'SEPARATE')
        co1.set_name_code(('111063', 'DCM', 'Successful Detections'))
        co.set_content_seq(co1.val)

        # detection performed container
        co2 = DataEntity('CONTAINS', 'CODE')
        co2.set_name_code(('111022', 'DCM', 'Detection Performed'))
        co2.set_code(('F-01796', 'SRT', 'Mammography breast density'))
        co2_list = []

        # algo name container and add to the detection performed container
        co3 = DataEntity('HAS PROPERTIES', 'TEXT')
        co3.set_name_code(('111001', 'DCM', 'Algorithm Name'))
        co3.co.TextValue = self.report_info['AlgorithmName']
        co2_list.append(co3.val)

        # algo version container and add to detection container.
        co4 = DataEntity('HAS PROPERTIES', 'TEXT')
        co4.set_name_code(('111003', 'DCM', 'AlgorithmVersion'))
        co4.co.TextValue = self.report_info['AlgorithmVersion']
        co2_list.append(co4.val)

        # for each dicom in the metadata, set the content identifier
        # maybe use to reference the items set by add_evidence_container
        for i in range(len(self.metadata)):

            co5 = DataEntity('HAS PROPERTIES')
            co5.co.ReferencedContentItemIdentifier = [1, 2, i + 1]
            co2_list.append(co5.val)

        co2.set_content_seq(co2_list)

        co1.set_content_seq(co2.val)

        return co.val

    def get_analysis_summary(self):

        co = DataEntity('CONTAINS', 'CODE')
        co.set_name_code(('111065', 'DCM', 'Summary of Analyses'))
        co.set_code(('111225', 'DCM', 'Not Attempted'))

        return co.val
