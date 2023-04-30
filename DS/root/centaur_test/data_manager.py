import hashlib
import os
import os.path as osp
import pandas as pd
import centaur_deploy.constants as const_deploy
import centaur_test.constants as const

class DataManager(object):
    # Studies Constants
    STUDY_01_HOLOGIC = "study_01_hologic"  # Hologic DXm Cancer
    STUDY_02_GE = "study_02_GE"  # GE Dxm No-Cancer
    STUDY_03_DBT_HOLOGIC = "study_03_dbt_hologic"  # Hologic DBT Cancer
    STUDY_04_HOLOGIC = "study_04_hologic" # Hologic Dxm No-Cancer
    STUDY_05_HOLOGIC_COMBO_MODE = "study_05_hologic_combo_mode"  # Hologic Combo mode (FFDM + DBT available)

    def __init__(self, dataset_name="DS_01", test_data_local_mode=True,
                 data_version=None, baseline_version=None, baseline_params=None):
        """
        Constructor
        Args:
            dataset_name (str): Name of the dataset (ex: DS_01)
            test_data_local_mode (bool): If True, all the testing data is expected to be in a single fixed local folder
                                        (/root/test_datasets/data and /root/test_datasets/baseline).
            data_version (str): data version
            baseline_version (str): baseline version
            baseline_params (str-dict): dictionary of additional params for the baseline path
        """

        self.dataset_name = dataset_name

        self._local_testing_mode = test_data_local_mode
        self.__filelist_df = None

        if test_data_local_mode:
            # All the data are expected to be in a local folder in centaur
            self._data_dir = osp.join(const_deploy.ROOT_PATH, const.CENTAUR_TEST_DATA_FOLDER)
            self._baseline_dir = osp.join(const_deploy.ROOT_PATH, const.CENTAUR_TEST_BASELINE_FOLDER)
        else:
            assert data_version is not None, "Data version not found"
            self._data_dir = osp.join(const.TEST_DATA_FOLDER, dataset_name, data_version)
            self._baseline_dir = osp.join(const.TEST_BASELINE_FOLDER, dataset_name, baseline_version) if \
                baseline_version is not None else None
        self._baseline_params = baseline_params if baseline_params else {}

    def ds_name(self):
        """
        Get the dataset name
        Returns:
            str
        """
        return self.dataset_name

    def get_all_studies(self):
        """
        Get a list with all the studies in a dataset based on the filelist
        :return:
        """
        df = self.get_filelist_df()
        return df.loc[df['study'].notnull(), 'study'].unique().tolist()

    def get_valid_studies(self, run_mode=None):
        """
        Get a list with the name of the studies that are supposed to pass the Checker validations
        Args:
            run_mode (str): one of the run modes defined in deploy constants

        Returns:
            List-str.
        """
        # If needed, the run_mode could be obtained from self._baseline_params['run_mode']
        if self.dataset_name == "DS_01":
            return [self.STUDY_01_HOLOGIC, self.STUDY_03_DBT_HOLOGIC, self.STUDY_04_HOLOGIC, self.STUDY_05_HOLOGIC_COMBO_MODE]
        raise NotImplementedError("I don't have a way to know which studies should pass the Checker")

    def get_invalid_studies(self, run_mode=None):
        """
        Get a list with the studies that are NOT supposed to pass the Checker validations.
        Args:
            run_mode (str): one of the run modes defined in deploy constants.
                            If needed, the run_mode could be obtained from self._baseline_params['run_mode']
        Returns:
            List-str
        """
        if self.dataset_name == "DS_01":
            return [self.STUDY_02_GE]
        raise NotImplementedError("I don't have a way to know which studies should pass the Checker")

    def get_suspicious_studies(self, run_mode=None):
        """
        Get a list with the studies that are Suspicious.
        Args:
            run_mode (str): one of the run modes defined in deploy constants.
                            If needed, the run_mode could be obtained from self._baseline_params['run_mode']
        Returns:
            List-str
        """
        if self.dataset_name == "DS_01":
            return [self.STUDY_01_HOLOGIC, self.STUDY_03_DBT_HOLOGIC]
        raise NotImplementedError("I don't have a way to know which studies should pass the Checker")


    def get_non_suspicious_studies(self, run_mode=None):
        """
        Get non suspicious studies
        Args:
            run_mode (str): one of the run modes defined in deploy constants.
                            If needed, the run_mode could be obtained from self._baseline_params['run_mode']
        Returns:
            List-str
        """
        if self.dataset_name == "DS_01":
            return [self.STUDY_04_HOLOGIC, self.STUDY_05_HOLOGIC_COMBO_MODE]
        raise NotImplementedError("I don't have a way to know which studies should pass the Checker")

    def get_combo_mode_studies(self, run_mode=None):
        """
        Get studies that have FFDM and DBT images
        Args:
            run_mode (str): one of the run modes defined in deploy constants.
                            If needed, the run_mode could be obtained from self._baseline_params['run_mode']
        Returns:
            List-str
        """
        if self.dataset_name == "DS_01":
            return [self.STUDY_05_HOLOGIC_COMBO_MODE]
        raise NotImplementedError("I don't have a way to know which studies should pass the Checker")

    @property
    def data_dir(self):
        """
        Root data dir (where filelist and studies are saved). Ex:/root/test_datasets/data
        Returns:
            str
        """
        return self._data_dir

    @property
    def data_studies_dir(self):
        """
        Root folder for all the studies in the dataset. Ex: /root/test_datasets/data/studies
        Returns:

        """
        return osp.join(self._data_dir, "studies")

    @property
    def baseline_dir(self):
        """
        Root baseline folder. Ex: /root/test_datasets/baseline.
        If self._baseline_params are available, another level in the hierarchy will be added built.
        Ex: /root/test_datasets/baseline/run_mode__cadx__checker_mode__production
        If self._baseline_params is None, the level will be named as "DEFAULT"

        Returns:
            str
        """
        if self._baseline_dir is None:
            raise AssertionError('The baseline directory is being requested, but no baseline version was specified.')
        folder = self.get_baseline_folder_args_suffix(self._baseline_params)
        return os.path.join(self._baseline_dir, folder)



    @property
    def baseline_dir_centaur_output(self):
        """
        Centaur output baseline folder
        Returns:

        """
        return osp.join(self.baseline_dir, const.CENTAUR_OUTPUT_FOLDER)


    # @property
    # def baseline_dir_agg_results(self):
    #     """
    #     Returns:
    #
    #     """
    #     return osp.join(self.baseline_dir, const.AGG_RESULTS_FOLDER)

    @property
    def baseline_dir_test_output(self):
        return osp.join(self.baseline_dir, const.TEST_OUTPUT_FOLDER)

    @property
    def baseline_dir_test_reports(self):
        return osp.join(self.baseline_dir, const.TEST_REPORTS_FOLDER)

    @property
    def baseline_dir_bm_output(self):
        return osp.join(self.baseline_dir, const.BM_OUTPUT_FOLDER)

    @property
    def baseline_dir_bm_reports(self):
        return osp.join(self.baseline_dir, const.BM_REPORTS_FOLDER)

    @classmethod
    def get_baseline_folder_args_suffix(cls, args_dict=None):
        """
        If args_dict are available, another level in the hierarchy will be added built as:
        param1__param1Value__param2__param2Value...
        Otherwise, return "DEFAULT"
        Args:
            args_dict:

        Returns:

        """
        if args_dict:
            return "__".join(map(lambda item: f"{item[0]}__{item[1]}", sorted(args_dict.items())))
        return "DEFAULT"

    def set_baseline_params(self, run_mode):
        """
        Set the most commonly used baseline params.
        Args:
            run_mode (str): one of the run modes defined in centaur_deploy.constants)
        """
        assert run_mode in (const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_CADT), \
            f"run_mode param has an unexpected value ({run_mode})"
        self._baseline_params = {'run_mode': run_mode}

    def set_default_baseline_params(self):
        """
        Set default baseline params for tests that are independent of the "baseline mode"
        """
        self.set_baseline_params(const_deploy.RUN_MODE_CADT)

    def get_filelist_df(self):
        """
        Return the dataframe with the filelist for this dataset.
        First, it tries to load it from cache
        :return: Dataframe with the files info
        """
        if self.__filelist_df is None:
            # Get the list of local_files for this dataset
            filelist_path = osp.join(self.data_dir, "{}.csv".format(const.FILELIST))
            assert osp.isfile(filelist_path), "The filelist for the dataset could not be found in " + filelist_path
            # Read the files
            df = pd.read_csv(filelist_path)
            # Cache the files for future access
            self.__filelist_df = df
        return self.__filelist_df


    def get_files(self, re_filter_pattern=None):
        """
        Get a list of paths that match the files_mask pattern
        :param re_filter_pattern: str. Regular expression pattern.
                                  Ex: All dicom files in a study = '(.*)study/(.*)\\.dcm'
                                  When None, all the files will be returned
        :return: List of str.
        """
        df = self.get_filelist_df()
        if re_filter_pattern is None:
            files_df = df
        else:
            files_df = df[df['file_path'].str.contains(re_filter_pattern)]
        # if self._local_testing_mode:
        #     files = [os.path.join(self.data_dir, f) for f in files_df['file_path'].to_list()]
        # else:
        #     files = (files_df['site_root_folder'] + files_df['file_path']).to_list()
        return files_df['file_path'].to_list()

    def get_dicom_images(self, study):
        """
        Get all the dicom files for a dataset/study
        :param study:
        :return: str-list. List of paths to the dicom files
        """
        pattern = '(.*){}/(.*)\\.dcm'.format(study)
        files = self.get_files(pattern)
        # Exclude non-images DICOM files
        files = [f for f in files if not f.endswith(".pdf.dcm")]
        assert len(files) > 0, "No DICOM images found in the study"

        return files

    def get_input_dir(self, study):
        """
        Get the folder for the DICOM input files for a study
        Args:
            study (str): study name

        Returns:
            str. Folder that contains the DICOM images
        """
        return osp.join(self.data_studies_dir, study)

    @classmethod
    def md5(cls, file_path):
        """
        Get the MD5 hash for a file
        :param file_path: str. Absolute path to the file
        :return: str. MD5 hash
        """
        with open(file_path, "rb") as f:
            md5 = hashlib.md5()
            md5.update(f.read())
            return md5.hexdigest()

