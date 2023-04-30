import os
import glob
import pandas as pd

from centaur_io.input.input import Input


class FileInput(Input):

    def __init__(self, ingress, logger=None):
        """
        Constructor
        Args:
            ingress (str): file/folder input path
            logger (Logger):
        """
        super(FileInput, self).__init__(ingress=ingress, logger=logger)

    def get_ingress(self):
        if os.path.isfile(self.ingress):

            if not any([self.ingress.endswith(ext) for ext in ['.txt', '.pkl', '.csv']]):
                raise IOError(f"Input file {self.ingress} does not have a supported format (please make sure that "
                              f"the extension is part of the file name)")

            file_list = []
            input_paths = []
            if self.ingress.endswith('.txt'):
                with open(self.ingress, 'r') as input_f:
                    input_paths = list(set([str(line) for line in input_f.read().split('\n') if len(str(line)) > 0]))

            elif self.ingress.endswith(('.pkl', '.csv')):
                if self.ingress.endswith('.csv'):
                    input_df = pd.read_csv(self.ingress)
                else:
                    input_df = pd.read_pickle(self.ingress)
                assert 'file_path' in input_df.columns, f"The file_path column is missing from {self.ingress}"
                input_paths = input_df['file_path'].unique().tolist()

            for path in input_paths:
                if os.path.isfile(path):
                    file_list.append(path)
                elif os.path.isdir(path):
                    file_list += [os.path.join(path, fname) for fname in os.listdir(path)]

            assert len(file_list) > 0, "No input files specified"
            input_file_df = pd.DataFrame({'file_path': file_list})
            input_file_df['study_name'] = input_file_df['file_path'].apply(
                lambda p: os.path.basename(os.path.dirname(p)))

            for study_name, study_df in input_file_df.groupby('study_name'):
                study_file_list = study_df['file_path'].unique().tolist()
                yield study_name, study_file_list

        elif not os.path.isdir(self.ingress):
            raise IOError("Directory {} does not exist, not processing.".format(self.ingress))

        else:
            for root, dirs, files in os.walk(self.ingress):
                # Ignore at least hidden files to avoid "dirty" folders
                files = [f for f in files if not f.startswith(".")]
                if len(files) == 0 and root == self.ingress:
                    continue
                file_list = [os.path.join(root, x) for x in files]
                #study_dir = self.get_study_dir(root)
                study_name = os.path.basename(root)
                yield study_name, file_list

    def get_next_study(self):
        # if self.file_lists is None:
        #     self.process_ingress()
        ingress_generator = self.get_ingress()
        # for (study_dir, file_list) in self.file_lists:
        for (study_name, file_list) in ingress_generator:
            yield study_name, file_list, None

    def count_studies(self):
        """
        Counts the number of studies. NOTE: This function does not work as is for the case that
        ingress is a .txt file.
        :return: The number of studies.
        """
        # Assumes file structure:

        # self.ingress
        # - Study Folder 1
        # - Study Folder 2
        # - File 1
        # - File 2

        return len([file for file in os.listdir(self.ingress) if os.path.isdir(os.path.join(self.ingress, file))])

    def count_files(self):
        """
        Counts the number of files in the studies. NOTE: This function does not work as is for the case that
        ingress is a .txt file.
        :return: The number of files in the studies.
        """
        return len([f for f in glob.glob(os.path.join(self.ingress, "**"), recursive=True) if os.path.isfile(f)])
