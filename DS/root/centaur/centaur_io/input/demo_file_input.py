import glob
import os

from .file_input import FileInput


class DemoFileInput(FileInput):
    """
    FileInput used for Demo mode. The file structure is different than in regular modes because there are not
    original DICOM images and the metadata is an independent file pre-obtained in the client
    """
    def get_ingress(self):
        """
        Expect a folder structure for the studies like this:
        root_folder
            study1
                metadata.pkl
                img1
                    frame_0.npy
                img2
                    frame_0.npy
                    frame_1.npy
            study2
                ...
        Returns:
            Iterator with study name and list of files, where the first file will always be metadata.pkl
        """
        for study_path in (f.path for f in os.scandir(self.ingress) if f.is_dir()):
            file_list = [f"{study_path}/study_metadata.pkl"]
            file_list.extend(glob.glob(f"{study_path}/**/*.npy", recursive=True))
            yield os.path.basename(study_path), file_list
