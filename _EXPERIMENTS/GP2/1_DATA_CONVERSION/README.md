# STEP 1: DATA CONVERSION

This folder contains data conversion notebooks utilized with the GP2 framework, which is being developed 
as part of the Omama-DB and CS410 UMB Intro to Software Engineering Project.

## Overview

The notebooks in this folder are responsible for loading, processing, and converting various segmentation 
datasets with binary masks into a format suitable for training and evaluating machine learning models using 
the GP2 framework. The example provided in this folder is based on an ocular disease dataset.

## Data

The data being used in the example notebooks in this folder are stored in local scratch spaces based on each developer's 
specific scratch space location. For the example, the Ocular Disease dataset is stored in the 
/hpcstor6/scratch01/r/ryan.zurrin001/Ocular_disease/ directory. The images and masks can be found in the images and masks 
subdirectories, respectively. To work with other segmentation datasets, make sure to update the dataset directory in the 
notebook accordingly.

## Notebooks

The main notebook in this folder is responsible for:

1. Loading the image and mask data from the dataset directory.
2. Preprocessing the images and masks, including resizing and normalization.
3. Creating NumPy arrays for the processed images and masks.
4. Saving the processed data as NumPy .npy files.

## Usage

To use the notebooks in this folder, follow these steps:

1. Ensure that all required dependencies are installed (e.g., NumPy, Mahotas, skimage, and matplotlib).
2. Update the DATAFOLDER variable in the notebook to point to the correct dataset directory in the respective developer's scratch space.
3. Run the notebook cells to load, preprocess, and save the data.
4. Set the permissions on the dataset to ensure that everyone has access to it. You can do this using the chmod command on UNIX-based systems, for example:
    ```bash
    chmod -R 755 /path/to/your/dataset_directory
    ```
5. The processed data can now be used with the GP2 framework for training and evaluating models.

For other segmentation datasets, modify the notebook accordingly to load and process the specific dataset, and ensure that the data is saved in the appropriate format.

## Dependencies

* NumPy
* Mahotas
* skimage
* matplotlib

## Contact

For any questions, issues, or suggestions related to this folder or the project in general, please contact the project team members.