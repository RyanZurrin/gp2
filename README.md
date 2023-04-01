# omama

### Modern deep learning systems can detect breast cancer early when trained with large amounts of data. We want to create the world’s largest publicly-available annotated mammography dataset with a collection of 70,000 breast cancer imaging studies with ground truth labels. This dataset will also be the first publicly available dataset that includes both 3D Digital Breast Tomosynthesis (DBT) and 2D Digital Mammography (DM) studies. The number of images of the proposed OMAMA-DB makes fully manual labeling infeasible. To tackle this problem, we will design intelligent annotation methods that join machine learning with human annotators.

## Getting Started with environment setup

### Install the O conda environment by running the following command:
Linux and Mac:
```  bash
conda env create -f O.yml
```
Windows:
```  bash
conda env create -f O.win.yml
```

## Task List for OMAMA
### IMMEDIATE TODO:
- [ ] Update the project proposal document _**[by 4/5/2023]**_
- [ ] make video of downloading and preparing data for standup _**[by 4/2/2023]**_
- [ ] have 3 new datasets being created following the same structure from notebook we were told to copy _**[by 4/3/2023]**_


### LONGTERM TODO:
- [ ] Run Kaggle single case test data through DeepSight classifier _**[by 2/17/2023]**_
- [ ] Set up GP2 to use the Omama data with DeepSight _**[by 3/15/2023]**_


### DONE:
- [x] Create a GitHub repo for the project
- [x] Using GitHub issues as a general task management system
- [x] Create a conda environment for the project
- [x] First standup review meeting _**[2/10/2023]**_
- [x] Access to Chimera was granted for all team members
- [x] Copy GP2 Toy data to shared scratch space for team to use _**[by 2/12/2023]**_
- [x] Have all the environment's setup on Chimera and a general workflow established _**[by 2/16/2023]**_
- [x] Run the GP2 Jupyter Notebooks to gain a better understanding of API use **_[by 2/15/2023]**_
- [x] Convert the Kaggle Dicom Header to the Omama Dicom Header (only the pixels on the two should be different) **_[by 2/14/2023]_**
