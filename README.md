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
### TODO:
- [ ] Have all the environment's setup on Chimera and a general workflow established _**[by 2/13/2021]**_
- [ ] Convert the Kaggle Dicom Header to the Omama Dicom Header (only the pixels on the two should be different) **_[by 2/14/2021]_**
- [ ] Copy Omama and GP2 data to shared scratch space for team to use _**[by 2/12/2021]**_
- [ ] Create list of non-technical requirements for the project _**[by 2/12/2021]**_
- [ ] Decide on Hyperparameter optimization library for outlier detection _**[by 2/15/2021]**_
- [ ] Decide on Experiment tracking library for outlier detection _**[by 2/15/2021]**_
- [ ] Run the GP2 Jupyter Notebooks to gain a better understanding of API use **_[by 2/15/2021]**_
- [ ] Establish a bare minimum data access API _**[by 2/17/2021]**_
- [ ] Establish a bare minimum DeepSight API _**[by 2/17/2021]**_
- [ ] Establish a bare minimum outlier detection API _**[by 2/17/2021]**_
- [ ] Get a single PyOD algorithm running with hyperparameter optimization _**[by 2/17/2021]**_
- [ ] Scale up the PyOD hyperparameter optimization to run on all algorithms _**[by 2/25/2021]**_
- [ ] Set up GP2 to use the Omama data with DeepSight _**[by 3/1/2021]**_



### DONE:
- [x] Create a GitHub repo for the project
- [x] Using GitHub issues as a general task management system
- [x] Create a conda environment for the project
- [x] First standup review meeting 2/10/2023
- [x] Access to Chimera was granted for all team members