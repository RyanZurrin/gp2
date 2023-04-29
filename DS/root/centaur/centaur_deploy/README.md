# Centaur  
  
DeepHealth's breast cancer detection product.  
  
### Prerequisites  
  
This installation guide is specific to running Centaur on Google Cloud Platform.  
  
The following setup has been tested to work on the following specifications:  
  
NVIDIA GPU Cloud Image for Deep Learning and HPC   
Machine type: 8 vCPUs, 30GB memory  
  
Number of GPUs: 1 (Nvidia P100 and above)  
Standard persistent disk: 30GB  
  
Within a GCloud Slurm login node, start a GPU-ready instance:  
```  
sshgpu <gpu-id>  
```  
  
### Installing  
  
 1. Clone this repository to your local home folder and switch to the dev branch: 
```  
git clone https://github.com/DeepHealthAI/centaur_deploy.git  
cd centaur_deploy  
git checkout dev  
```  
  
### Starting Up Centaur  
  
1. The Centaur docker should now be running:  
```  
sudo docker ps -a  
```  
2. Attach the container for interactive mode, taking the container ID from the previous command:  
```  
sudo docker start <container-id>  
sudo docker attach <container-id>  
```  
  
To detach this container at any time:  
```  
ctrl+p+q  
```  
## Deploy:  
### Inputs:  
Input can be one of the following   
1. A directory of studies.   
2. A Text file where each line is a path to a study to be evaluated.   
3. IP address and port of PACS.  
  
### Outputs:  
1. DeepHealth PDF report  
2. MammoSR report  
3. CSV file that summarize all studies for this run session, where each line present the results of one study.  
  
### Usage  
  
To test a study that has already been copied, go into `centaur_deploy` and run:  
```  
python deploy.py --input_dir=/root/test_studies --output_dir=/root/reports  
```  
  
### Error Handling:  
1. If a study did not pass all requirements, the module will display "Study Failed to   
be processed.Study not processed, moving on..." To evaluate these studies regardless, pass in the --ignore_check flag.   
Please keep in mind that the input data did not meet the quality standards specified by Centaur, and may have decreased the accuracy of the results.
=======
# Centaur

DeepHealth's breast cancer detection product.

### Prerequisites

This installation guide is specific to running Centaur on Google Cloud Platform.

The following setup has been tested to work on the following specifications:

NVIDIA GPU Cloud Image for Deep Learning and HPC 
Machine type: 8 vCPUs, 30GB memory

Number of GPUs: 1 (Nvidia P100 and above)
Standard persistent disk: 30GB

Within a GCloud Slurm login node, start a GPU-ready instance:
```
sshgpu <gpu-id>
``


### Starting Up Centaur

1. The Centaur docker should now be running:
```
sudo docker ps -a
```
2. Attach the container for interactive mode, taking the container ID from the previous command:
```
sudo docker start <container-id>
sudo docker attach <container-id>
```

To detach this container at any time:
```
ctrl+p+q
```
## Deploy:
### Inputs:
Input can be one of the following 
1. A directory of studies. 
2. A Text file where each line is a path to a study to be evaluated. 
3. IP address and port of PACS.

### Outputs:
1. DeepHealth PDF report
2. MammoSR report
3. CSV file that summarize all studies for this run session, where each line present the results of one study.

### Usage

To test a study that has already been copied, go into `centaur_deploy` and run:
```
python deploy.py --input_dir=/root/test_studies --output_dir=/root/reports
```

### Error Handling:
1. If a study did not pass all requirements, the module will display "Study Failed to 
be processed.Study not processed, moving on..." To evaluate these studies regardless, pass in the --ignore_check flag. 
Please keep in mind that the input data did not meet the quality standards specified by Centaur, and may have decreased the accuracy of the results.

### FAQ and Common Issues
* In order to have permissions to perform certain operations on a gpu or login node that you are accessing remotely via slurm or another method, you might need to first SSH into the gpu or login node directly at console.cloud.google.com.
* As of 3 July 2019, to run docker.py, you must be in the centaur_deploy/docker directory.
