# centaur-io


### Inputs:


#### FileIO:
1. Study input

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) A directory of studies. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; or 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b)  A Text file where each line is a path to a study to be evaluated. 

2. An output directory: where the outputs of Centaur will be stored. 

#### PacsIO:
1. IP address of the PACS
2. port of PACS.

### Outputs:
1. IO object that stores information regarding Centaur inputs and outputs. This class is also able
output a generator that yield the next study to be evaluated.


### Usage

To use this module first import the relevant file into a python script.
```
from centaur_io.io_file import FileIO
io_instance = FileIO(ingress=input_dir, egress=output_dir)
```
To create a study generator:

```
studies_generator = io_instance.get_next_study()

```


### Error Handling:
1. "Invalid study directory, continuing" --- make sure the path to each study is correct in the text file
2. "Directory does not exist" --- make sure the input_dir is correctly entered

