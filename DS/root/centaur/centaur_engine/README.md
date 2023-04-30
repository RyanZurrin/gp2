
# Centaur Engine 

### Inputs:

1. A preprocessor instance
2. A ModelEnsemble instance
3. Engine config file

### Outputs:
1. Engine object that can check, preprocess, and evaluate on a study.


### Usage

To use this module first import the relevant files into a python script.
```

from centaur_engine.engine import Engine
from centaur_engine.model import ModelEnsemble
from centaur_engine.preprocessor import Preprocessorfrom config import *
from centuar_deploy.constants import *
from centaur_deploy.config import *
```
To create an Engine:

```
model_config = load_config(MODEL_CONFIG_PATH)
engine_config = load_config(ENGINE_CONFIG_PATH)
preprocessor_config = load_config(PREPROCESSOR_CONFIG_PATH)

preprocessor_instance = Preprocessor(preprocessor_config, verbose=args.verbose, ignore_check=args.ignore_check)
modelEnsemble_instnace = ModelEnsemble(model_config)
engine_instance = Engine(preprocessor=pp, ensemble=en, config=engine_config, reuse=False)

```
To Process and Evaluate data:
```
engine_instance.clean()
engine_instance.set_file_list(List_of_files_for_a_study)
engine_instance.set_output_dir(output_dir_for_the_results)

engine_instance.check()
engine_instance.evaluate()
engine_instance.save()
```

### Error Handling:













