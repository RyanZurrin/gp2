
import sys

import numpy as np

sys.path.insert(0, '../..')

import gp2
from gp2 import Runner

# to get info about the model run the following:
help(gp2.KUNetPlus2D)

# to use the keras unet_plus2d unet you can specify it as the classifier argument
# to the Runner instance when creating
R = Runner(verbose=True, 
           classifier='kunetplus2d',
           discriminator='cnn',
           filter_num= [16, 32, 64, 128, 256, 512, 1024],
           stack_num_down=3, 
           stack_num_up=3, 
           activation='ReLU', 
           output_activation='Sigmoid', 
           batch_norm=True, 
           pool=True, 
           unpool=True, 
           deep_supervision=False,
           weights=None, 
           freeze_backbone=True, 
           freeze_batch_norm=True,
           optimizer=None, 
           loss=gp2.Util.hybrid_loss, 
           metric=None           
          )


# load data
images = np.load('/hpcstor6/scratch01/r/ryan.zurrin001/GP2TOYEXAMPLE_LARGE/images.npy')
masks = np.load('/hpcstor6/scratch01/r/ryan.zurrin001/GP2TOYEXAMPLE_LARGE/masks.npy')

images = images[:5000]
masks = masks[:5000]

# specifiy weights for distribution of data into train test and val datasets for
# dataset A and B
weights = {
    'A': 0.5,
    'A_train': 0.1,
    'A_val': 0.3,
    'A_test': 0.6,
    'B': 0.3,
    'B_train': 0.7,
    'B_val': 0.1,
    'B_test': 0.2,
    'Z': 0.2
}

# to run training use the Runners run method passing in the images, masks, weights and
# how many training loops to run
R.run(images=images, masks=masks, weights=weights, runs=7)     

# use the runners plot method to easily visualize results
R.plot()
