"""
Initailize the project with necessary folders

- data: saving raw dataset, tokens and train, val data in the format of .bin file 
- configs: saving hyperparameters of the model structure, training and sampling
- checkpoint: for checkpoint files saving during the training
- out: for text file of the final generated output
- params: for final model weight files after training

"""

import os


checkpoint_dir = 'checkpoint'
params_dir = 'params'
save_dir = 'out'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(params_dir):
    os.makedirs(params_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)