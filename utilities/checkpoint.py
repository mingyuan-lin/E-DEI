import glob
import os
import torch
from natsort import natsorted


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.runs = natsorted(glob.glob(os.path.join(self.args.model_path, 'exp_*')))
        self.run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.args.model_path, 'exp_{}'.format(str(self.run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_experiment_config()

    def save_checkpoint(self, model, filename='model.pth'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(model.state_dict(), filename)

    def save_experiment_config(self):
        with open(os.path.join(self.experiment_dir, 'parameters.txt'), 'w') as file:
            config_dict = vars(self.args)
            for k in vars(self.args):
                file.write(f"{k}={config_dict[k]} \n")
