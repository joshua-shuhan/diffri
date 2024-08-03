import pickle

import os
import errno
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset


class kura_Dataset(Dataset):
    def __init__(self, num_nodes, eval_length=100, seed=0, train=True, val=False, gt_mr=0.0, density=0.5, noise=False, amortized=False):
        self.eval_length = eval_length
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        if train == True and val == False:
            path = (
                f"data/kura_seed_{seed}_num_node_{num_nodes}_T_{eval_length}_noise_{noise}_density_{density}_amort_{amortized}_conn_train.npy",
                f"data/kura_seed_{seed}_num_node_{num_nodes}_T_{eval_length}_noise_{noise}_density_{density}_amort_{amortized}_traj_train.npy",
            )
        if train == False:
            path = (
                f"data/kura_seed_{seed}_num_node_{num_nodes}_T_{eval_length}_noise_{noise}_density_{density}_amort_{amortized}_conn_test.npy",
                f"data/kura_seed_{seed}_num_node_{num_nodes}_T_{eval_length}_noise_{noise}_density_{density}_amort_{amortized}_traj_test.npy",

            )
        if train == True and val == True:
            path = (
                f"data/kura_seed_{seed}_num_node_{num_nodes}_T_{eval_length}_noise_{noise}_density_{density}_amort_{amortized}_conn_val.npy",
                f"data/kura_seed_{seed}_num_node_{num_nodes}_T_{eval_length}_noise_{noise}_density_{density}_amort_{amortized}_traj_val.npy",
            )

        for path_ind in path:
            if not os.path.isfile(path_ind):
                print("Please create data first")
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), path_ind)
            else:
                print(f'Loaded data: {path_ind}')

        input_data = torch.tensor(np.load(path[-1])).float()
        self.observed_values = input_data
        rand_for_mask = torch.rand_like(self.observed_values)
        
        # Randomly mask some values. The model never see masked parts.
        # mimic missing values in real-world data. Only observed values (where observed_masks=1) are used for training.
        self.observed_masks = torch.ones_like(self.observed_values)        
        if gt_mr != 0:
          for i in range(self.observed_values.shape[0]):
            for j in range(self.observed_masks.shape[1]):
              sample_ratio = gt_mr  # missing ratio
              num_observed = sum(self.observed_masks[i,j,:])
              num_masked = (num_observed * sample_ratio).round()
              rand_for_mask[i,j,:][rand_for_mask[i,j].topk(int(num_masked)).indices] = -1
        self.observed_masks = (rand_for_mask > 0).reshape(self.observed_masks.shape).float()
        # use following code if you are training the model with truly existing missing values
        # self.observed_masks = ~np.isnan(observed_values)

        self.target = np.load(path[0])
        self.use_index_list = np.arange(self.observed_values.shape[0])
        
    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "timepoints": np.arange(self.eval_length),
            "targets": self.target
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(train=True, val=True, seed=1, num_nodes=50, batch_size=16, gt_mr=0.5, T=100, density=0.5, noise=False, amortized=False):

    # only to obtain total length of dataset
    if train == True and val == False:
      dataset = kura_Dataset(train=True, val=False, seed=seed, num_nodes=num_nodes, gt_mr=gt_mr, eval_length=T, density=density, noise=noise, amortized=amortized)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    elif train == True and val == True:
      dataset = kura_Dataset(train=True, val=True, seed=seed, num_nodes=num_nodes, gt_mr=gt_mr, eval_length=T, density=density, noise=noise, amortized=amortized)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    elif train == False:
      dataset = kura_Dataset(train=False, seed=seed, num_nodes=num_nodes, gt_mr=gt_mr, eval_length=T, density=density, noise=noise, amortized=amortized)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    
    return loader