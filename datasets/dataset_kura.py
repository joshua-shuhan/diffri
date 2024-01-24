import pickle

import os
import errno
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset

class kura_Dataset(Dataset):
    def __init__(self, num_nodes, eval_length=100, seed=0, train=True, val=False, test_mr=0.5, gt_mr=0.0, density=0.5, noise=False, amortized=False):
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
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_ind)      
            else:
                print(f'Loaded data: {path_ind}')     
                           
            
        input_data = torch.tensor(np.load(path[-1])).float()
        self.observed_values = input_data

        # self.observed_values = (self.observed_values - torch.min(self.observed_values, dim=2, keepdim=True)[0]) / (self.observed_values.max(dim=2, keepdim=True)[0]  - self.observed_values.min(dim=2, keepdim=True)[0] )
        # print(self.observed_values[0,0])
        rand_for_mask = torch.rand_like(self.observed_values)
        self.observed_masks = torch.ones_like(self.observed_values)
        if gt_mr >= 0.1:
          for i in range(self.observed_values.shape[0]):
            mask_len = random.choice([i for i in range(2,5)])
            mask_period = int(self.observed_values.shape[2] * gt_mr / mask_len)
            start = [i for i in range(0, self.observed_values.shape[2]-1, int(self.observed_values.shape[2]/mask_period))][:-1]
            end = [i+mask_len for i in start]
            for index, bin in enumerate(start):
              self.observed_masks[i, :, bin:end[index]]=0

        if train == True:
          self.gt_masks = self.observed_masks #np.ones_like(self.observed_values)
        else:
          rand_for_mask = torch.rand(self.observed_masks.shape) * self.observed_masks
          for i in range(self.observed_masks.shape[0]):
            for j in range(self.observed_masks.shape[1]):
              sample_ratio = test_mr  # missing ratio
              num_observed = sum(self.observed_masks[i,j,:])
              num_masked = (num_observed * sample_ratio).round()
              rand_for_mask[i,j,:][rand_for_mask[i,j].topk(int(num_masked)).indices] = -1
          cond_mask = (rand_for_mask > 0).reshape(self.observed_masks.shape).float()
          self.gt_masks = cond_mask
        self.target = np.load(path[0])
        self.use_index_list = np.arange(self.observed_values.shape[0])
        
    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "targets": self.target
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(train=True, val=True, seed=1, num_nodes=50, batch_size=16, test_mr=0.5, gt_mr=0.5, T=100, density=0.5, noise=False, amortized=False):

    if train == True and val == False:
      dataset = kura_Dataset(train=True, val=False, seed=seed, num_nodes=num_nodes, test_mr=test_mr, gt_mr=gt_mr, eval_length=T, density=density, noise=noise, amortized=amortized)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    elif train == True and val == True:
      dataset = kura_Dataset(train=True, val=True, seed=seed, num_nodes=num_nodes, test_mr=test_mr, gt_mr=gt_mr, eval_length=T, density=density, noise=noise, amortized=amortized)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    elif train == False:
      dataset = kura_Dataset(train=False, seed=seed, num_nodes=num_nodes, test_mr=test_mr, gt_mr=gt_mr, eval_length=T, density=density, noise=noise, amortized=amortized)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    
    return loader