import torch

def init(num_node):
    global record_mat
    record_mat = torch.zeros([num_node, num_node])