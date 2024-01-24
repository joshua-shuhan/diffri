import torch

def init(num_node):
    global final_mat
    record_mat = torch.zeros([num_node, num_node])