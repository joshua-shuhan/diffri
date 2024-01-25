import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from utils import mask


class MLPBlock(nn.Module):
    """The building block for the MLP-based encoder
    """

    def __init__(self, in_dims, hidden_dims, out_dims, do_prob):
        super().__init__()

        self.layer1 = nn.Linear(in_dims, hidden_dims)
        self.elu1 = nn.ELU()

        self.drop_out = nn.Dropout(do_prob)

        self.layer2 = nn.Linear(hidden_dims, out_dims)
        self.elu2 = nn.ELU()

        self.bn = nn.BatchNorm1d(out_dims)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)

        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, input_batch):

        input_batch = input_batch.reshape(input_batch.size(0),
                                       input_batch.size(1), -1)
        out = self.layer1(input_batch)
        out = self.elu1(out)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.elu2(out)
        return self.batch_norm(out)
    
class CNNBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def my_softmax(self, input, axis=1):
        trans_input = input.transpose(axis, 0).contiguous()
        soft_max_1d = F.softmax(trans_input)
        return soft_max_1d.transpose(axis, 0)
    
    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = self.my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
    
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_nodes, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, target_nodes):
        x = self.embedding[target_nodes]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_nodes, dim=64):
        steps = torch.arange(num_nodes).unsqueeze(1)  # (K,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (K,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (K,dim*2)
        return table

    
class edge_prediction_cnn(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = 64
        self.config = config

        if config['model']['is_unconditional'] == True:
            input_dim = 1
        else:
            input_dim = 2 
            
        self.cnn = CNNBlock(input_dim, hidden, hidden, do_prob=0.0)

        self.mlp1 = MLPBlock(2 * hidden, hidden,hidden, do_prob=0.0)
        self.output_projection =  nn.Linear(hidden,2)

        self.diffusion_embedding = DiffusionEmbedding(
            num_nodes=config['model']["number_series"],
            embedding_dim=hidden,
        )
        
        self.init_weights()

        
    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        rel_rec = rel_rec.to(inputs.device)
        rel_send = rel_send.to(inputs.device)

        send_batch = torch.einsum('bij,bjl->bil', rel_send, inputs)
        rec_batch = torch.einsum('bij,bjl->bil', rel_rec, inputs)
        # receivers and senders have shape:
        # [B, K-1, C, L]
        edges = torch.cat([send_batch, rec_batch], dim=2)
        return edges
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, input, target_list):
        B, C, K, L = input.shape
        input = input.permute(0,2,1,3)

        input = input.reshape(input.shape[0] * input.shape[1], input.shape[2], input.shape[3])
        input = self.cnn(input)
        input = input.reshape(B, K, -1)
        node_emb = self.diffusion_embedding([int(i) for i in range(K)])
        node_emb = node_emb[None, :, :]
        node_emb = torch.repeat_interleave(node_emb, B, dim=0)  
        input = input + node_emb
        
        send_mask, rec_mask = mask(K, B, target_list)
        edges = self.node2edge_temporal(input, rec_mask, send_mask)
        
        edges_rep = self.mlp1(edges)
        h_edges = self.output_projection(edges_rep)

        return h_edges
