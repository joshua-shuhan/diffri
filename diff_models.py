import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
import numpy as np
from edge_prediction import edge_prediction_cnn
import setting

class MLPBlock(nn.Module):
    """The building block for the MLP-based encoder
    """

    def __init__(self, in_dims, hidden_dims, out_dims, do_prob):
        super().__init__()
        self.layer1 = nn.Linear(in_dims, hidden_dims)
        self.drop_out = nn.Dropout(do_prob)
        self.layer2 = nn.Linear(hidden_dims, out_dims)
        self.gate = nn.LeakyReLU(0.05)
    def forward(self, input_batch):
        out = self.layer1(input_batch)
        out = self.gate(out)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.gate(out)
        return out
    
def reparametrization(z_mean, z_log_var):
    epsilon = np.random.normal(loc=1, scale=0.5) #torch.randn(z_mean.shape[0], z_mean.shape[1])
    return z_mean + z_log_var * epsilon

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float() 
    # print(U)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10, epoch_no=None):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)

    y = logits + gumbel_noise.to(logits.device)

    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, epoch_no=None, hard=False, eps=1e-10):
    """
    NOTE: From https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps, epoch_no=epoch_no)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape).to(y_soft.device)
        k = k.to(y_soft.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class TargetEmbedding(nn.Module):
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

    
class DiffusionStepEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_models(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config['diffusion']["channels"]

        self.diffusion_embedding = DiffusionStepEmbedding(
            num_steps=config['diffusion']["num_steps"],
            embedding_dim=config['diffusion']["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)

        self.relation = edge_prediction_cnn(config)

        nn.init.zeros_(self.output_projection2.weight)

        side_dims = config["model"]["timeemb"] + config["model"]["featureemb"] if config['model']['is_unconditional'] == True else config["model"]["timeemb"] + config["model"]["featureemb"]+1

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dims,
                    channels=self.channels,
                    diffusion_embedding_dim=config['diffusion']["diffusion_embedding_dim"],
                    nheads=config['diffusion']["nheads"],
                    config=config
                )
                for _ in range(config['diffusion']["layers"])
            ]
        )

        # self.latest_graph = torch.zeros(config['model']["number_series"], config['model']["number_series"]-1)

    def forward(self, x, cond_info, diffusion_step, target_list, **kwargs):
        B, inputdim, K, L = x.shape
        # assign parameters values
        epoch_no = kwargs['epoch_no']

        # h_edges: (B, K-1, 2)
        h_edges = self.relation(x, target_list)

        sampled_edges = gumbel_softmax(h_edges, epoch_no=epoch_no, tau=1, hard=True)

        edges = sampled_edges[:,:,0] 
        pre_edges = sampled_edges[:,:,0] 
        reg_loss = pre_edges 

        edges_temp = pre_edges.clone().detach() 
        edges_new = torch.zeros(B, K).to(edges.device)
        edges_temp_new = torch.zeros(B, K)
        for i in range(B):
            edges_new[i] = torch.cat([edges[i, :target_list[i].long()],  torch.tensor([1]).to(edges.device), edges[i, target_list[i].long():]])
            edges_temp_new[i] = torch.cat([edges_temp[i, :target_list[i].long()], torch.tensor([1]).to(edges_temp.device), edges_temp[i, target_list[i].long():]])

        for i in range(B):
            setting.record_mat[target_list[i].long()] += edges_temp_new[i]
        # print(setting.record_mat[:10,:10])

        # time_reg = torch.tensor([torch.sum(torch.abs(self.latest_graph[target_list[i].long()].to(edges.device) - edges[i])) for i in range(B)])
        # time_reg = torch.mean(time_reg)

        # for i in range(B):
        #     self.latest_graph[target_list[i].long()] = edges[i] 

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
            
        x = x.reshape(B, self.channels, K, L)
        x = F.relu(x)
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, edges_new, target_list=target_list)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, 1 * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, 1, L)
        return x, reg_loss #, time_reg

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, config):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.cond_projection_2 = nn.Linear(config['model']["number_series"] * config['model']['time_steps'],config['model']['time_steps'])
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        if config['model']['feature_layer'] == 'lstm':
            self.feature_layer = torch.nn.LSTM(input_size=config['model']["number_series"], hidden_size=256,batch_first=True,num_layers=1,dropout=0.5,proj_size=1)
        elif config['model']['feature_layer'] == 'mlp':
            self.feature_layer = MLPBlock(in_dims=config['model']["number_series"], hidden_dims=64 * config['model']["number_series"] // 5,out_dims=1, do_prob=0.5) # nn.ModuleList(MLPBlock(in_dims=config['model']["number_series"], hidden_dims=config['model']["number_series"] // 2, out_dims=1, do_prob=0.0) for i in range(config['model']["number_series"]))
        self.target_embedding = TargetEmbedding(
            num_nodes=config['model']["number_series"],
            embedding_dim=channels ,
        )
        self.init_weights()
        self.config = config
        upper_indices = np.triu_indices(config['model']['time_steps'], k = 1)
        self.causal_mask = torch.zeros((config['model']['time_steps'], config['model']['time_steps']))
        self.causal_mask[upper_indices] = -float('inf')
        # print(self.causal_mask)

        self.pre_feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        
    def node2edge(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        # Inputs has shape [B, L, C, K]
        rel_rec = rel_rec.to(inputs.device)
        rel_send = rel_send.to(inputs.device)

        send_batch = torch.einsum('bclk,bek->becl', inputs, rel_send)
        rec_batch = torch.einsum('bclk,bek->becl', inputs, rel_rec)
        # receivers and senders have shape:
        # [B, K-1, 2C, L]
        edges = torch.cat([send_batch, rec_batch], dim=2)
        edges = edges.permute(0,3,1,2)
        return edges
    
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        
        y = self.time_layer(y.permute(2, 0, 1), mask=self.causal_mask, is_causal=True).permute(1, 2, 0)
        #y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y= y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature_interaction(self, y, base_shape, self_edges, target_list):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 1, 3, 2)
        self_edges_aug = self_edges[:, None, None, :]
        self_edges_aug = torch.repeat_interleave(self_edges_aug, repeats=L, dim=2)
        self_edges_aug = torch.repeat_interleave(self_edges_aug, repeats=channel, dim=1)
    
        l1 = torch.arange(1, L+1)
        l1_temp = l1.unsqueeze(1)
        l2_temp = l1.unsqueeze(0)
        window_size = 20
        filter = (torch.abs(l1_temp - l2_temp) <= window_size) 
        # upper_indices = np.triu_indices(L, k = 0)
        # filter[upper_indices] = 0
        filter = filter / ( 2 * window_size + 1)
        y_blur = torch.einsum('ts,bctk->bcsk', filter.to(y.device), y)
        y_masked = y_blur + self_edges_aug * (y - y_blur)
        #y_masked = self_edges_aug * y

        if self.config['model']['feature_layer'] == 'lstm':
            y_masked = torch.reshape(y_masked, (B*channel, L, K))
            y = torch.squeeze(self.feature_layer(y_masked)[0])
            y = torch.reshape(y, (B, channel,L))
        elif self.config['model']['feature_layer'] == 'mlp':
            y = torch.squeeze(self.feature_layer(y_masked))
        elif self.config['model']['feature_layer'] == 'sum':
            y = torch.max(y_masked, dim=3)[0]
        return y

    def forward(self, x, cond_info, diffusion_emb, self_edges, target_list):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature_interaction(y, base_shape, self_edges, target_list)  # (B,channel,L)
        y = self.mid_projection(y)  # (B,2*channel,L)
        _, cond_dim, _, _ = cond_info.shape
    
        cond_info = cond_info.reshape(B, cond_dim, cond_info.shape[2] * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,L)
        cond_info = self.cond_projection_2(cond_info)
        y = y + cond_info
        
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        x = x.reshape(B,channel,K,L)
        if K == 1:
            x_new = x
        else:
            x_new = x[[i for i in range(B)],:,target_list.long(),:].clone()
            x_new = torch.unsqueeze(x_new, dim=2)

        residual = residual.reshape(B,channel,1,L)
    
        skip_new = skip.reshape(B,channel,1,L)

        return (x_new + residual) / math.sqrt(2.0), skip_new