import argparse
import torch
import json
import yaml
import os
import numpy as np
from main_model import DiffRI
from datasets.dataset_spr import get_dataloader
from utils import evaluate


parser = argparse.ArgumentParser(description="DiffRI")
parser.add_argument("--config", type=str, default="spr.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--eval-sample", type=int, default=5)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, required=True)
parser.add_argument('--test-mr', default=0.5, type=float)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--T", type=int, required=True)
parser.add_argument("--density", type=float, required=True)
parser.add_argument("--num-node", type=int, required=True)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["time_steps"] = args.T * 2
assert config["exp_set"]["network_density"] == args.density
assert config["exp_set"]["seed"] == args.seed
assert config["exp_set"]["num_node"] == args.num_node


print(json.dumps(config, indent=4))

data_loader = get_dataloader(
    seed=args.seed,
    batch_size=1,
    num_nodes = args.num_node,
    train=False,
    test_mr=args.test_mr,
    T = args.T,
    density = args.density,
)

model = DiffRI(config, args.device, target_dim=args.num_node).to(args.device)

loaded = torch.load("./save/" + args.modelfolder)
if 'model_state_dict' in loaded.keys():
    model.load_state_dict(loaded['model_state_dict'])
else: 
    model.load_state_dict(loaded)

store_result = evaluate(model, data_loader, nsample=args.eval_sample, scaler=1, num_node=args.num_node)

conn_mat = np.load(f'data/spr_seed_{args.seed}_num_node_{args.num_node}_T_{args.T}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_test.npy')
conn_mat = conn_mat[0]
conn_mat = conn_mat != 0
np.fill_diagonal(conn_mat, 0)
print(conn_mat)
# conn_mat = torch.flatten(conn_mat[~np.eye(conn_mat.shape[0],dtype=bool)])

acc_list = np.zeros([store_result.shape[0]])
for i in range(store_result.shape[0]):
    inferred_mat = store_result[i]
    inferred_mat = np.array(inferred_mat)
    corr_w_conn = inferred_mat[conn_mat]
    corr_wo_conn = inferred_mat[~(conn_mat+(np.eye(args.num_node)==1))]

    np.fill_diagonal(inferred_mat, 0)
    threshold = np.sort(inferred_mat, axis=None)[-int(args.density * args.num_node * (args.num_node-1))]
    acc = (((sum(corr_w_conn>=threshold) + sum(corr_wo_conn<threshold)) / (len(corr_w_conn)+len(corr_wo_conn))))    
    acc_list[i] = acc

print(np.mean(acc_list), np.std(acc_list))