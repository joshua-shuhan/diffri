import argparse
import torch
import datetime
import json
import yaml
import os
import setting
from main_model import DiffRI
from datasets.dataset_kura import get_dataloader
from utils import train, load_checkpoint_train

parser = argparse.ArgumentParser(description="DiffRI")
parser.add_argument("--config", type=str, default="kura.yaml")
parser.add_argument('--device', default='cuda:0')
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--checkpoint-path", type=str,
                    default="", help="give model path for continue training")
parser.add_argument('--gt-mr', default=0.0, type=float, help="the ratio of unobserved data.")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--T", type=int, required=True, help="raw input data length")
parser.add_argument("--density", type=float, required=True, help="retrive datasets with corresponding network density. Also used for the optional regularized term")
parser.add_argument("--num-node", type=int, required=True, help="number of nodes in the network")
parser.add_argument("--no-reg", action="store_true", help="whether to use regularized term")



args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["number_series"] = args.num_node
config["model"]["time_steps"] = args.T

# exp settings
config["exp_set"] = {}
config["exp_set"]["network_density"] = args.density
config["exp_set"]["seed"] = args.seed
config["exp_set"]["num_node"] = args.num_node
if args.no_reg:
    config["exp_set"]["no-reg"] = True
else:
    config["exp_set"]["no-reg"] = False
    
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/kuramoto_{args.seed}_{args.num_node}nodes_{current_time}/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    num_nodes = args.num_node,
    train=True,
    val=False,
    gt_mr=args.gt_mr,
    T=args.T,
    density=args.density,
)

val_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    num_nodes = args.num_node,
    train=True,
    val=True,
    gt_mr=args.gt_mr,
    T=args.T,
    density=args.density,
)
model = DiffRI(config, args.device, target_dim=args.num_node, density=args.density).to(args.device)
setting.init(args.num_node)


continue_train = False

if args.checkpoint_path:
    continue_train = True
    checkpoint_path = './save/' + args.checkpoint_path
    load_checkpoint_train(model, config=config['train'], train_loader=train_loader,checkpoint_path=checkpoint_path, valid_loader=val_loader)

# Start brand new training
if continue_train == False:
    train(
        model,
        config["train"],
        train_loader=train_loader,
        valid_loader=val_loader,
        foldername=foldername,
    )