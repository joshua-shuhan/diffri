import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import itertools

from kuramoto import Kuramoto
import math
import random
import warnings
warnings.filterwarnings("error")

parser = argparse.ArgumentParser(description="Kuramoto data generation")
parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1, required=True)
parser.add_argument("--num-node", type=int, default=10, required=True)
parser.add_argument("--T", type=float, default=5)
parser.add_argument("--dt", type=float, default=0.05)
parser.add_argument("--noise", action="store_true", default=False)
parser.add_argument("--nsample", type=int, default=500, required=True)
parser.add_argument("--density", type=float, default=0.5, required=True)
parser.add_argument("--amortized", action="store_true", default=False)

args = parser.parse_args()
print(args)

np.random.seed(args.seed)

train_time_series_arr = np.zeros([args.nsample, args.num_node, int(args.T/args.dt)])
train_connection_arr = np.zeros([args.nsample, args.num_node, args.num_node])

val_time_series_arr = np.zeros([int(args.nsample // 5), args.num_node, int(args.T/args.dt)])
val_connection_arr = np.zeros([int(args.nsample // 5), args.num_node, args.num_node])


test_time_series_arr = np.zeros([int(args.nsample // 5), args.num_node, int(args.T/args.dt)])
test_connection_arr = np.zeros([int(args.nsample // 5), args.num_node, args.num_node])

# graph_nx = nx.erdos_renyi_graph(n=args.num_node, p=0.5) # p=1 -> all-to-all connectivity

freq_list = np.random.uniform(low=2, high=10, size=args.num_node) #[2 * math.pi] * args.num_node # np.random.normal(size=args.num_node)

gap_sample = 1 if args.amortized == True else args.nsample + 1

list_1 = [i for i in range(args.num_node)]

for i in range(args.nsample):
    if i % gap_sample ==0:
        temp=np.random.rand(int(args.num_node * (args.num_node-1) / 2))
        num_zero = int((1-args.density) * args.num_node * (args.num_node-1) / 2)
        temp[np.argpartition(temp,-num_zero)[-num_zero:]]=0
        temp[temp!=0]=1

        graph = np.zeros([args.num_node, args.num_node])
        graph[np.triu_indices(graph.shape[0], k = 1)] = temp
        graph = graph + graph.T - np.diag(np.diag(graph))
        print(graph)
    model = Kuramoto(coupling=2, dt=args.dt, T=args.T, n_nodes=args.num_node, natfreqs=freq_list)
    act_mat = model.run(adj_mat=graph)

    train_connection_arr[i, :, :] = graph

    if args.noise:
        train_time_series_arr[i, :, :] = np.sin(act_mat) + 0.05 * np.random.randn(act_mat.shape[0], act_mat.shape[1])
    else:
        train_time_series_arr[i, :] = np.sin(act_mat)

        #train_time_series_arr[i, :, 0] = act_mat[:, 1] -  act_mat[:, 0]
        #train_time_series_arr[i, :, 1:] = act_mat[:, 1:] - act_mat[:, :-1]
        
np.random.seed(args.seed+1)

for i in range(int(args.nsample // 5)):
    if i % gap_sample == 0 and args.amortized:
        temp=np.random.rand(int(args.num_node * (args.num_node-1) / 2))
        num_zero = int((1-args.density) * args.num_node * (args.num_node-1) / 2)
        temp[np.argpartition(temp,-num_zero)[-num_zero:]]=0
        temp[temp!=0]=1

        graph = np.zeros([args.num_node, args.num_node])
        graph[np.triu_indices(graph.shape[0], k = 1)] = temp
        graph = graph + graph.T - np.diag(np.diag(graph))
        print(graph)

    model = Kuramoto(coupling=2, dt=args.dt, T=args.T, n_nodes=args.num_node, natfreqs=freq_list)
    act_mat = model.run(adj_mat=graph)
    
    test_connection_arr[i, :, :] = graph

    if args.noise:
        test_time_series_arr[i, :, :] = np.sin(act_mat) + 0.05 * np.random.randn(act_mat.shape[0], act_mat.shape[1])
    else:
        test_time_series_arr[i, :,] = np.sin(act_mat)
        #test_time_series_arr[i, :, 0] = act_mat[:, 1] -  act_mat[:, 0]
        #test_time_series_arr[i, :, 1:] = act_mat[:, 1:] - act_mat[:, :-1]

for i in range(int(args.nsample // 5)):
    if i % gap_sample == 0 and args.amortized:
        temp=np.random.rand(int(args.num_node * (args.num_node-1) / 2))
        num_zero = int((1-args.density) * args.num_node * (args.num_node-1) / 2)
        temp[np.argpartition(temp,-num_zero)[-num_zero:]]=0
        temp[temp!=0]=1

        graph = np.zeros([args.num_node, args.num_node])
        graph[np.triu_indices(graph.shape[0], k = 1)] = temp
        graph = graph + graph.T - np.diag(np.diag(graph))
        print(graph)

    model = Kuramoto(coupling=2, dt=args.dt, T=args.T, n_nodes=args.num_node, natfreqs=freq_list)
    act_mat = model.run(adj_mat=graph)
    
    val_connection_arr[i, :, :] = graph

    if args.noise:
        val_time_series_arr[i, :, :] = np.sin(act_mat) + 0.05 * np.random.randn(act_mat.shape[0], act_mat.shape[1])
    else:
        val_time_series_arr[i, :,] = np.sin(act_mat)
        #val_time_series_arr[i, :, 0] = act_mat[:, 1] -  act_mat[:, 0]
        #val_time_series_arr[i, :, 1:] = act_mat[:, 1:] - act_mat[:, :-1]

train_traj_path = f'kura_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.dt)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_traj_train.npy'
train_conn_path = f'kura_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.dt)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_train.npy'

test_traj_path = f'kura_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.dt)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_traj_test.npy'
test_conn_path = f'kura_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.dt)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_test.npy'

val_traj_path = f'kura_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.dt)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_traj_val.npy'
val_conn_path = f'kura_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.dt)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_val.npy'


np.save(train_traj_path, train_time_series_arr)
np.save(train_conn_path, train_connection_arr)
np.save(test_traj_path, test_time_series_arr)
np.save(test_conn_path, test_connection_arr)
np.save(val_conn_path, val_connection_arr)
np.save(val_traj_path, val_time_series_arr)