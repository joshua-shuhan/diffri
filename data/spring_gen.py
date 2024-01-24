import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(description="Kuramoto data generation")
parser.add_argument('--device', default='cpu', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1, required=True)
parser.add_argument("--num-node", type=int, default=10, required=True)
parser.add_argument("--T", type=float, default=5000)
parser.add_argument("--sample-freq", type=float, default=100)

parser.add_argument("--noise", action="store_true", default=False)
parser.add_argument("--nsample", type=int, default=500, required=True)
parser.add_argument("--density", type=float, default=0.5, required=True)
parser.add_argument("--amortized", action="store_true", default=False)

args = parser.parse_args()
print(args)

np.random.seed(args.seed)


class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, network, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        # edges = np.random.choice(self._spring_types,
        #                          size=(self.n_balls, self.n_balls),
        #                          p=spring_prob)
        # edges = np.tril(edges) + np.tril(edges, -1).T
        # np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * network
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * network
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc


# Adjusted parameters for the spring system
num_nodes = args.num_node             # Number of nodes in the system
time_steps = int(args.T / args.sample_freq)           # Number of time steps to simulate

sim = SpringSim(n_balls=num_nodes)


train_time_series_arr = np.zeros([args.nsample, args.num_node, int(args.T/args.sample_freq-1),2])
test_time_series_arr = np.zeros([int(args.nsample / 5), args.num_node, int(args.T/args.sample_freq-1),2])
val_time_series_arr = np.zeros([int(args.nsample / 5), args.num_node, int(args.T/args.sample_freq-1),2])

train_connection_arr = np.zeros([args.nsample, args.num_node, args.num_node])
test_connection_arr = np.zeros([int(args.nsample / 5), args.num_node, args.num_node])
val_connection_arr = np.zeros([int(args.nsample / 5), args.num_node, args.num_node])


# Simulate the spring system with damping

gap_sample = 5 if args.amortized == True else args.nsample + 1

for i in range(args.nsample):

    if i % gap_sample ==0:


        temp=np.random.rand(int(args.num_node * (args.num_node-1) / 2))
        num_zero = int((1-args.density) * args.num_node * (args.num_node-1) / 2)
        temp[np.argpartition(temp,-num_zero)[-num_zero:]]=0
        temp[temp!=0]=1

        graph = np.zeros([args.num_node, args.num_node])
        graph[np.triu_indices(graph.shape[0], k = 1)] = temp
        graph = graph + graph.T - np.diag(np.diag(graph))

    train_connection_arr[i, :, :] = graph

    noise_scale = 0.01 if args.noise == True else 0
    data_gen = sim.sample_trajectory(network=graph, T=args.T, sample_freq=args.sample_freq)
    train_time_series_arr[i,:,:,:] = np.transpose(data_gen, (2, 0, 1))


for i in range(int(args.nsample / 5)):

    if i % gap_sample ==0 and args.amortized:

        temp=np.random.rand(int(args.num_node * (args.num_node-1) / 2))
        num_zero = int((1-args.density) * args.num_node * (args.num_node-1) / 2)
        temp[np.argpartition(temp,-num_zero)[-num_zero:]]=0
        temp[temp!=0]=1

        graph = np.zeros([args.num_node, args.num_node])
        graph[np.triu_indices(graph.shape[0], k = 1)] = temp
        graph = graph + graph.T - np.diag(np.diag(graph))

    test_connection_arr[i, :, :] = graph

    noise_scale = 0.01 if args.noise == True else 0
    data_gen = sim.sample_trajectory(network=graph, T=args.T, sample_freq=args.sample_freq)
    
    test_time_series_arr[i,:,:] =  np.transpose(data_gen, (2, 0, 1))

for i in range(int(args.nsample / 5)):

    if i % gap_sample ==0 and args.amortized:

        temp=np.random.rand(int(args.num_node * (args.num_node-1) / 2))
        num_zero = int((1-args.density) * args.num_node * (args.num_node-1) / 2)
        temp[np.argpartition(temp,-num_zero)[-num_zero:]]=0
        temp[temp!=0]=1

        graph = np.zeros([args.num_node, args.num_node])
        graph[np.triu_indices(graph.shape[0], k = 1)] = temp
        graph = graph + graph.T - np.diag(np.diag(graph))

    val_connection_arr[i, :, :] = graph

    noise_scale = 0.01 if args.noise == True else 0
    data_gen = sim.sample_trajectory(network=graph, T=args.T, sample_freq=args.sample_freq)
    
    val_time_series_arr[i,:,:] = np.transpose(data_gen, (2, 0, 1))

train_traj_path = f'spr_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.sample_freq-1)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_traj_train.npy'
train_conn_path = f'spr_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.sample_freq-1)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_train.npy'

test_traj_path = f'spr_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.sample_freq-1)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_traj_test.npy'
test_conn_path = f'spr_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.sample_freq-1)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_test.npy'

val_conn_path = f'spr_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.sample_freq-1)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_conn_val.npy'
val_traj_path = f'spr_seed_{args.seed}_num_node_{args.num_node}_T_{int(args.T/args.sample_freq-1)}_noise_{args.noise}_density_{args.density}_amort_{args.amortized}_traj_val.npy'


np.save(train_traj_path, train_time_series_arr)
np.save(train_conn_path, train_connection_arr)
np.save(test_traj_path, test_time_series_arr)
np.save(test_conn_path, test_connection_arr)
np.save(val_conn_path, val_connection_arr)
np.save(val_traj_path, val_time_series_arr)