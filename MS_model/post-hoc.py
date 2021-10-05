import warnings
warnings.filterwarnings("ignore")
    
# Standard imports that we will need in the rest of the program.
import numpy as np
from numpy import inf
from scipy import stats
import time

# Discrete distributions and sampling
from gym import Env, spaces, utils
from gym.utils import seeding

# Inverse reinforcement learning
from inv_reinforce_learning_state_action import (compute_s_a_visitations, 
                                                 vi_boltzmann, 
                                                 compute_D)

# Generate trajectories
from generate_trajectories import generate_trajectories

# Bayesian inference
import emcee
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "1"

# MDP
from environment import MDP, Environment

def loglikeli(theta, n_data, behavior_state, behavior_action):
    theta_state = (theta.reshape(-1,1))[:state_feature.shape[1],:]
    theta_action = (theta.reshape(-1,1))[state_feature.shape[1]:,:]
    r_s = np.squeeze(np.asarray(np.dot(state_feature, theta_state)))
    r_a = np.squeeze(np.asarray(np.dot(action_feature, theta_action)))
    V, Q, policy = vi_boltzmann(mdp, gamma, r_s, r_a, traj_len, temperature)
    s, a = behavior_state[:n_data*traj_len], behavior_action[:n_data*traj_len]
    log_like = np.log(policy[s,a]).sum()
    return np.array(log_like)

def prior(theta, mu=None, sigma=None):
    if mu is None:
        a = (prior_range[0] - prior_mean) / prior_std
        b = (prior_range[1] - prior_mean) / prior_std
        return np.prod(stats.truncnorm.pdf(theta, a, b, 
                                           loc=prior_mean, 
                                           scale=prior_std), axis=0)

def log_prob(theta, n_data, behavior_state, behavior_action):
    return (loglikeli(theta, n_data, behavior_state, behavior_action) 
            + np.log(prior(theta)))
    
def get_kl(theta_post):
    n_sample = theta_post.shape[0]
    kernel = stats.gaussian_kde(theta_post.T)
    kl_div = 0.0
    kl_div += kernel.logpdf(theta_post.T).sum()
    kl_div -= np.log(prior(theta_post.T)).sum()
    kl_div /= n_sample
    return kl_div

if __name__ == '__main__':
    # Read-in controls
    fi = open('post-hoc-controls.inp', 'r+')
    fi.readline()
    path = fi.readline().split()[0] # folder for files to constuct MDP
    print(path)

    fi.readline()
    fi.readline()
    filename = int(fi.readline().split()[0]) # prefix of stored files
    path_store = fi.readline().split()[0] # path to store results

    fi.readline()
    fi.readline()
    tmp = fi.readline().split()[0:2]
    prior_range = [float(tmp[0]), float(tmp[1])] # range of prior
    tmp = fi.readline().split()[0:2]
    prior_mean, prior_std = float(tmp[0]), float(tmp[1]) # mead and std of prior
    gamma = float(fi.readline().split()[0]) # discount factor
    temperature = float(fi.readline().split()[0]) # temperature for softmax
    traj_len = int(fi.readline().split()[0]) # length of each data instance
    tmp = fi.readline().split()
    n_data_list = [] # number of data instances (n_d)
    while True:
        try:
            n_data_list.append(int(tmp[len(n_data_list)]))
        except:
            break
    n_data_list.sort()
    print('n_data_list: ', n_data_list)
    n_traj = n_data_list[-1]

    fi.readline()
    fi.readline()
    n_cores = int(fi.readline().split()[0]) # number of cpu cores
    n_samples = int(fi.readline().split()[0]) # numer of samples in each chain
    print(path, filename, path_store, prior_range, prior_mean, prior_std,
          gamma, temperature, traj_len)
    print(n_data_list)
    print(n_cores, n_samples)
    fi.close()

    # Construct MDP
    # Name of state features, size n_state_feature
    state_feature_name = np.load(path + '/state_feature_name.npy', 
                                 allow_pickle=True)
    n_state_feature = len(state_feature_name)
    # Name of action features, size n_action_feature
    action_feature_name = np.load(path + '/action_feature_name.npy', 
                                  allow_pickle=True)
    n_action_feature = len(action_feature_name)
    n_feature = n_state_feature + n_action_feature
    # State feature matrix, size n_state * n_state_feature
    state_feature = np.load(path + '/state_feature.npy')
    # Action feature matrix, size n_action * n_action_feature
    action_feature = np.load(path + '/action_feature.npy')
    # Transition probability P(s'|s,a)
    # Size n_transitions * 4
    # For each transition, it's a tuple (s, a, s', P(s'|s,a))
    transition_proba = np.load(path + '/transition_proba.npy')
    # Initial state probability, size n_state
    initial_state_dist = np.load(path + '/initial_state_dist.npy')
    # Indexs of valid initial states
    valid_init = np.load(path + '/valid_init_state.npy')
    environment = Environment(state_feature_name, action_feature_name, 
                              state_feature, action_feature, 
                              transition_proba, initial_state_dist)
    mdp = MDP(environment)
    mdp.valid_sa = np.sum(mdp.T, axis=2)
    mdp.valid_s = np.sum(state_feature, axis=1) > 0
    mdp.isd = initial_state_dist

    # Read-in behavior instances
    behavior_instances = np.load(path+'/behavior_instances.npy')

    np.random.seed(filename)
    for i_sample in range(1000):
        a = (prior_range[0] - prior_mean) / prior_std
        b = (prior_range[1] - prior_mean) / prior_std
        # Generate true theta
        theta_expert = stats.truncnorm.rvs(a, b,
                                           loc=prior_mean, 
                                           scale=prior_std, 
                                           size=n_feature)
        # Re-order the behavior instances
        order = np.arange(len(behavior_instances))
        np.random.shuffle(order)
        behavior_instances = behavior_instances[order,:,:]
        behavior_state = behavior_instances[:,:,0].flatten()
        behavior_action = behavior_instances[:,:,1].flatten()
        
        # Create folders to store results
        if not os.path.isdir(path_store + '/trace'):
            os.mkdir(path_store + '/trace')
        if not os.path.isdir(path_store + '/KL'):
            os.mkdir(path_store + '/KL')

        trace_list = []
        kl_list = []
        t_record = time.time()
        for i in range(len(n_data_list)):
            ndim, nwalkers = n_feature, 2 * n_feature
            p0 = np.random.rand(nwalkers, ndim)
            with Pool(n_cores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, 
                    args=[n_data_list[i], behavior_state, behavior_action], 
                    pool=pool)
                sampler.run_mcmc(p0, n_samples, progress=True)
            trace = {}
            chain = sampler.get_chain()
            trace['n_data'] = n_data_list[i]
            trace['chain'] = chain
            trace['log_prob'] = sampler.get_log_prob()
            trace['acceptance_fraction'] = sampler.acceptance_fraction
            data = chain[:,:,:].reshape(-1, ndim)
            kl = []
            for j in range(10):
                kl.append(get_kl(data[np.random.choice(len(data), 
                                 min(10000, n_walkers * n_samples)),:]))
            kl_list.append(np.mean(kl))
            print(n_data_list[i], np.mean(kl))
            trace['kl'] = kl
            trace_list.append(trace)
            np.save(path_store + '/trace/trace_list_{}_{}.npy'
                    .format(filename,i_sample), trace_list)
            np.save(path_store + '/KL/kl_list_{}_{}.npy'
                    .format(filename,i_sample), kl_list)