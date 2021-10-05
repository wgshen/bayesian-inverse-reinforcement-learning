import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_mean_std(kl_list):
    mu = np.zeros(len(kl_list))
    sigma = np.zeros(len(kl_list))
    for i,kl in enumerate(kl_list):
        mu[i] = np.mean(kl)
        sigma[i] = np.std(kl)
    return mu, sigma

def transform_kl_list(kl_old):
    kl_new = []
    for kl in kl_old:
        for i in range(len(kl)):
            if i+1>len(kl_new):
                kl_new.append([])
            kl_new[i].append(kl[i])
    return kl_new


# Read-in controls
fi = open('pre-hoc-controls.inp', 'r+')
fi.readline()
path = fi.readline().split()[0] # folder for files to constuct MDP

fi.readline()
fi.readline()
fi.readline()
path_sync = fi.readline().split()[0] # path to store results

for _ in range(7):
    fi.readline()
tmp = fi.readline().split()
x_sync = [] # number of data instances (n_d)
while True:
    try:
        x_sync.append(int(tmp[len(x_sync)]))
    except:
        break
x_sync.sort()
fi.close()

fi = open('post-hoc-controls.inp', 'r+')
for _ in range(5):
    fi.readline()
path_true = fi.readline().split()[0] # path to store results

for _ in range(7):
    fi.readline()
tmp = fi.readline().split()
x_true = [] # number of data instances (n_d)
while True:
    try:
        x_true.append(int(tmp[len(x_true)]))
    except:
        break
x_true.sort()
fi.close()

state_feature_name = np.load(path + '/state_feature_name.npy', 
                             allow_pickle=True)
n_state_feature = len(state_feature_name)
action_feature_name = np.load(path + '/action_feature_name.npy', 
                              allow_pickle=True)
n_action_feature = len(action_feature_name)
n_feature = n_state_feature + n_action_feature

files= os.listdir(path_sync+'KL')
files = [file for file in files if file[-4:] == '.npy']
kl_sync = []
for file in files:
    kl = np.load(path_sync+'KL/'+file)
    if not np.isnan(kl).any():
        kl_sync.append(kl)
files= os.listdir(path_true+'KL')
files = [file for file in files if file[-4:] == '.npy']
kl_true = []
for file in files:
    kl = np.load(path_true+'KL/'+file)
    if not np.isnan(kl).any():
        kl_true.append(kl)
    
kl_sync = transform_kl_list(kl_sync)
kl_true = transform_kl_list(kl_true)
mu_sync, sigma_sync = get_mean_std(kl_sync)
mu_true, sigma_true = get_mean_std(kl_true)

plt.figure(figsize=(10,6))
plt.plot(x_sync, mu_sync, '--*', c='navy', label='mean (pre-hoc)')
plt.fill(np.concatenate([x_sync, x_sync[::-1]]), 
         np.concatenate([mu_sync - 1.9600 * sigma_sync, 
                        (mu_sync + 1.9600 * sigma_sync)[::-1]]),
         alpha=.6, fc='c', ec='None', label='95% Confidence Interval (pre-hoc)')
plt.plot(x_true, mu_true, '--*', c='deeppink', label='mean (post-hoc)')
plt.fill(np.concatenate([x_true, x_true[::-1]]), 
         np.concatenate([mu_true - 1.9600 * sigma_true, 
                        (mu_true + 1.9600 * sigma_true)[::-1]]),
         alpha=.6, fc='violet', ec='None', 
         label='95% Confidence Interval (post-hoc)')
plt.legend(loc=4, fontsize=20, title_fontsize=25)
plt.xlabel('$n_d$', fontsize=25)
plt.ylabel('EIG', fontsize=25)
plt.xlim(-20, 1100)
plt.ylim(0, 130)
# plt.yticks(np.linspace(0,180,13))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
# plt.savefig('KL_vs_datanum_true.pdf', bbox_inches='tight')
plt.show()


