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
path_store = fi.readline().split()[0] # path to store results

for _ in range(7):
    fi.readline()
tmp = fi.readline().split()
n_data_list = [] # number of data instances (n_d)
while True:
    try:
        n_data_list.append(int(tmp[len(n_data_list)]))
    except:
        break
n_data_list.sort()

fi.close()

state_feature_name = np.load(path + '/state_feature_name.npy', 
                             allow_pickle=True)
n_state_feature = len(state_feature_name)
action_feature_name = np.load(path + '/action_feature_name.npy', 
                              allow_pickle=True)
n_action_feature = len(action_feature_name)
n_feature = n_state_feature + n_action_feature

path_sync = path_store

files= os.listdir(path_sync+'KL')
files = [file for file in files if file[-4:] == '.npy']
kl_sync = []
for file in files:
    kl = np.load(path_sync+'KL/'+file)
    if not np.isnan(kl).any():
        kl_sync.append(kl)
    
kl_sync = transform_kl_list(kl_sync)
mu_sync, sigma_sync = get_mean_std(kl_sync)

x_sync = n_data_list

a_sync, b_sync = np.polyfit(np.log(x_sync), mu_sync, deg=1)

plt.figure(figsize=(10,6))
plt.plot(x_sync, mu_sync, '--*', c='navy', label='mean')
plt.fill(np.concatenate([x_sync, x_sync[::-1]]), 
         np.concatenate([mu_sync - 1.9600 * sigma_sync, 
                        (mu_sync + 1.9600 * sigma_sync)[::-1]]),
         alpha=.6, fc='c', ec='None', label='95% confidence Interval')
x_grid = np.linspace(1, n_data_list[-1], 1000)
plt.plot(x_grid, a_sync * np.log(x_grid) + b_sync, c='navy', label='fitting')
plt.legend(loc=4, fontsize=20, title_fontsize=25)
plt.xlabel('$n_d$', fontsize=25)
plt.ylabel('EIG', fontsize=25)
# plt.xticks(np.linspace(0,10000,11))
plt.ylim(0, 130)
# plt.yticks(np.linspace(0,200,11))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
# plt.savefig('KL_vs_datanum_sync.pdf', bbox_inches='tight')
plt.show()

n_d_max = int(input("n_d_max (maximum number of samples): "))
while True:
    p = float(input("target percentage: "))
    n_d = np.exp(p / 100.0 * np.log(n_d_max) + 
                    (p - 100) * b_sync / 100 / a_sync)
    print("target n_d: ", int(n_d))














