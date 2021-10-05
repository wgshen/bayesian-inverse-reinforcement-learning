import numpy as np

def generate_trajectories(mdp, policy, timesteps=5, num_traj=50):
    '''
    Generates trajectories in the MDP given a policy.
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    policy : 2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    timesteps : int
        Length of each of the generated trajectories.
    num_traj : 
        Number of trajectories to generate.
    
    Returns
    -------
    3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, 2: state & action].
    '''
    
    trajectories = np.zeros([num_traj, timesteps, 2]).astype(int)
    
    s = mdp.reset()
    for i in range(num_traj):
        for t in range(timesteps):
            action = np.random.choice(mdp.nA, p=policy[s, :])
            trajectories[i, t, :] = [s, action]
            s = mdp.step(action)
        s = mdp.reset()
    
    return trajectories