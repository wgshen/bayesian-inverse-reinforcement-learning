import numpy as np
from numpy import inf

def vi_rational(mdp, gamma, r, horizon=None, threshold=1e-16):
    '''
    Finds the optimal state and state-action value functions via value 
    iteration with the Bellman backup.
    
    Computes the rational policy \pi_{s,a} = \argmax(Q_{s,a}).
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    r : 1D numpy array
        Initial reward vector with the length equal to the 
        number of states in the MDP.
    horizon : int
        Horizon for the finite horizon version of value iteration.
    threshold : float
        Convergence threshold.
    Returns
    -------
    1D numpy array
        Array of shape (mdp.nS, 1), each V[s] is the value of state s under 
        the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a] under the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    '''
    
    V = np.copy(r)

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        
        # Q[s,a] = (r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'})
        Q = r.reshape((-1,1)) + gamma * np.dot(mdp.T, V_prev)
        # V_s = max_a(Q_sa)
        V = np.amax(Q, axis=1)

        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if horizon is not None:
            if t==horizon: break
    
    V = V.reshape((-1, 1))

    # Compute policy
    # Assigns equal probability to taking actions whose Q_sa == max_a(Q_sa)
    # max_Q_index = (Q == np.tile(np.amax(Q,axis=1),(mdp.nA,1)).T)
    # policy = max_Q_index / np.sum(max_Q_index, axis=1).reshape((-1,1))
    policy = np.exp(5*Q)* mdp.valid_sa
    policy /= policy.sum(axis=1).reshape(-1,1)

    return V, Q, policy