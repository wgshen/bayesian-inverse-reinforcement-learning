# Standard imports that we will need in the rest of the notebook.
import numpy as np
from numpy import inf
import sparse

# Discrete distributions and sampling
from gym import Env, spaces, utils
from gym.utils import seeding

class MDP(object):
    '''
    MDP object
    Attributes
    ----------
    self.nS : int
        Number of states in the MDP.
    self.nA : int
        Number of actions in the MDP.
    self.P : two-level dict of lists of tuples
        First key is the state and the second key is the action.
        self.P[state][action] is a list of tuples (prob, nextstate, reward).
    self.T : 3D numpy array
        The transition prob matrix of the MDP. p(s'|s,a) = self.T[s,a,s']
    '''
    def __init__(self, env):
        P, nS, nA, desc = MDP.env2mdp(env)
        self.isd = env.isd # initial state probabilities
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means
        self.env = env
        self.transition_tuple = env.transition_tuple
        self.T = self.get_transition_matrix()
        # self.T = self.P
        self.s = self.reset()

    def env2mdp(env):
        return ({s : {a : [tup[:3] for tup in tups]
                for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()},
                env.nS, env.nA, env.desc)

    def get_transition_matrix(self):
        '''Return a matrix with index S,A,S' -> P(S'|S,A)'''
        # T = np.zeros([self.nS, self.nA, self.nS])
        # for s in range(self.nS):
        #     for a in range(self.nA):
        #         transitions = self.P[s][a]
        #         s_a_s = {t[1]:t[0] for t in transitions}
        #         for s_prime in range(self.nS):
        #             if s_prime in s_a_s:
        #                 T[s, a, s_prime] = s_a_s[s_prime]
        # return T
        coords = self.transition_tuple[:,:3].T.astype(np.int)
        data = self.transition_tuple[:,3]
        return sparse.COO(coords, data, shape=(self.nS, self.nA, self.nS))


    def reset(self):
        self.s = np.random.choice(self.nS, p=self.isd)
        return self.s

    def step(self, a, s=None):
        if s == None: s = self.s
        if len(self.P[s][a])==1:
            self.s = self.P[s][a][0][1]
            return self.s
        else:
            p_s_sa = np.asarray(self.P[s][a])[:,0]
            next_state_index = np.random.choice(range(len(p_s_sa)), p=p_s_sa)
            self.s = self.P[s][a][next_state_index][1]
            return self.s

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})    

class Environment(DiscreteEnv):
    """
    Defines the world in which a person decides to bring or not  bring a jacket with them when they leave home.
    
    """

    def __init__(self, state_feature_names, action_feature_names, 
                       state_feature, action_feature, 
                       transition_proba, initial_state_dist):
        
        self.desc = 'TBYSO'
        
        feature_matrix = None
        nA = 0
        nS = 0

        # Flatten.
        self.state_feature_names = state_feature_names

        self.action_feature_names = action_feature_names

        state_feature_num = len(self.state_feature_names)
        action_feature_num = len(self.action_feature_names)

        # # Initial state indicator (1 when initial state, 0 otherwise)
        # isi = []

        nS = state_feature.shape[0]
        nA = action_feature.shape[0]

        #  # Create all possible actions (i.e., reject those that are not possible).
        # for jacket in action_feature_names_dict['Jacket']:
        #     #  There is one action for each feature.
        #     nA = nA + 1

        self.state_feature_matrix = state_feature
        self.action_feature_matrix = action_feature
        
        #  Initial state distribution. Initial states are the one when you are at home before you leave.
        # isd = np.array(isi).astype('float64').ravel()
        # isd /= isd.sum()
        isd = initial_state_dist

        # What does isd tell us about probability of being hot or cold?
        # print("Initial state distribution: ", isd)

        # Create transition matrix.
        # P = transition_proba
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for transition_tuple in transition_proba:
            from_state, action, to_state, probability = transition_tuple
            li = P[int(from_state)][int(action)]
            li.append((probability, int(to_state), 0, False))

        self.transition_tuple = transition_proba

        super(Environment, self).__init__(nS, nA, P, isd)        