# Standard imports that we will need in the rest of the notebook.
import numpy as np
from numpy import inf

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
        self.T = self.get_transition_matrix()
        # self.T = self.P
        self.s = self.reset()

    def env2mdp(env):
        return ({s : {a : [tup[:3] for tup in tups]
                for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()},
                env.nS, env.nA, env.desc)

    def get_transition_matrix(self):
        '''Return a matrix with index S,A,S' -> P(S'|S,A)'''
        T = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.P[s][a]
                s_a_s = {t[1]:t[0] for t in transitions}
                for s_prime in range(self.nS):
                    if s_prime in s_a_s:
                        T[s, a, s_prime] = s_a_s[s_prime]
        return T

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

        # # Create all possible states (i.e., reject those that are not possible).
        # for weather in state_feature_names_dict['Weather']:
        #     for at_home in state_feature_names_dict['At Home']:
        #         for has_jacket in state_feature_names_dict['Has Jacket']:

        #             # If you are at home you always have a jacket.
        #             if at_home == 'True' and has_jacket == 'False':
        #                 continue
                        
        #             for holding_jacket in state_feature_names_dict['Holding Jacket']:
                        
        #                 # If you are at home you are not holding your jacket.
        #                 if at_home == 'True' and holding_jacket == 'True':
        #                     continue
                            
        #                 # If you are not at home you can hold the jacket only when you have it.
        #                 if at_home == 'False' and has_jacket == 'False' and holding_jacket == 'True':
        #                     continue

        #                 for wearing_jacket in state_feature_names_dict['Wearing Jacket']:

        #                     # Come on! Nobody wears jackets at home.
        #                     if at_home == 'True' and wearing_jacket == 'True':
        #                         continue

        #                     # You can only wear jacket when you have it with you.
        #                     if has_jacket == 'False' and wearing_jacket == 'True':
        #                         continue
                                
        #                     # You cannot wear and hold the jacket and the same time.
        #                     if holding_jacket == 'True' and wearing_jacket == 'True':
        #                         continue
                                
        #                     # If you are outside and you have a jacket, you must either wearing it or hold it.
        #                     if at_home == 'False' and has_jacket == 'True' and holding_jacket == 'False' and wearing_jacket == 'False':
        #                         continue
                                
        #                     # If you are outside and you don't have a jacket, you can't hold it or wear it.
        #                     if at_home == 'False' and has_jacket == 'False' and (holding_jacket == 'True' or wearing_jacket == 'True'):
        #                         continue

        #                     for feeling in state_feature_names_dict['Feeling']:

        #                         # When at home you do not wear a jacket and feel just right.

        #                         if at_home == 'True':
        #                             if not(feeling == 'Just Right'):
        #                                 continue                            
        #                         elif at_home == 'False':
        #                             if weather == 'Cold':
        #                                 #If the weather is cold, you can feel just right only when wearing a jacket, and cold otherwise.
        #                                 if wearing_jacket == 'True' and not(feeling == 'Just Right'):
        #                                     continue
        #                                 elif wearing_jacket == 'False' and not(feeling == 'Cold'):
        #                                     continue

        #                             elif weather == 'Hot':
        #                                 #If the weather is hot, you can feel just right only when not wearing a jacket, and hot otherwise.
        #                                 if wearing_jacket == 'True' and not(feeling == 'Hot'):
        #                                     continue
        #                                 elif wearing_jacket == 'False' and not(feeling == 'Just Right'):
        #                                     continue

        #                         #  We are here, so it must mean it is a valid state because we haven't rejected it.
        #                         current_state = np.array([weather == 'Hot', weather == 'Cold', at_home == 'True', at_home == 'False', has_jacket == 'True', has_jacket == 'False', holding_jacket == 'True', holding_jacket == 'False', wearing_jacket == 'True', wearing_jacket == 'False', feeling == 'Cold', feeling == 'Just Right', feeling == 'Hot'], dtype = int)

        #                         print('State: ', ['Weather:'+weather, 'At Home:' + at_home, 'Has Jacket:' + has_jacket, 'Holding Jacket:' + holding_jacket, 'Wearing Jacket:' + wearing_jacket, 'Feeling:' + feeling])

        #                         if feature_matrix is None:
        #                             feature_matrix = np.matrix(current_state).T
        #                         else:
        #                             feature_matrix = np.concatenate((feature_matrix, np.matrix(current_state).T), axis=1)


        #                         isi.append(at_home == 'True')

        #                         nS = nS + 1

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
        print("Initial state distribution: ", isd)

        # Create transition matrix.
        # P = transition_proba
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for from_state in range(nS):
            # print(from_state)
            for action in range(nA):
                for to_state in range(nS):
                    if transition_proba[from_state,action,to_state]==0: continue
                    li = P[from_state][action]

                    # from_state_weather_cold = feature_matrix.item(1,from_state)
                    # from_state_at_home_true = feature_matrix.item(2,from_state)
                    # from_state_has_jacket_true = feature_matrix.item(4,from_state)
                    # from_state_holding_jacket_true = feature_matrix.item(6,from_state)
                    # from_state_wearing_jacket_true = feature_matrix.item(8,from_state)
                    # from_state_feeling_hot = feature_matrix.item(10,from_state)
                    # from_state_feeling_justright = feature_matrix.item(11,from_state)
                    # from_state_feeling_cold = feature_matrix.item(12,from_state)

                    # to_state_weather_cold = feature_matrix.item(1,to_state)
                    # to_state_at_home_true = feature_matrix.item(2,to_state)
                    # to_state_has_jacket_true = feature_matrix.item(4,to_state)
                    # to_state_holding_jacket_true = feature_matrix.item(6,to_state)
                    # to_state_wearing_jacket_true = feature_matrix.item(8,to_state)
                    # to_state_feeling_hot = feature_matrix.item(10,to_state)
                    # to_state_feeling_justright = feature_matrix.item(11,to_state)
                    # to_state_feeling_cold = feature_matrix.item(12,to_state)

                    # # Initialize transition probability
                    # p = 1.0

                    # # The weather can change and you have no control over this!
                    # p *= weather_transition_matrix.item(from_state_weather_cold, to_state_weather_cold)

                    # # Can only transition from home to outside and from outside to outside.
                    # if from_state_at_home_true == 1 and to_state_at_home_true == 1:
                    #     # We cannot stay at home. Have to keep moving.
                    #     p = 0.0

                    # if from_state_at_home_true == 0 and to_state_at_home_true == 1:
                    #     # We cannot go home. Not in this example.
                    #     p = 0.0

                    # # Where are you?
                    # if(from_state_at_home_true == 1):
                    #     # You are inside.
                    #     if action == 0:
                    #         # Bring. That means in the next state you will have it but not wear it or p is 0.
                    #         if to_state_has_jacket_true == 0 or to_state_wearing_jacket_true == 1:
                    #             p = 0
                    #     elif action == 1:
                    #         # Leave. That means in the next state you will not have it and thus not be wearing it or p is 0.
                    #         if to_state_has_jacket_true == 1:
                    #             p = 0
                    #     elif action == 2:
                    #         # Put on. That means in the next state you will have it and wear it or p is 0.
                    #         if to_state_has_jacket_true == 0 or to_state_wearing_jacket_true == 0:
                    #             p = 0
                    #     elif action == 3:
                    #         # Can't take off at home because you are not wearing it.
                    #         p = 0

                    # else:
                    #     # You are outside.
                    #     #If you have jacket you must have it with you no matter what you do and if you don't you can't get it.
                    #     if not(from_state_has_jacket_true == to_state_has_jacket_true):
                    #         p = 0
                    #     else:
                    #         if action == 0:
                    #             # Bring. You cannot bring the jacket when you are outside.
                    #             p = 0
                    #         elif action == 1:
                    #             # Leave. You cannot leave the jacket when you are outside.
                    #             p = 0
                    #         elif action == 2:
                    #             # Put on  or keep on. That means in the next state you will have it and wear it, but only when you already have it, or p is 0.
                    #             if from_state_has_jacket_true == 0 or to_state_has_jacket_true == 0 or to_state_wearing_jacket_true == 0:
                    #                 p = 0
                    #         elif action == 3:
                    #             # Take off or keep off. That means that in the next state you will have it, but not wear it, but only  if you already had it with you.
                    #             if to_state_wearing_jacket_true == 1:
                    #                 p = 0

                    # # Suppose that the "true" reward is when you are feeeling just right. We do not use this reward in IRL, we learn it. In RL you would identify it using preference illicitation and calculate optimal behavior.
                    # r = to_state_feeling_justright

                    li.append((transition_proba[from_state,action,to_state], to_state, 0, False))

        super(Environment, self).__init__(nS, nA, P, isd)        