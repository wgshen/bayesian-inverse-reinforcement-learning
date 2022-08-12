# bayesian-inverse-reinforcement-learning

## Running steps (MS_model as an example):
* Replace inputs in pre-hoc-inputs and post-hoc-inputs as your own inputs
* Modify pre-hoc-controls.inp with your own settings
* run pre-hoc.py to get MCMC samples and corresponding KL divergences of pre-hoc design
* run pre-hoc-design.py to obtain the sample size design
* Modify post-hoc-controls.inp with your own settings
* run post-hoc.py to get MCMC samples and corresponding KL divergences of post-hoc assessment
* run post-hoc-check.py to asses if the collected sample size is sufficient


The format and size of npy files (pre-hoc-inputs and post-hoc-inputs) are:
action_feature.npy (n_action, n_action_feature), the one hot encoding of action features.
action_feature_name.npy (n_action_feature), the name of action features
behavior_instances.npy (n_instance, n_horizon, 2), state-action pairs
initial_state_dist.npy (n_state), the initial probability of each state
state_feature.npy (n_state, n_state_feature), the one hot encoding of state features
state_feature_name.npy (n_state_feature), the name of state features
transition_proba.npy (n_possible_transition, 4), the transition probabilities, each entry is a 4 element tuple (state_old, action, state_new, prob)
valid_init_state.npy (n_valid_state), the valid initial states, each entry is an index of valid state.
