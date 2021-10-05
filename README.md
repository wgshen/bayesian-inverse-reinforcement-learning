# bayesian-inverse-reinforcement-learning

## Running steps (MS_model as an example):
* Replace inputs in pre-hoc-inputs and post-hoc-inputs as your own inputs
* run pre-hoc.py to get MCMC samples and corresponding KL divergences of pre-hoc design
* run pre-hoc-design.py to obtain the sample size design
* run post-hoc.py to get MCMC samples and corresponding KL divergences of post-hoc assessment
* run post-hoc-check.py to asses if the collected sample size is sufficient
