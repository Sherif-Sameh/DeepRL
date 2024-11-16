# Summary

This repository currently includes implementations of the majority of the core algorithms in Deep Reinforcement Learning as listed below. All the implementations are in Python and make use of PyTorch for DL models, optimizers and training in general. Each algorithm is independent of the implementation of the others, even when sharing a lot of overlapping ideas. The main structure of each algorithm is as follows:

* **Models**: Includes the implementation of all needed NN models, extending the generic PyTorch nn.Module class, like DQNs, critics, actors for different action spaces and a combined actor-critic model.

* **Train**: Includes implementations of a experience buffer, whether for off or on-policy methods, loss functions, parameter update function and the main training loop. Uses MPI processes to parallelize the training across multiple CPU processes. 

* **Test**: Loads saved agent and environment and runs them inside the OpenAI gym with rendering until manual termination. 

* **Notebook**: Is named after the corresponding method and includes the initialization of the agent, training parameters and environment as well as code for training the agent. Limited to training using a single process as MPI processes did not work inside a Jupyter Notebook. 

# 

The current implementations are based on an older version of Python (3.6.13) and therefore Python libraries. This was to maintain compatibility with the OpenAI SpinningUp repository for comparisons as well as use with the old OpenAI Gym.  

Currently none of the implementations make use of a GPU as they're all made up of small MLP networks. Also, parameter sharing was not implemented for the same reasons.  

# 

Main developments in the future would be to add the possibility of learning through images using a shared CNN backend for both actor and critic. And/or the use of an separately trained Autoencoder for extacting low-dimensional latent features from the images. 

Also, the implementation of some of the methods listed below that complement these core algorithms to improve their robustness, generalization, or sample efficiency are of interest to me. 

Lastly, methods for learning from very sparse rewards or using human feedback as well as enhancing exploration are also of interest for the future. 

#

# Core DRL Algorithms

## Off-Policy

* Deep Q-Learning Network (DQN)
	* Basic DQNs   
	* Double DQNs (DDQN)
	* Dueling DQNs
	* DQNS with Prioritized Experience Replay (PER)

* Deep Deterministic Policy Gradient (DDPG)

* Twin Delayed Deep Deterministic Policy Gradient (TD3)

* Soft Actor-Critic (SAC)

## On-Policy
* Vanilla Policy Gradient (VPG aka REINFORCE)
 
* Advantage Actor-Critic (A2C)*

* Trust Region Policy Gradient (TRPO)

* Proximal Policy Optimization (PPO)

\* Current parallelized VPG implementation with GAE is the same as A2C


# Sparse Rewards & Exploration

* Hindsight Experience Replay (HER)

* To do:
	* Reinforcement Learning with Unsupervised Auxiliary Tasks
	* Curiosity-driven Exploration by Self-supervised Prediction

# Hierarchical RL 


# Imitation Learning & Human Feedback

* To do:
	* DeepMimic: Example-Guided Deep Reinforcement Learning
of Physics-Based Character Skills
	* Deep Reinforcement Learning from Human Preferences

# Physics & Model-Based RL

* To do:
	* Model-Ensemble Trust-Region Policy Optimization
	* Model-Based Reinforcement Learning via Meta-Policy Optimization

# Meta, Adversarial and Transfer Learning

* To do:
	* Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
	* Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
	* Robust Adversarial Reinforcement Learning
	* Policy Distillation 
	* Teacher-student curriculum learning for reinforcement learning
