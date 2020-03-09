[//]: # (Image References)

[env]: images/env.png "Environment"

# Project 3 - Udacity DRL: Collaboration and Competition

## Project Details

For this project, I will train two agents to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, that was developed by [Unity]( https://unity3d.com/es).

### Environment Description

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

![Environment][env]

## Getting Started

For this project, you are going to need some basic libraries like _numpy_ and _matplotlib_. For the neural network of the agent, we are going to use PyTorch. In order to be able to interact with the Unity Toolkit we have to install the following version of torch and torchvision.

* __torch=0.4.0__
* __torchvision=0.2.1__

You can see instructions for installation these old libraries directly on the PyTorch [website](https://pytorch.org/get-started/previous-versions/)

### Download the environment

1. You can download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
2. Place the file in the `root/` folder of the project, and unzip (or decompress) the file.


## Instructions

This repository contains the following scripts
- __maddpg.py__
    - This script initializes the ddpg agents needed to solve the environment, from here, you can get the actors and the critics architecture.
    - But the most important thing is that here is the function __update__ that manages the training process for all of the models. 
- __ddpg.py__
    - This script manages the internal variables of the agents individually, for example for each model you have a critic, a target critic, an actor and a target actor, these collections of models for each agent are defined here. 
    - It also has functions that allow the actors to act.  
- __model.py__
    - Here is where the architectures of the models are initialized.
- __buffer.py__
    - It contains the replay memory.
- __utilities.py__
    - As the name indicates, it contains some utilities needed in the training process such as **soft_update** and **transpose_to_tensor**.
- __environment.py__
    - It is a simplified script to interact with the Unity environment. It has two main methods:
        - __reset:__ resets the environment and returns the observations and the state of the env.
        - __execute:__ receives a list of actions corresponding to each agent in the environment and returns the next observations, the next state, the rewards and a bool list indicating if the env is in a terminal state.
- __Tennis_train.ipynb__
    - This script allows you to see step by step the training phase.

Open the __Tennis_train.ipynb__ and execute the cells to see how to interact with the environment until you will train the agent.