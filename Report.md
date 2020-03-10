[//]: # (Image References)

[graph0]: images/graph_0.png "Performance 0"
[graph1]: images/graph_1.png "Performance 1"
[ddpg_image1]: images/ddpg_def.png "DDPG"
[maddpg_image1]: images/MADDPG.png "MADDPG"

# Report of the Project 3 - Udacity DRL: Collaboration and Competition

The following is the report corresponding to the project 3 for the Udacity DRL Nanodegree. I am going to divide it into three main parts:

- Learning Algorithm
- Plot of Rewards
- Ideas for Future Work
  
## 1. Learning Algorithm



The algorithm used in this project to solve the [environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) is __Multi-Agent DDPG__. For a deeper understanding of this algorithm please read the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).

### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

**Deep Deterministic Policy Gradient** or [DDPG](https://arxiv.org/pdf/1509.02971.pdf) is an off policy algorithm that uses the concept of target networks, it belongs to the **Actor-Critic Methods**. Here we use two deep neural networks, one for the _actor_ and the other for the _critic_. The _actor_ is used to approximate the maximizer over the Q-values of the next state, it approximates a deterministic policy that predicts the believed best action. The _critic_  learns to evaluate the optimal action-value function by using the actors' best-believed action.

![DDPG][ddpg_image1]

However, this algorithm, as well as many others used in Deep Reinforcement Learning, are designed to perform in environments where just one agent is considered to act; this kind of environment is called _stationary_. The problem comes when we want to extend it to _Non-Stationary Environments_ where the environment as seen from the perspective of a single agent changes dynamically and so the conditions of which leads the agent to converge is no longer guaranteed.

But, why do we want to solve _Non-Stationary Environments?_. The answer is because these environments are closely related to multi-agent systems. We live in a Multi-Agent World; we have for example the stock market, where each person who is trading can be considered as an agent and the trading profit maximization process can be modeled as a multi-agent problem.

There are different kinds of interactions going on between agents in these environments:

- Coordination
- Competition
- Negotiation
- Communication
- Prediction
  
However, for practical purposes we reduce to three the number of scenarios:

1. Cooperative
2. Competitive
3. Mixed

Multi-Agent Deep Deterministic Policy Gradient or **MADDPG** can be seen as a generalization of the **DDPG** algorithm. 

What we do is to add a critic, a target-critic, an actor and a target-actor for each agent present is the env. This algorithm uses a centralized training and decentralized execution algorithm that can be used to solve any of the environments described above.

During training, the critic for each agent uses extra information like states observed and actions taken by all the other regions. On the other hand, each actor has access to only its agents' observation and actions.

During execution time, only the actors are present, and hence, on observations and actions are used. Learning critic for each agent allows us to use a different reward structure for each. Hence, the algorithm can be used in all, **cooperative**, **competitive** and **mixed** scenarios.

![MADDPG][maddpg_image1]

### Agent Architecture

For this project I used the same architecture for all of the agents involved (two in this case), that is the architecture for the critic of agent 1 is the same that the one for the critic of agent 2. 

#### Actor and Target-Actor Architecture

* The architecture of the model is as follows:
    - __Input Layer__
        -   24 neurons, the same size of the local observations of the agents.
    - __Hidden Layers__
        -   256 neurons with __relu__ as the activation function
        -   128 neurons with __relu__ as the activation function
    - __Output Layer__
        -   2 neurons with __tanh__ as the activation function, the same size as the action vector of the environment for each agent.
    -   __Optimzer__
        -   Adam
    -   __Learning Rate__
        -   1e-4
   
#### Critic and Target-Critic Architecture

* The architecture of the model is as follows:
    - __Input Layer__
        -   48 neurons, the same size as the full state of the environment, that is the concatenation of both local observations of 24 for each agent.
    - __Hidden Layers__
        -   260 neurons with __relu__ as the activation function, 256 neurons + 4 extra neurons corresponfing to the action vectors.
        -   128 neurons with __relu__ as the activation function
    - __Output Layer__
        -   1 neurons with __relu__ as the activation function, the value of the action dictated by the actor.
    -   __Optimzer__
        -   Adam
    -   __Learning Rate__
        -   1e-3

### Agent Hyperparameters

As well as **Q-Learning** we need to set the __discount factor &gamma;__ that is the one that defines how much is taking into account the future rewards, so that if it is equal to 0 the agent only considers the immediate reward and the __experience replay__ which is nothing more than a collection of experiences __(S, A, R, S')__. These tuples are gradually added to the buffer of the agent.

* The value of the __discount factor &gamma;__ for this project was set to __0.95__
* The value of the __experience replay__ for this project was set to __1024__ with a __buffer size__ of __1e6__ 

Remember that some of the advantages of the use of experience replay are:
 - Breaking harmful correlations. 
 - To learn more from individual tuples multiple times.
 - Recall of rare occurrences, and 
 - In general, make better use of the experience.

Finally, the value of TAU **(&tau;)** for the soft update is equal to __2e-2__.

## 2. Plot of Rewards

The agent run 2000 episodes for training, however, it was able to solve the environment after __1700 episodes__ accomplishing the goal of, at least, 100 consecutive episodes with a reward of +0.5. The maximal average score reached for the agent was __+0.8971__. However, after solving the environment the agent started to decay until getting an average reward below +0.2.

Below you can see a capture of the graph got in training. 

![Training Graph][graph1]

## 3. Ideas for Future Work

Even when the environment was considered solved using a MADDPG approach, it started to decay and we may be able to make it converge again after training it during more episodes or we may not. For this, many possible changes can be done to improve the performance of the agents.

### Try other neural networks architectures for the critic and the actor
The simplest improvement that we can try is varying the number of hidden layers of the models and so the number of neurons of them or to try with different activations functions just like any other neural network.

### Play around with the hyperparameter
We may achieve better performance with a different value for the learning rates, the value of __&gamma;__ or __&tau;__ for the soft update.

### Try other MARL methods

We may try another algorithm based on the Multi-Agent Reinforcement Learning (**MARL**) Approach. A concrete idea is to use the algorithm of the well-known paper [AlphaZero](https://arxiv.org/abs/1712.01815) that was able to beat LeeSedol, a GO professional player

The best part of the **AlphaZero** algorithm is its simplicity: it consists of a **Monte Carlo Tree Search**, guided by a deep neural network. This is analogous to the way humans think about board games, where professional players employ hard calculations guides with intuitions.