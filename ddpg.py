# individual network settings for each actor + critic pair
# see networkforall for details

#from networkforall import Network
from model import ActorNetwork, CriticNetwork

#from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#GAMMA = 0.99            # discount factor
#TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1.e-5       # L2 weight decay

def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPGAgent:
    #def __init__(self, in_actor=14, hidden_in_actor=16, hidden_out_actor=8, out_actor=2, 
                #in_critic=20, hidden_in_critic=32, hidden_out_critic=16, 
                #lr_actor=1.0e-2, lr_critic=1.0e-2):
    def __init__(self, state_size, obs_size, action_size, num_agents):
        super(DDPGAgent, self).__init__()

        #self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        #self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        #self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        #self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        
        self.actor  = ActorNetwork(obs_size, action_size).to(device)
        self.critic = CriticNetwork(state_size, action_size*num_agents).to(device)
        self.target_actor = ActorNetwork(obs_size, action_size).to(device)
        self.target_critic = CriticNetwork(state_size, action_size*num_agents).to(device)

        #self.noise = OUNoise(out_actor, scale=1.0 )
        self.noise = OUNoise(action_size, scale=1.0 )

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    def act(self, obs, noise=0.0):
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(device)
        #self.actor.eval()
        action = self.actor(obs)
        action += noise*self.noise.noise()
        #self.actor.train()
        #return action.cpu().data.numpy()
        return action

    def target_act(self, obs, noise=0.0):
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float().to(device)
        #obs = obs.to(device)
        #self.target_actor.eval()
        #action = self.target_actor(obs) + noise*self.noise.noise()
        action = self.target_actor(obs)
        action += noise*self.noise.noise()
        #self.target_actor.train()
        #return action.cpu().data.numpy()
        return action


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float().to(device)
