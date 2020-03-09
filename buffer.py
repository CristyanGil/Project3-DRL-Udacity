#from collections import deque
import random
import numpy as np
#from utilities import transpose_list
from collections import namedtuple, deque
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    #def __init__(self, action_size, buffer_size, batch_size, seed):
    def __init__(self, buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, seed=0):        
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["obs", "state", "actions", "rewards", "next_obs", "next_state", "dones"])
        self.seed = random.seed(seed)
    
    def push(self, obs, state, actions, rewards, next_obs, next_state, dones):
        """Add a new experience to memory."""
        e = self.experience(obs, state, actions, rewards, next_obs, next_state, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        #obs      - lista 1000, obs[0] - numpy(3,14)
        #obs_full - lista 1000, obs_full[0] - numpy(14,)
        #action   - lista 1000, action[0] - numpy(3,2)
        #reward   - lista 1000, reward[0] - numpy(3,)
        #next_obs  - lista 1000, obs[0] - numpy(3,14)
        #next_obs_full - lista 1000, obs_full[0] - numpy(14,)
        #done - lista 1000, numpy(3,)
        
        #obs_vector      = torch.from_numpy(np.vstack([[e.obs] for e in experiences if e is not None])).float().to(device)
        #states          = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions_vector  = torch.from_numpy(np.vstack([[e.actions] for e in experiences if e is not None])).float().to(device)
        #rewards_vector  = torch.from_numpy(np.vstack([[e.rewards] for e in experiences if e is not None])).float().to(device)
        #next_obs_vector = torch.from_numpy(np.vstack([[e.next_obs] for e in experiences if e is not None])).float().to(device)
        #next_states     = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #dones_vector    = torch.from_numpy(np.vstack([[e.dones] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        #obs_vector      = np.vstack([[e.obs] for e in experiences if e is not None])
        #states          = np.vstack([e.state for e in experiences if e is not None])
        #actions_vector  = np.vstack([[e.actions] for e in experiences if e is not None])
        #rewards_vector  = np.vstack([[e.rewards] for e in experiences if e is not None])
        #next_obs_vector = np.vstack([[e.next_obs] for e in experiences if e is not None])
        #next_states     = np.vstack([e.next_state for e in experiences if e is not None])
        #dones_vector    = np.vstack([[e.dones] for e in experiences if e is not None])
        
        obs_vector      = [ np.array(e.obs) for e in experiences if e is not None]
        states          = [ np.array(e.state) for e in experiences if e is not None]
        actions_vector  = [ np.array(e.actions) for e in experiences if e is not None]
        rewards_vector  = [ np.array(e.rewards) for e in experiences if e is not None]
        next_obs_vector = [ np.array(e.next_obs) for e in experiences if e is not None]
        next_states     = [ np.array(e.next_state) for e in experiences if e is not None]
        dones_vector    = [ np.array(e.dones) for e in experiences if e is not None]
        
        #[ np.array(e.dones) for e in exp if e is not None]

        return (obs_vector, states, actions_vector, rewards_vector, next_obs_vector, next_states, dones_vector)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
