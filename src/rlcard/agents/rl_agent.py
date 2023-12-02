
import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque
from copy import deepcopy

from ..utils.utils import remove_illegal

"""
Game State Representation: 

Card information + Action information encoding 


Card Feature Representation:
- Tensor: 6 channels, (2 hole, 3 flop, 1 turn, 1 river). 
- Each channel is a 4x13 sparse binary matrix, with a 1 representing the corresponding card. 

Action tensor:
- 6 sequential actions in each of the 4 betting rounds, we have 24 channels.
- each channel is 4xn_b where n_b the number of betting options. The 4 dimensions correspond to: (adjust n_b based on desired complexity).
    - first player’s action
    - second player’s action
    - sum of two player’s action
    - and legal actions.
        
"""

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class DQN(nn.Module):

    def __init__(self, input_dimension, output_dimension, hidden_layers_sizes): 
        super(DQN, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers_sizes = hidden_layers_sizes
        self.fc = [nn.Flatten()]

        self.layer_dimensions = [self.input_dimension] + self.hidden_layers_sizes + [self.output_dimension]

        for i in range(len(self.layer_dimensions) - 1):
            self.fc.append(nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i+1]))
            self.fc.append(nn.ReLU())
        
        self.fc.append(nn.Linear(self.layer_dimensions[-1], self.output_dimension))

        self.fc_layers = nn.Sequential(*self.fc)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        return self.fc_layers(state)


class HoldemModel(object):

    def __init__(self, num_actions, learning_rate, state_shape, hidden_layers_sizes=[364,364], device=None):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.hidden_layers_sizes = hidden_layers_sizes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qnet = DQN(self.state_shape, self.num_actions, self.hidden_layers_sizes).to(self.device)
        self.qnet.eval()

        # initialize the weights
        for layer in self.qnet.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.mse_loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)


    def predict_nograd(self, s):
        """
        Predict the q-values for all legal actions

        s (np.ndarray): (batch_size, state_shape)
        """
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_values = self.main_nn(s).cpu().numpy()
        return q_values
    
    def update(self, s, a, y):
        """
        Args:
            s (np.ndarray): (batch_size, state_shape) -- state representation
            a (np.ndarray): (batch_size, ) -- integer sampled action indices
            y (np.ndarray): (batch_size, ) -- target action values 
        
        Returns:
            loss (float): loss value
        """
        self.optimizer.zero_grad()
        self.qnet.train()
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss

        
    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint
        '''
        return {
            'qnet': self.qnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'device': self.device
        }
        
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint
        '''
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
            device=checkpoint['device']
        )
        
        estimator.qnet.load_state_dict(checkpoint['qnet'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer'])
        return estimator



  
class RLAgent(object):


    def __init__(self, 
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions=2,
                #  state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf')):
        
        self.use_raw = False
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every
        self.mlp_layers = mlp_layers
        self.learning_rate = learning_rate
        self.device = device
        self.save_path = save_path
        self.save_every = save_every

        self.epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        """
        
        State Shape = Card Feature + Action Tensor


        Card Feature: 
        Channel: cards (2 hole, 3 flop, 1 turn, 1 river, all public cards, and all hole and public cards).
        6 channels, each channel is 4x13 binary matrix, with a 1 representing the corresponding card.

        Action Tensor: 4 rounds * 6 actions = 24 channels. 

        rounds: pre-flop, flop, turn, river
        actions: at most six sequential actions in each of the four rounds. (usually 2-3)
        - ex: p1 check, p2 bet, p1 raise, p2 call

        Each channel contains: 4 x n_b where n_b is the number of betting options. The 4 dimensions correspond to:
        - first player’s action
        - second player’s action
        - sum of two player’s action
        - and legal actions (1 if action is legal, 0 otherwise)

        """


        ##### NN Input Configuration #####
        card_channels = 6 # 2 hole, 3 flop, 1 turn, 1 river, all public cards, and all hole and public cards
        card_channel_shape = (4, 13) # (4 suits, 13 ranks)

        card_feature_shape = (card_channels, *card_channel_shape)

        num_betting_options = 9

        num_rounds = 4 # pre-flop, flop, turn, river
        max_actions_per_round = 6 # at most six sequential actions in each of the four rounds. (usually 2-3)

        action_channels = num_rounds * max_actions_per_round # 4 rounds * 6 actions
        # rounds: pre-flop, flop, turn, river

        action_channel_shape = (4, num_betting_options) # (4 dimensions, 6 actions)
        # actions: at most six sequential actions in each of the four rounds. (usually 2-3)
        # 4 dimensions correspond to:
        # - first player’s action
        # - second player’s action
        # - sum of two player’s action
        # - and legal actions (1 if action is legal, 0 otherwise)


        action_feature_shape = (action_channels, *action_channel_shape)        


        state_shape = (card_feature_shape, action_feature_shape)
        print(state_shape)



        self.state_shape = (card_feature_shape, action_feature_shape) 

        # ((6, 4, 13), (24, 4, 9))

        self.input_dimension = np.prod(card_feature_shape) + np.prod(action_feature_shape)
        print(self.input_dimension)
        
        ### NN Output Configuration #####

        # 6 actions (fold, check, call, raise half pot, raise pot, all in)
        self.action_shape = (6,) 
        self.output_dimension = np.prod(self.action_shape)

        self.main_nn = HoldemModel(self.num_actions, self.learning_rate, self.input_dimension, self.hidden_layers_sizes, self.device)
        self.target_nn = HoldemModel(self.num_actions, self.learning_rate, self.input_dimension, self.hidden_layers_sizes, self.device)

        self.memory = Memory(size=self.replay_memory_size, batch_size=self.batch_size)

    
    def convert_ts(self, ts):
        pass

    def feed(self, rlcard_ts):
        """
        Feed trajectories into agent memory, and train the agent

        ts: (state, action, reward, next_state, done)
        """
        ts = self.convert_ts(rlcard_ts) 

        (state, action, reward, next_state, done) = ts

        self.memory.add(ts)



        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()
        
    def step(self, state):
        pass


    def convert_state(self, rlcard_state):
        """
        Convert the state into the input format for the NN

        rlcard_state: (dict): the raw state from rlcard

        returns state in format (card_feature, action_feature)
        """
        pass


    def predict(self, rlcard_state):
        """
        predict the masked q-values, given the current state

        masking makes the q-values of illegal actions -inf
        """
        q_values = self.main_nn.predict_nograd(self.convert_state(rlcard_state))
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(rlcard_state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]
        return masked_q_values


    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        q_values_next = self.main_nn.predict_nograd(next_state_batch)
        legal_actions = []

        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)

        q_values_next_target = self.target_nn.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        state_batch = np.array(state_batch)

        loss = self.main_nn.update(state_batch, action_batch, target_batch)
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_nn = deepcopy(self.main_nn)
        self.train_t += 1

        return loss


class Memory(object):
    """Experience replay buffer that samples uniformly.

    From: https://colab.research.google.com/drive/1w5xFX2wJvtuVbcrDHny7YPcTdGqMOqMu#sandboxMode=true&scrollTo=N7OT489fkEXO


    Used to give the agent a chance to learn from past experiences.

    """
    def __init__(self, size=100, batch_size=32):
        self.memory_size = size
        self.memory = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, transition): 
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        """
        Returns: a list of transition tuples
        """
        samples = random.sample(self.memory, self.batch_size)
        return samples 

    def checkpoint_attributes(self):
        return {
            'memory_size' : self.memory_size, 
            'batch_size' : self.batch_size,
            'memory' : self.memory
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        memory = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        memory.memory = checkpoint['memory']
        return memory


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


agent = RLAgent()