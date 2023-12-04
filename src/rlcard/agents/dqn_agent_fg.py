import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple
from collections import deque

from copy import deepcopy



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

NUM_BETTING_OPTIONS = 5



Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


from ..envs.nolimitholdem import NolimitholdemEnv

def flatten_state(state):
    card_feature, action_feature = state
    flattened_card_feature = card_feature.flatten()
    flattened_action_feature = action_feature.flatten()
    flattened_state = np.concatenate((flattened_card_feature, flattened_action_feature))
    return flattened_state


def process_state(state):
    '''
    transform the state from rlcard format to desired format
    '''
    raw_obs = state['raw_obs']
    hand = raw_obs['hand'] # ex: ['S10', 'D10']
    public_cards = raw_obs['public_cards']
    all_chips = raw_obs['all_chips']
    my_chips = raw_obs['my_chips'] 
    legal_actions = raw_obs['legal_actions']
    stakes = raw_obs['stakes']
    current_player = raw_obs['current_player']
    pot = raw_obs['pot']
    stage = raw_obs['stage']
    raw_legal_actions = state['raw_legal_actions']
    action_record = state['action_record'] # list of actions taken so far (player_id, action)

    # card feature
    card_feature = np.zeros((6, 4, 13))
    # hole cards
    '''
    card_feature[0] = hole cards
    card_feature[1] = flop cards
    card_feature[2] = turn cards
    card_feature[3] = river cards
    card_feature[4] = all public cards
    card_feature[5] = all hole and public cards
    '''

    card_feature[0] = hand_to_tensor(hand) 

    # public cards
    if len(public_cards) > 0:
        card_feature[1] = hand_to_tensor(public_cards[:3])
    if len(public_cards) > 3:
        card_feature[2] = hand_to_tensor([public_cards[3]]) 
    if len(public_cards) > 4:
        card_feature[3] = hand_to_tensor([public_cards[4]]) 
    
    # all public cards
    if len(public_cards) > 0:
        card_feature[4] = hand_to_tensor(public_cards)
    card_feature[5] = card_feature[0] + card_feature[4]


    # action feature
    action_feature = np.zeros((24, 3, NUM_BETTING_OPTIONS))

     

    for i, action in enumerate(action_record):
        # print(action)
        # print(state)
        player_id, action_enum, stage_legal_actions = action
        action_tensor = np.zeros((3, NUM_BETTING_OPTIONS))
        action_id = action_enum.value

        
        """
        action_tensor[2] = 1 if action is legal, 0 otherwise
        action_tensor[0] = first player's action (AI)
        action_tensor[1] = second player's action
        """
        action_tensor[player_id, action_id] = 1
        for legal_action in stage_legal_actions:
            legal_action = legal_action.value
            action_tensor[2, legal_action] = 1
        action_feature[i] = action_tensor

    state_tuple = (card_feature, action_feature)
    return state_tuple




def hand_to_tensor(hand):
    '''
    hand: list of strings, ex: ['SQ', 'D10']
    '''
    tensor = np.zeros((4, 13))
    for card in hand:
        suit = card[0]
        rank = card[1:]
        suit_idx = {'S': 0, 'H': 1, 'D': 2, 'C': 3}[suit]
        # convert rank from string to int, if J, Q, K, A, convert to 11, 12, 13, 14
        if rank == 'T':
            rank = 10
        elif rank == 'J':
            rank = 11
        elif rank == 'Q':
            rank = 12
        elif rank == 'K':
            rank = 13
        elif rank == 'A':
            rank = 1
        else:
            rank = int(rank)
        tensor[suit_idx, rank-1] = 1
    return tensor

def process_ts(ts):
    '''
    transform the transition from rlcard format to desired format
            (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done)
        self.total_t += 1

    state, next state -- process 

    '''
    (state, action, reward, next_state, done) = tuple(ts)
    legal_actions = list(next_state['legal_actions'].keys())
    state = process_state(state)
    next_state = process_state(next_state)
    done = 1 if done else 0
    return Transition(state, action, reward, next_state, done, legal_actions)

class NNModel(nn.Module):

    def __init__(self, card_tensor_input_dim, action_tensor_input_dim, action_dim):
        super(NNModel, self).__init__()

        self.card_tensor_input_dim = card_tensor_input_dim
        self.action_tensor_input_dim = action_tensor_input_dim
        self.action_dim = action_dim

        self.card_conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.card_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.card_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)


        self.card_flattened_size = 64 * 4 * 13 
        self.action_flattened_size = 24 * 3 * NUM_BETTING_OPTIONS

        # input shape: [batch_size, sequence_length, features]
        self.action_lstm = nn.LSTM(input_size=15, hidden_size=128, num_layers=2, batch_first=True)

        self.action_fc1 = nn.Linear(self.action_tensor_input_dim, 128)
        self.action_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(self.card_flattened_size + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, card_tensor, action_tensor):
        # card tensor shape: [batch size, card channels, suits, ranks] -- [batch_size, 6, 4, 13]
        # Action tensor shape: [batch_size, sequence_length, features] -- [batch_size, 24, features]

        x = F.relu(self.card_conv1(card_tensor))
        x = F.relu(self.card_conv2(x))
        x = F.relu(self.card_conv3(x))
        card_features = x.view(x.size(0), -1) # [batch.size, 3328]
        # action_tensor = action_tensor.view(action_tensor.shape[0], action_tensor.shape[1], -1)


        # output, (hn, cn) = self.action_lstm(action_tensor)

        # action_features = hn[-1].view(hn[-1].size(0), -1) 

        action_tensor = action_tensor.view(action_tensor.shape[0], -1)
        action_features = F.relu(self.action_fc1(action_tensor))
        action_features = F.relu(self.action_fc2(action_features))



        combined = torch.cat((card_features, action_features), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        q_values = self.fc4(combined)

        return q_values

class DQNAgent:
    """
    Required functions:
    feed(ts) - feed transition into memory
    eval_step(state) - return action # not training
    step(state) - return action  # training
    """
    def __init__(self,
                # env : NolimitholdemEnv,
                 discount_factor=0.95,
                 epsilon_greedy=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.01,
                 max_memory_size=20000,
                 batch_size=1000,
                 train_every=1,
                 save_path=None, 
                 save_every=float('inf'),
                 device=None,
                 update_target_every=500):


        self.device = device
        ##### NN Input Configuration #####
        card_channels = 6 # 2 hole, 3 flop, 1 turn, 1 river, all public cards, and all hole and public cards
        card_channel_shape = (4, 13) # (4 suits, 13 ranks)
        card_feature_shape = (card_channels, *card_channel_shape)

        num_rounds = 4 # pre-flop, flop, turn, river
        max_actions_per_round = 6 # at most six sequential actions in each of the four rounds. (usually 2-3)

        action_channels = num_rounds * max_actions_per_round # 4 rounds * 6 actions

        action_channel_shape = (3, NUM_BETTING_OPTIONS) # (4 dimensions, 6 actions)

        action_feature_shape = (action_channels, *action_channel_shape)        

        self.state_shape = (card_feature_shape, action_feature_shape) 

        self.card_tensor_input_dim = np.prod(card_feature_shape)
        self.action_tensor_input_dim = np.prod(action_feature_shape)
        # ((6, 4, 13), (24, 4, 9))

        self.input_dimension = np.prod(card_feature_shape) + np.prod(action_feature_shape)
        
        ### NN Output Configuration #####

        # 6 actions (fold, check, call, raise half pot, raise pot, all in)
        self.action_shape = (6,) 
        self.output_dimension = np.prod(self.action_shape)


        self.use_raw = False # this is for the env
        # self.env = env
        self.discount_factor = discount_factor
        self.state_size = self.input_dimension 
        self.action_size = self.output_dimension
        self.memory = deque(maxlen=max_memory_size)

        self.gamma = discount_factor    # discount rate
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self._build_nn_model()

        self.train_every = train_every
        self.total_t = 0
        self.train_t = 0
        self.save_path = save_path
        self.save_every = save_every

        self.batch_size = batch_size # number of samples to train on

        self.update_target_every = update_target_every



    def _build_nn_model(self):
        self.model = NNModel(self.card_tensor_input_dim, self.action_tensor_input_dim, self.action_size).to(self.device)
        self.target_model = NNModel(self.card_tensor_input_dim, self.action_tensor_input_dim, self.action_size).to(self.device)
       
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
    

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        self.memory.append(process_ts(ts))
        self.total_t += 1
        tmp = self.total_t - self.batch_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()


    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''


        legal_actions = list(state['legal_actions'].keys())
        if np.random.rand() <= self.epsilon: # exploration
            return np.random.choice(legal_actions)


        q_values = self.predict(state)
        masked_q_values = -np.inf * np.ones(NUM_BETTING_OPTIONS, dtype=float)
        masked_q_values[legal_actions] = q_values[legal_actions]
        action = np.argmax(masked_q_values).item()

        return action

    def predict(self, state):
        """predict the q values, mask illegal actions with -inf"""
        card_state, action_state = process_state(state)
        with torch.no_grad():
            card_tensor = torch.tensor(card_state, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action_state, dtype=torch.float32, device=self.device)
            card_tensor = card_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)


            q_values = self.model(card_tensor, action_tensor)[-1].cpu().numpy()
            
            masked_q_values = -np.inf * np.ones(NUM_BETTING_OPTIONS, dtype=float)
            legal_actions = list(state['legal_actions'].keys())
            masked_q_values[legal_actions] = q_values[legal_actions]

            return masked_q_values

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        q_values = self.predict(state)
        action = np.argmax(q_values).item()
        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info

    def sample_from_memory(self):
        ''' Randomly sample a batch of transitions from the replay buffer for training

        Returns:
            batch (list): a list of transition tuples
        '''
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def train(self):    
        '''
        Train the agent by sampling experiences from the memory buffer.
        '''
        # Sample a batch of transitions from the memory
        batch = self.sample_from_memory()
        states, actions, rewards, next_states, dones, _ = zip(*batch)


        card_tensors = [torch.tensor(state[0], dtype=torch.float32, device=self.device) for state in states]
        action_tensors = [torch.tensor(state[1], dtype=torch.float32, device=self.device) for state in states]
        card_tensors = torch.stack(card_tensors)
        action_tensors = torch.stack(action_tensors)

        next_card_tensors = [torch.tensor(state[0], dtype=torch.float32, device=self.device) for state in next_states]
        next_action_tensors = [torch.tensor(state[1], dtype=torch.float32, device=self.device) for state in next_states]
        next_card_tensors = torch.stack(next_card_tensors)
        next_action_tensors = torch.stack(next_action_tensors)
        

        # actions = torch.tensor(np.array(actions), dtype=torch.long) 
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        predicted_q_values = self.model(card_tensors, action_tensors)
        next_q_values = self.target_model(next_card_tensors, next_action_tensors)
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones).unsqueeze(1))
        loss = self.loss_fn(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()


        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_t += 1
        # update target model
        if self.train_t % self.update_target_every == 0:
            self.target_model = deepcopy(self.model).to(self.device)