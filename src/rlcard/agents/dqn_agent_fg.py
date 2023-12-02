import numpy as np
import torch
import random
from collections import namedtuple
from collections import deque


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