''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse
import random
import numpy as np

import torch

from rlcard.agents.random_agent import RandomAgent

from rlcard.utils.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    plot_curve,
)

from rlcard.utils.logger import Logger

import rlcard

from rlcard.agents.dqn_agent_fg import NUM_BETTING_OPTIONS, DQNAgent, hand_to_tensor

import matplotlib.pyplot as plt
import time

import itertools

def evaluate_preflop_range(args):
    # Initialization
    # Environment setup
    # env = rlcard.make('no-limit-holdem-fg', config={'seed': args.seed})
    agent = torch.load(args.model_path, map_location=torch.device('cpu'))
    # agents = [agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(1, env.num_players)]
    # env.set_agents(agents)

    # Initialize pre-flop chart
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['S', 'H', 'D', 'C']
    preflop_chart = np.zeros((13, 13), dtype=str)

    # Iterate over all possible starting hands
    for i, rank1 in enumerate(ranks):
        for j, rank2 in enumerate(ranks[i:]):
            for suit1, suit2 in itertools.product(suits, repeat=2):
                if i == j and suit1 >= suit2:
                    # Skip duplicate combinations for pairs
                    continue

                # Construct the hand
                hand = [suit1 + rank1, suit2 + rank2 if i != j else suit2 + rank2]
                card_feature, action_feature = prepare_features(hand)

                # Predict action for this hand
                pred = agent.predict_raw(card_feature, action_feature)
                chosen_action = np.argmax(pred)

                # Update preflop chart
                chart_symbol = "P" if chosen_action != 0 else "F"
                preflop_chart[i, j if i != j else i] = chart_symbol

    # Output the chart
    print_preflop_chart(preflop_chart)

def prepare_features(player_hand):
    # Convert hand to tensor or other required format
    # ... [Implementation depends on the agent's requirements]
    card_feature = np.zeros((6,4,13))

    card_feature[0] = hand_to_tensor(player_hand)

    card_feature[5] = hand_to_tensor(player_hand)

    action_feature = np.zeros((24, 3, NUM_BETTING_OPTIONS))

    action_tensor = np.zeros((3, NUM_BETTING_OPTIONS))

    action_tensor[2] = np.ones(NUM_BETTING_OPTIONS)

    action_feature[0] = action_tensor

    return card_feature, action_feature



def print_preflop_chart(chart):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    print("   " + "  ".join(ranks))
    for i, row in enumerate(chart):
        print(ranks[i] + " " + "  ".join(row))

def evaluate_ol(args):


    pass
    # Check whether gpu is available
    device = get_device()
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        'no-limit-holdem-fg',
        config={
            'seed': args.seed,
        }
    )

    agent = torch.load(args.model_path)
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))

    env.set_agents(agents)
    

    reward = tournament(env, args.num_eval_games)[0]
    print(reward)


    # determine pre-flop chart from the model
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    suits = ['S', 'H', 'D', 'C']

    deck = []
    for rank in ranks:
        for suit in suits:
            deck.append(suit + rank)
    
    # shuffle the deck
    random.shuffle(deck)

        
    # deal two random cards to the player
    player_hand = []
    for _ in range(2):
        card = deck.pop()
        player_hand.append(card)



    pred = agent.predict_raw(prepare_features(player_hand))
    print(player_hand)
    print(pred)
    chosen_action = np.argmax(pred)
    if chosen_action == 0:
        print("fold")
    else: 
        print("play")
    

    



   



if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1000,
    )
    args = parser.parse_args()
    evaluate_preflop_range(args)
