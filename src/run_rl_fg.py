''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

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

from rlcard.agents.dqn_agent_fg import DQNAgent


def train(args):
    # Check whether gpu is available
    device = get_device()
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    agent = DQNAgent(
                # env=env,
                discount_factor=args.discount_factor,
                epsilon_greedy=args.epsilon_greedy,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                learning_rate=args.learning_rate,
                max_memory_size=args.max_memory_size,
                batch_size=args.batch_size,
                train_every=args.train_every,
                save_path=args.log_dir,
                save_every=args.save_every,
                device=device,
                update_target_every=args.update_target_every,
            )

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem-fg',
        choices=[
            'limit-holdem',
            'no-limit-holdem',
            'no-limit-holdem-fg'
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/no_limit_holdem_fg_dqn_result/',
    )
    
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=-1)
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.95)
    parser.add_argument(
        "--epsilon_greedy",
        type=float,
        default=1.0)

    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=0.01)
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.995)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01)
    parser.add_argument(
        "--max_memory_size",
        type=int,
        default=20000)
    parser.add_argument(
        "--train_every",
        type=int,
        default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128)
    parser.add_argument(
        "--update_target_every",
        type=int,
        default=100)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)