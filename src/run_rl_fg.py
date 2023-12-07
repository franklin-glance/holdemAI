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

import matplotlib.pyplot as plt
import time


print("running version 0.0.1")

def train(args):
    # Check whether gpu is available
    device = get_device()
    set_seed(args.seed)
    print("training with args: ", args)

    # Make the environment with seed
    env = rlcard.make(
        'no-limit-holdem-fg',
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
    # set args.log_dir to custom based on parameters
    log_dir = ''
    for key, value in vars(args).items():
        if key == 'log_dir':
            continue
        # make key into abbreviation
        key = key.split('_')
        key = ''.join([word[0] for word in key])

        log_dir += key + '_' + str(value) + '_'
    log_dir = log_dir[:-1]
    log_dir += '/'
    args.log_dir = os.path.join(args.log_dir, log_dir)
    rewards = []
    eval_every_time = time.time()

    total_num_evaluations = args.num_episodes // args.evaluate_every
    with Logger(args.log_dir) as logger:
        logger.log(f'args: {args}')
        for episode in range(args.num_episodes):
            episode_start_time = time.time()
            # Generate data from the environment

            data_generation_start_time = time.time()
            trajectories, payoffs = env.run(is_training=True)
            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            data_generation_duration_seconds = time.time() - data_generation_start_time



            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            feeding_start_time = time.time()
            feed_times = []
            for ts in trajectories[0]:
                feed_times.append(time.time())
                agent.feed(ts)
                feed_times[-1] = time.time() - feed_times[-1]

            feeding_duration_seconds = time.time() - feeding_start_time


            episode_duration_seconds = time.time() - episode_start_time
            # Evaluate the performance. Play with random agents.



            eval_times = []
            if episode % args.evaluate_every == 0 and episode >= args.batch_size:
                duration = time.time() - eval_every_time
                eval_every_time = time.time()

                print(f'episode: {episode}')
                print(f'time since last evaluation: {duration}')
                print(f'average episode duration: {duration/args.evaluate_every}')
                expected_time_until_completion = (args.num_episodes - episode) * duration / args.evaluate_every
                # add time from evaluation
                print(f'expected time until completion: {expected_time_until_completion}')
                if eval_times:
                    additional_time_from_future_evals = sum(eval_times) / len(eval_times) * ((args.num_episodes - episode) // args.evaluate_every)
                    print(f'expected time until completion (including eval time): {expected_time_until_completion + additional_time_from_future_evals}')


                print("starting evaluation")
                eval_start_time = time.time()
                reward = tournament(
                    env,
                    args.num_eval_games,
                )[0]
                logger.log_performance(
                    episode,
                    reward
                )
                rewards.append(reward)
                print(f'evaluation took {time.time() - eval_start_time} seconds')
                eval_times.append(time.time() - eval_start_time)
                logger.log(f'epsilon: {agent.epsilon}')
                logger.log(f'agent_buffer_size: {len(agent.memory)}')
                # overwrite the file with all losses
                with open(os.path.join(args.log_dir, 'losses.csv'), 'w') as f:
                    f.write('\n'.join([str(loss) for loss in agent.losses]))

                with open(os.path.join(args.log_dir, 'rewards.csv'), 'a') as f:
                    f.write(str(reward) + '\n')
                
                # save plot of losses
                plt.plot(agent.losses)
                plt.title('Losses')
                plt.savefig(os.path.join(args.log_dir, 'losses.png'))
                # clear plot
                plt.clf()

                # plot rewards
                plt.plot(rewards)
                # add title with --num_eval_games
                plt.title(f'Average reward over {args.num_eval_games} games')
                plt.savefig(os.path.join(args.log_dir, 'rewards.png'))
                # clear plot
                plt.clf()

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    plt.plot(agent.losses)
    plt.title('Losses')
    plt.savefig(os.path.join(args.log_dir, 'losses_final.png'))
    plt.clf()


    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'fg_dqn')

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
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
        default=1000000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/',
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
        default=0.95)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00005)
    parser.add_argument(
        "--max_memory_size",
        type=int,
        default=2000000)
    parser.add_argument(
        "--train_every",
        type=int,
        default=32)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024*10)
    parser.add_argument(
        "--update_target_every",
        type=int,
        default=4)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)