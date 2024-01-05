import sys
from env import StockMarket
import argparse
import time
import os
import TD3
import numpy as np
from RLUtils import ReplayBuffer
from datetime import datetime

# Specifically for logging ML metrics
import torch
from torch.utils.tensorboard import SummaryWriter

# Used to integrate with tensorboard
tb = None

# Define some fixed variables for experiments
train_seed = 42
eval_seed = 84

# Runs policy for X episodes and returns average reward
def eval_policy(policy, eval_env, seed, render_ui=False, tb=None, eval_count=0, eval_episodes=4):
    avg_reward = 0.
    print("Evaluating Policy...")
    for i in range(eval_episodes):
        obs, done = eval_env.reset(seed=seed + i, has_ui=False, new_dates=False, new_tickers=False), False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    # Log eval results to TB
    if tb:
        tb.add_scalar('Eval Average Reward', avg_reward / eval_episodes, eval_count)

    print("---------------------------------------")
    print(f"Finished! Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def get_random_seed():
    return round(time.time())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                      # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Base")                        # Name of environment (Base, Partial, or Full)
    parser.add_argument("--cash", default=30000)                        # Starting cash for portfolio
    parser.add_argument("--max_trade_perc", default=0.90)               # The maximum amount of remaining cash that can be traded at once.
    parser.add_argument("--seed", default=train_seed, type=int)         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e5, type=int)     # Collect 100k random samples for learning first
    parser.add_argument("--eval_freq", default=10, type=int)            # How often (episodes) we evaluate the model
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max GLOBAL time steps to run environment
    parser.add_argument("--expl_noise", default=0.15, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--policy_noise", default=0.1)                  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=True)                   # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                     # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    home_dir = "Z:/VSCode/DRL_Trader/"

    file_name = f"{args.policy}_{args.env}_{args.seed}_3Asset_test"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(home_dir + "results"):
        os.makedirs(home_dir + "results")

    if args.save_model and not os.path.exists(home_dir + "models"):
        print("Auto-saving model as training progresses...")
        os.makedirs(home_dir + "models")

    # Initialize environment
    lookback_steps = 20
    ti_list = ["SMA", "RSI", "EMA", "STOCH", "MACD", "ADOSC", "OBV", "ROC", "WILLR"]
    ti_args = {"RSI": {"timeperiod": lookback_steps}, "SMA": {"timeperiod": lookback_steps}, "EMA": {"timeperiod": lookback_steps}}

    # Environment for training...
    trn_env = StockMarket(
        cash=args.cash,
        max_trade_perc=None,
        include_ti=True,
        period_months=72,
        num_assets=3,
        fixed_start_date=datetime(2010, 1, 1),
        trade_cost=7.99,
        lookback_steps=lookback_steps,
        indicator_list=ti_list,
        indicator_args=ti_args
    )

    # Prepare environment
    s = train_seed
    obs = trn_env.reset(seed=train_seed)

    # Environment for testing / evaluating
    eval_env = StockMarket(
        cash=args.cash,
        include_ti=True,
        period_months=24,
        num_assets=trn_env.num_assets,
        fixed_start_date=datetime(2016, 1, 1),
        fixed_portfolio=trn_env.assets,
        trade_cost=7.99,
        lookback_steps=lookback_steps,
        indicator_list=ti_list,
        indicator_args=ti_args
    )

    # Handle multi-dimensional spaces and scaling ranges
    state_dim = trn_env.observation_space.shape
    action_dim = trn_env.action_space.shape
    dim_range = list(zip(trn_env.action_space.low, trn_env.action_space.high))
    action_range = [(float(low), float(high)) for low, high in dim_range]

    # Initialize the TD3 DRL Algorithm
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_range": action_range,
        "discount": args.discount,
        "tau": args.tau,
    }
    kwargs["policy_noise"] = [args.policy_noise * high for _, high in action_range]
    kwargs["noise_clip"] = [args.noise_clip * high for _, high in action_range]
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    # Instantiate Replay Buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # If asked to, load weights from previous saved instance of model
    if args.load_model != "":
        print(f"Loading model {file_name} and continuing training...")
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    tb = SummaryWriter(f'logs/{file_name}')  # You can choose any directory for your logs

    # Evaluate untrained policy
    eval_num = 1
    evaluations = [eval_policy(policy, eval_env, eval_seed, False, tb, eval_num)]

    # Declare training vars
    global_t = 0
    t = 0
    episode_num = 0
    episode_reward = 0

    # Run complete training iterations until args.max_timesteps is crossed - at which point finish episode and end
    while (global_t < args.max_timesteps):

        # Select action randomly or according to policy
        action = None
        if global_t < args.start_timesteps:

            action = trn_env.action_space.sample()

        else:

            action = policy.select_action(obs)

            # Used to add noise to action signal - clipped within range
            for i in range(0, len(action)):
                low, high = action_range[i]
                noise = np.random.normal(
                    0,
                    high * args.expl_noise,
                    size=None
                )

                # Log total episode reward to TensorBoard
                tb.add_scalar('Exploration Noise', noise, global_t)

                action[i] += noise
                action[i] = action[i].clip(low, high)

        # Perform action
        obs, reward, done, _ = trn_env.step(action)

        # Continue on with episode - advancing t
        if not done:

            # Store data in replay buffer
            replay_buffer.add(obs, action, obs, reward, done)

            # Train agent after collecting sufficient data
            if global_t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            # Update counters
            t += 1
            episode_reward += reward

        # If done trading simulation
        elif done:

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {global_t + 1} Episode Num: {episode_num + 1} Episode T: {t + 1} Reward: {episode_reward:.3f}")

            # Print action counts
            #for action in list(trn_env.action_counts.keys()):
            #    print(f"- {action} occured {trn_env.action_counts[action]} times...")

            # Log total episode reward to TensorBoard
            tb.add_scalar('Total Episode Reward', episode_reward, episode_num)

            episode_num += 1
            episode_reward = 0
            t = 0

            # Render UI every fifth episode after sufficient training has occured
            obs = trn_env.reset(seed=s + (episode_num * 10), new_dates=False, new_tickers=False)

            # Evaluate every n episodes - rendering every third eval after the training has started
            if (episode_num + 1) % args.eval_freq == 0:
                eval_num += 1
                render_ui = False #True if eval_num % 3 == 0 and global_t < args.start_timesteps else False
                evaluations.append(eval_policy(policy, eval_env, args.seed, render_ui, tb, eval_num))
                np.save(home_dir + f"results/{file_name}", evaluations)
                if args.save_model:
                    policy.save(home_dir + f"models/{file_name}")

        global_t += 1

