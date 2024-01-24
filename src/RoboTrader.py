# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import logging

# My imports
from TradingSim.Env import StockMarket
from TradingSim.DataManager import DataManager
from Algorithms.TD3.PyTorch import TD3
from TradingSim.async_envs import *

class RoboTrader:

    def __init__(self, exp_params, algo_params, train_params, test_params, use_cpu=False):

        # Parameters for this run
        self.exp_p = exp_params
        self.algo_p = algo_params
        self.train_p = train_params
        self.test_p = test_params

        # Initialize metric tracking
        self.init_tracking()

        # Configure logger - SET LOG LEVEL HERE
        if self.algo_p.debug_mode:
            logging.basicConfig(filename='Environment.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        # Set device and print to screen
        self.device = torch.device("cpu") if use_cpu else torch.device("cuda" if torch.cuda.is_available() and self.exp_p.cuda else "cpu")
        print(f"Running algorithm on {self.device.type}...")


    # Returns True "percentage" amount of the time - used for taking random actions
    def _evaluate_epsilon(self, A, B, t, t_max):
        t = max(0, min(t, t_max))
        percentage = A + (B - A) * (t / t_max)
        random_number = random.random()
        return random_number <= (percentage / 100)


    def _get_random_seed(self):
        return round(time.time())


    def _make_env(self, **kwargs):
        def thunk():
            env = StockMarket(
                **kwargs
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(kwargs['seed'])
            return env

        return thunk


    def init_tracking(self):

        # Setup unique and auto-incrementing folder for run summary
        run_i = 1
        run_summary_base = f"runs/RT"
        run_summary_path = f"{run_summary_base}_{run_i}/"
        while os.path.exists(run_summary_path):
            run_i += 1
            run_summary_path = f"{run_summary_base}_{run_i}/"
        self.run_name = run_summary_path
        self.writer = SummaryWriter(self.run_name)

        # Start Tensorboard based on generated run folder
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', run_summary_path])
        url = tb.launch()
        print(f"Tensorboard listening on {url}")

        """
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        """

        if self.exp_p.track:

            import wandb

            wandb.init(
                project=self.exp_p.wandb_project_name,
                entity=self.exp_p.wandb_entity,
                sync_tensorboard=True,
                config={
                    'Experiment': self.exp_p.__dict__,
                    'Algorithm': self.algo_p.__dict__,
                    'Train_Environment': self.train_p.__dict__,
                    'Test_Environment': self.test_p.__dict__
                },
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

    def train(self):

        # TRY NOT TO MODIFY: seeding
        random.seed(self.exp_p.seed)
        np.random.seed(self.exp_p.seed)
        torch.manual_seed(self.exp_p.seed)
        torch.backends.cudnn.deterministic = self.exp_p.torch_deterministic

        # Intialize data manager for training
        dm = DataManager(
            seed=self.exp_p.seed,
            rolling_window_size=30,
            range_start_date=self.train_p.range_start_date,
            range_end_date=self.train_p.range_end_date,
            fixed_portfolio=None,
            fixed_start_date=None,
            period_months=self.train_p.period_months,
            num_envs=self.algo_p.num_envs,
            num_assets=self.train_p.num_assets,
            include_ti=self.train_p.include_ti,
            indicator_list=self.train_p.indicator_list,
            indicator_args=self.train_p.indicator_args,
            episodes=250
        )

        # trn env setup
        env_list = [
            self._make_env(
                seed=self.exp_p.seed,
                env_num=i,
                cash=self.train_p.cash,
                max_trade_perc=self.train_p.max_trade_perc,
                max_drawdown=self.train_p.max_drawdown,
                include_ti=self.train_p.include_ti,
                period_months=self.train_p.period_months,
                num_assets=self.train_p.num_assets,
                fixed_start_date=self.train_p.fixed_start_date,
                fixed_portfolio=self.train_p.fixed_portfolio,
                use_fixed_trade_cost=self.train_p.use_fixed_trade_cost,
                perc_trade_cost=self.train_p.perc_trade_cost,
                lookback_steps=self.train_p.lookback_steps,
                indicator_list=self.train_p.indicator_list,
                indicator_args=self.train_p.indicator_args
            )
            for i in range(self.algo_p.num_envs)
        ]
        envs = get_envs(env_list)
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        """
        # Intialize data manager for evals
        dm_e = DataManager(
            seed=self.exp_p.seed,
            rolling_window_size=30,
            range_start_date=self.test_p.range_start_date,
            range_end_date=self.test_p.range_end_date,
            fixed_portfolio=['QCOM', 'MSFT'],
            fixed_start_date=datetime(2016, 1, 1),
            period_months=self.test_p.period_months,
            num_envs=self.algo_p.num_envs,
            num_assets=self.test_p.num_assets,
            include_ti=self.test_p.include_ti,
            indicator_list=self.test_p.indicator_list,
            indicator_args=self.test_p.indicator_args,
            episodes=250
        )
        """

        eval_env = StockMarket(
            seed=self.exp_p.seed+1,
            cash=self.test_p.cash,
            max_trade_perc=self.test_p.max_trade_perc,
            max_drawdown=self.test_p.max_drawdown,
            include_ti=self.test_p.include_ti,
            period_months=self.test_p.period_months,
            num_assets=self.test_p.num_assets,
            fixed_start_date=self.test_p.fixed_start_date,
            fixed_portfolio=self.test_p.fixed_portfolio,
            use_fixed_trade_cost=self.test_p.use_fixed_trade_cost,
            perc_trade_cost=self.test_p.perc_trade_cost,
            lookback_steps=self.test_p.lookback_steps,
            indicator_list=self.test_p.indicator_list,
            indicator_args=self.test_p.indicator_args
        )

        envs.single_observation_space.dtype = np.float32
        replay_buffer = ReplayBuffer(
            self.algo_p.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

        # Initialize the TD3 Algorithm, create networks, and commit to GPU
        p = TD3(
            envs,
            self.device,
            self.algo_p.tau,
            self.algo_p.gamma,
            self.algo_p.noise_clip,
            self.algo_p.policy_frequency,
            self.algo_p.policy_noise,
            replay_buffer,
            self.algo_p.actor_learning_rate,
            self.algo_p.critic_learning_rate,
            self.algo_p.exploration_noise,
            envs.single_action_space.low,
            envs.single_action_space.high,
            self.algo_p.batch_size,
            debug=self.algo_p.debug_mode
        )

        # Log last training start time
        self.last_start_time = time.time()

        # episode count
        episode_num = 1
        eval_episode = 1

        # Enable batch norm learning
        p.switch_to_train_mode()

        # Perform initial reset
        obs, _ = envs.reset()

        # Begin training loop
        for global_step in range(self.algo_p.total_timesteps):

            # ALGO LOGIC: put action logic here
            if global_step < self.algo_p.learning_starts or self._evaluate_epsilon(
                    self.algo_p.epsilon_start,
                    self.algo_p.epsilon_end,
                    global_step,
                    self.algo_p.total_timesteps):
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

            else:
                actions = p.get_actions(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # Log various metrics to file when episode is complete
            if "final_info" in infos:
                for info in infos["final_info"]:

                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], episode_num)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], episode_num)

                    # Log actions taken by network
                    self.writer.add_scalars(f"Action/Type", {
                        "BUY": info['action_counts']["BUY"],
                        "SELL": info['action_counts']["SELL"],
                        "HOLD": info['action_counts']["HOLD"]
                    }, episode_num)

                    # Log total number of trades
                    self.writer.add_scalar("Action/TotalTrades",
                                      info['action_counts']["BUY"] + info['action_counts']["SELL"], episode_num)

                    # Log action quantities taken by network
                    self.writer.add_scalars(f"Action/Quantity", {
                        "BUY": info['action_avgs']["BUY"],
                        "SELL": info['action_avgs']["SELL"],
                        "HOLD": info['action_avgs']["HOLD"]
                    }, episode_num)

                    # Log net worth over time
                    self.writer.add_scalar(f"Action/Profit", info['net_worth'] - 10000, episode_num)

                    episode_num += 1

                    print(f"Episode {episode_num} Complete.")

                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            # Add to Replay Buffer
            for i in range(envs.num_envs):
                replay_buffer.add(obs[i], real_next_obs[i], actions[i], rewards[i], terminations[i], infos[i])

            # Get next observation
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.algo_p.learning_starts:
                p.train_on_batch(update_policy=global_step % self.algo_p.policy_frequency)

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", p.qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf2_values", p.qf2_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", p.qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf2_loss", p.qf2_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf_loss", p.qf_loss.item() / 2.0, global_step)
                    self.writer.add_scalar("losses/actor_loss", p.actor_loss.item(), global_step)
                    self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.last_start_time)), global_step)

            if global_step % 20000 == 0 and global_step >= self.algo_p.learning_starts:

                # Disable batch norm learning
                p.switch_to_eval_mode()

                logging.info(f"===== EVAL MODEL STARTED =====")

                # Eval policy on unseen data / date-range
                for i in range(0, self.algo_p.eval_episodes):
                    logging.info(f"- Starting eval episode {eval_episode + i}...")

                    reward, profit = p.evaluate(eval_env, self.exp_p.seed + eval_episode + i)
                    self.writer.add_scalar('Evals/Eval Profit', profit, eval_episode + i)
                    self.writer.add_scalar('Evals/Eval Reward', reward, eval_episode + i)

                    logging.debug("[METRICS FOR EVAL EPISODE]:"
                        +f"- Profit: {profit:.2f}"
                        +f"- Reward: {reward:.4f}"
                    )

                logging.info(f"===== EVAL MODEL FINISHED =====")

                # Re-Enable batch norm learning
                p.switch_to_train_mode()

                eval_episode += self.algo_p.eval_episodes

                model_path = f"{self.run_name}{self.exp_p.exp_name}.cleanrl_model"
                torch.save((p.actor.state_dict(), p.qf1.state_dict(), p.qf2.state_dict()), model_path)
                print(f"model saved to {model_path}")

        envs.close()
        self.writer.close()
