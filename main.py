# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import sys
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
import stable_baselines3 as sb3
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import logging

# My imports
from env import StockMarket
from datetime import datetime
from algorithms.TD3.PyTorch import TD3

def get_random_seed():
    return round(time.time())

@dataclass
class Args:
    pass

@dataclass
class Experiment(Args):

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = get_random_seed()
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

@dataclass
class Algorithm(Args):

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    epsilon_start: float = 4.00
    epsilon_end: float = 1.00
    """The odds on any given time-step that a random action is taken vs using the actor network"""
    actor_learning_rate: float = 1e-5
    """the learning rate of the actor (adam optimizer)"""
    critic_learning_rate: float = 2e-4
    """the learning rate of the critic (adam optimizer)"""
    buffer_size: int = int(1e7) # Default value of 1e6
    """the replay memory buffer size"""
    gamma: float = 0.80
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    eval_episodes: int = 5
    """Number of episodes used to evaluate model on un-seen data"""
    policy_noise: float = 0.15
    """the scale of policy noise"""
    exploration_noise: float = 0.15
    """the scale of exploration noise"""
    learning_starts: int = 8e4
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    debug_mode: bool = False
    """Enables debugging data logging and conditional code"""

@dataclass
class TradingParams(Args):

    cash = 10000,
    """Initial cash for the portfolio"""
    max_trade_perc = 1.0,
    """Capped percentage of cash available for trading on any timestep"""
    max_drawdown = 0.90,
    """Limit on maximum amount of capital that can be allocated at a given time"""
    short_selling = False,
    """Enables/disables short-selling"""
    rolling_window_size = 30,
    """Size of rolling window used when applying standardization/scaling to state"""
    period_months = 24,
    """Number of months the algorithm will trade for"""
    lookback_steps = 20,
    """The number of days the technical indicators will use in calculations"""
    fixed_start_date = None,
    """(Optional) Sets a fixed start date for each trading episode"""
    range_start_date = None,
    range_end_date = None,
    """(Optional) Set a range in which random start dates are selected for each trading episode
    NOTE: Start dates will fall between specified start date, and specified end date MINUS period_months"""
    fixed_portfolio = False,
    """(Optional) Define a fixed portfolio of assets to trade"""
    use_fixed_trade_cost = False,
    """(Optional) Whether to use a fixed or percentage-based commission scheme on trades"""
    fixed_trade_cost = None,
    """(Optional) Defines a fixed commission fee per trade"""
    perc_trade_cost = None,
    """(Optional) Defines a percentage-based commission fee per trade"""
    holding_cost = 0.02,
    """(Optional) Specify a cost (percentage) that is applied and compounded the longer an investment is held"""
    num_assets = 5,
    """(Optional) Specify a number of assets to randomly select for trading experiments"""
    include_ti = False,
    """Whether or not technical indicators will be computed and used in the observation/state"""
    indicator_list = None,
    """(Optional) A list of technical indicators to be used in state representation for each asset"""
    indicator_args = {},
    """(Optional) Arguments that are required for the listed indicators; script will fail if an arg is missing"""
    include_news = False
    # TODO:) NOT YET IMPLEMENTED: Enable/Disable average sentiment scores on related news per asset"""


# Configure logger - SET LOG LEVEL HERE
if Algorithm.debug_mode:
    logging.basicConfig(filename='Environment.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device and print to screen
device = torch.device("cuda" if torch.cuda.is_available() and Experiment.cuda else "cpu")
print(f"Running algorithm on {device.type}...")


# Returns True "percentage" amount of the time - used for taking random actions
def evaluate_epsilon(A, B, t, t_max):
    t = max(0, min(t, t_max))
    percentage = A + (B - A) * (t / t_max)
    random_number = random.random()
    return random_number <= (percentage / 100)


def make_env(seed, run_name, **kwargs):
    def thunk():
        env = StockMarket(
            **kwargs
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def run():

    args = tyro.cli(Experiment)
    run_name = f"StockMarket__{Experiment.exp_name}__{Experiment.seed}__{int(time.time())}"
    if args.track:

        import wandb

        wandb.init(
            project=Experiment.wandb_project_name,
            entity=Experiment.wandb_entity,
            sync_tensorboard=True,
            config=vars(Experiment),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Setup unique and auto-incrementing folder for run summary
    run_i = 1
    run_summary_base = f"runs/{run_name}"
    run_summary_path = f"{run_summary_base}_{run_i}/"
    while os.path.exists(run_summary_path):
        run_i += 1
        run_summary_path = f"{run_summary_base}_{run_i}/"
    writer = SummaryWriter(run_summary_path)

    # Start Tensorboard based on generated run folder
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', run_summary_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # Add custom scalar for grouped BUY, SELL, and HOLD graphs
    action_plots = {
        "Network Actions:": {
            "Action Type": ["Multiline", ["action/BUY", "action/SELL", "action/HOLD"]],
            "Quantity": ["Multiline", ["qty/BUY", "qty/SELL", "qty/HOLD"]],
        },
    }
    writer.add_custom_scalars(action_plots)

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(Experiment.seed)
    np.random.seed(Experiment.seed)
    torch.manual_seed(Experiment.seed)
    torch.backends.cudnn.deterministic = Experiment.torch_deterministic

    # Initialize environment
    lookback_steps = 20
    ti_list = ["SMA", "RSI", "EMA"] #, "STOCH", "MACD", "ADOSC"]
    ti_list_simple = ["SMA", "RSI", "EMA"]

    ti_args = {"RSI": {"timeperiod": lookback_steps}, "SMA": {"timeperiod": lookback_steps},
               "EMA": {"timeperiod": lookback_steps}}

    assets_used = ['QCOM', 'MSFT']
    """
           range_start_date=datetime(2007, 1, 1),
           range_end_date=datetime(2016, 1, 1),
           """

    # trn env setup
    trn_env = make_env(
        Experiment.seed,
        run_name,
        cash=10000,
        max_trade_perc=1,
        max_drawdown=0.95,
        include_ti=True,
        period_months=60,
        num_assets=2,
        fixed_start_date=datetime(2011, 1, 1),
        fixed_portfolio=assets_used,
        use_fixed_trade_cost=False,
        perc_trade_cost=0.02,
        lookback_steps=lookback_steps,
        indicator_list=ti_list,
        indicator_args=ti_args
    )
    envs = gym.vector.SyncVectorEnv([trn_env])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # eval env setup
    eval_env = StockMarket(
        cash=10000,
        max_trade_perc=1,
        max_drawdown=0.95,
        include_ti=True,
        period_months=24,
        num_assets=2,
        fixed_start_date=datetime(2016, 1, 1),
        fixed_portfolio=assets_used,
        use_fixed_trade_cost=False,
        perc_trade_cost=0.02,
        lookback_steps=lookback_steps,
        indicator_list=ti_list,
        indicator_args=ti_args
    )

    print(f"ACTOR LEARNING RATE: {Algorithm.actor_learning_rate}; CRITIC LEARNING RATE: {Algorithm.critic_learning_rate}")

    envs.single_observation_space.dtype = np.float32
    replay_buffer = ReplayBuffer(
        Algorithm.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # Initialize the TD3 Algorithm, create networks, and commit to GPU
    p = TD3(
        envs,
        device,
        Algorithm.tau,
        Algorithm.gamma,
        Algorithm.noise_clip,
        Algorithm.policy_frequency,
        Algorithm.policy_noise,
        replay_buffer,
        Algorithm.actor_learning_rate,
        Algorithm.critic_learning_rate,
        Algorithm.exploration_noise,
        envs.single_action_space.low,
        envs.single_action_space.high,
        Algorithm.batch_size,
        debug=Algorithm.debug_mode
    )

    start_time = time.time()

    # episode count
    episode_num = 1
    eval_episode = 1

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=Experiment.seed)
    for global_step in range(Algorithm.total_timesteps):

        # ALGO LOGIC: put action logic here
        if global_step < Algorithm.learning_starts or evaluate_epsilon(
                Algorithm.epsilon_start,
                Algorithm.epsilon_end,
                global_step,
                Algorithm.total_timesteps):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        else:

            # Enable batch norm learning
            p.actor.switch_to_train_mode()
            p.target_actor.switch_to_train_mode()

            actions = p.get_actions(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log various metrics to file when episode is complete
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], episode_num)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], episode_num)

                # Log actions taken by network
                writer.add_scalars(f"Action/Type", {
                    "BUY": info['action_counts']["BUY"],
                    "SELL": info['action_counts']["SELL"],
                    "HOLD": info['action_counts']["HOLD"]
                }, episode_num)

                # Log total number of trades
                writer.add_scalar("Action/TotalTrades",
                                  info['action_counts']["BUY"] + info['action_counts']["SELL"], episode_num)

                # Log action quantities taken by network
                writer.add_scalars(f"Action/Quantity", {
                    "BUY": info['action_avgs']["BUY"],
                    "SELL": info['action_avgs']["SELL"],
                    "HOLD": info['action_avgs']["HOLD"]
                }, episode_num)

                # Log net worth over time
                writer.add_scalar(f"Action/Profit", info['net_worth'] - 10000, episode_num)

                episode_num += 1
                print(f"EPISODE {episode_num} =====")

                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Add to Replay Buffer
        replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Get next observation
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > Algorithm.learning_starts:
            p.train_on_batch(update_policy=global_step % Algorithm.policy_frequency)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", p.qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", p.qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", p.qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", p.qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", p.qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", p.actor_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % 20000 == 0 and global_step >= Algorithm.learning_starts:

            # Enable batch norm learning
            p.actor.switch_to_eval_mode()
            p.target_actor.switch_to_eval_mode()

            logging.info(f"===== EVAL MODEL STARTED =====")

            # Eval policy on unseen data / date-range
            for i in range(0, Algorithm.eval_episodes):
                logging.info(f"- Starting eval episode {eval_episode + i}...")

                reward, profit = p.evaluate(eval_env, Experiment.seed + eval_episode + i)
                writer.add_scalar('Evals/Eval Profit', profit, eval_episode + i)
                writer.add_scalar('Evals/Eval Reward', reward, eval_episode + i)

                logging.debug("[METRICS FOR EVAL EPISODE]:"
                    +f"- Profit: {profit:.2f}"
                    +f"- Reward: {reward:.4f}"
                )

            logging.info(f"===== EVAL MODEL FINISHED =====")

            eval_episode += Algorithm.eval_episodes

            model_path = f"{run_summary_path}{Experiment.exp_name}.cleanrl_model"
            torch.save((p.actor.state_dict(), p.qf1.state_dict(), p.qf2.state_dict()), model_path)
            print(f"model saved to {model_path}")

    envs.close()
    writer.close()

if __name__ == "__main__":
    """
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    # Save the profiling results to a file
    profile_stats = pstats.Stats(profiler)
    profile_stats.dump_stats("perf.data")
    """
    run()
