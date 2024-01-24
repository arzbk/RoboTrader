import os

# Passed to an instance of RoboTrader to set experiment (metadata) params
class Experiment:

    def __init__(
        self,
        exp_name=os.path.basename(__file__)[: -len(".py")],
        seed=42,
        torch_deterministic=True,
        cuda=True,
        track=False,
        wandb_project_name="RoboTrader",
        wandb_entity=None,
        capture_video=False,
        save_model=True,
        upload_model=False,
        hf_entity="",
    ):
        self.exp_name = exp_name
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.cuda = cuda
        self.track = track
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.capture_video = capture_video
        self.save_model = save_model
        self.upload_model = upload_model
        self.hf_entity = hf_entity


# Passed to an instance of RoboTrader to set algorithm params (learning related)
class Algorithm:

    def __init__(
        self,
        total_timesteps=1000000,
        epsilon_start=4.00,
        epsilon_end=1.00,
        actor_learning_rate=1e-5,
        critic_learning_rate=2e-4,
        buffer_size=int(1e6),
        gamma=0.90,
        tau=0.003,
        batch_size=1024,
        eval_episodes=3,
        policy_noise=0.15,
        exploration_noise=0.15,
        learning_starts=8e4,
        policy_frequency=2,
        noise_clip=0.5,
        num_envs=1,
        debug_mode=False,
        profile_mode=False,
    ):
        self.total_timesteps = total_timesteps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eval_episodes = eval_episodes
        self.policy_noise = policy_noise
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.noise_clip = noise_clip
        self.num_envs = num_envs
        self.debug_mode = debug_mode
        self.profile_mode = profile_mode


# Passed to an instance of RoboTrader to set trading parameters
class TradingParams:
    def __init__(self, cash, max_trade_perc, max_drawdown, short_selling,
                 rolling_window_size, period_months, lookback_steps,
                 fixed_start_date, range_start_date, range_end_date,
                 fixed_portfolio, use_fixed_trade_cost, fixed_trade_cost,
                 perc_trade_cost, num_assets, include_ti, indicator_list,
                 indicator_args, include_news):
        self.cash = cash
        self.max_trade_perc = max_trade_perc
        self.max_drawdown = max_drawdown
        self.short_selling = short_selling
        self.rolling_window_size = rolling_window_size
        self.period_months = period_months
        self.lookback_steps = lookback_steps
        self.fixed_start_date = fixed_start_date
        self.range_start_date = range_start_date
        self.range_end_date = range_end_date
        self.fixed_portfolio = fixed_portfolio
        self.use_fixed_trade_cost = use_fixed_trade_cost
        self.fixed_trade_cost = fixed_trade_cost
        self.perc_trade_cost = perc_trade_cost
        self.num_assets = num_assets
        self.include_ti = include_ti
        self.indicator_list = indicator_list
        self.indicator_args = indicator_args
        self.include_news = include_news