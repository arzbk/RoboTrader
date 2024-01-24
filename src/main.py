from RoboTrader import *
from TradingSim.Parameters import *
import multiprocessing
from datetime import datetime
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

if __name__ == "__main__":

    multiprocessing.freeze_support()

    # Specify Common Algo Params
    stocks_to_trade=None
    num_assets = 2
    lookback_steps = 20
    use_indicators = True
    indicators = ["SMA", "RSI", "EMA"]
    indicator_args = {
        "RSI": {"timeperiod": lookback_steps},
        "SMA": {"timeperiod": lookback_steps},
        "EMA": {"timeperiod": lookback_steps}
    }

    # Setup experiment params
    exp_params = Experiment(
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
        hf_entity=""
    )

    # Setup algo/learning params
    algo_params = Algorithm(
        total_timesteps=1000000,
        epsilon_start=4.00,
        epsilon_end=1.00,
        actor_learning_rate=1e-5,
        critic_learning_rate=2e-4,
        buffer_size=int(1e6),
        gamma=0.90,
        tau=0.003,
        batch_size=256,
        eval_episodes=3,
        policy_noise=0.15,
        exploration_noise=0.15, # 0.25 previously
        learning_starts=25e3,
        policy_frequency=2,
        noise_clip=0.5,
        num_envs=2,
        debug_mode=False,
        profile_mode=False
    )

    # Trading sim params for training
    tp_trn = TradingParams(
        cash=10000,
        max_trade_perc=1.0,
        max_drawdown=0.90,
        short_selling=False,
        rolling_window_size=30,
        period_months=60,
        lookback_steps=lookback_steps,
        fixed_start_date=datetime(2011, 1, 1),
        range_start_date=None,
        range_end_date=None,
        fixed_portfolio=stocks_to_trade,
        use_fixed_trade_cost=False,
        fixed_trade_cost=None,
        perc_trade_cost=0.03,
        num_assets=num_assets,
        include_ti=use_indicators,
        indicator_list=indicators,
        indicator_args=indicator_args,
        include_news=False
    )

    # Trading sim params for testing
    tp_tst = TradingParams(
        cash=10000,
        max_trade_perc=1.0,
        max_drawdown=0.90,
        short_selling=False,
        rolling_window_size=30,
        period_months=24,
        lookback_steps=lookback_steps,
        fixed_start_date=datetime(2016, 1, 1),
        range_start_date=None,
        range_end_date=None,
        fixed_portfolio=stocks_to_trade,
        use_fixed_trade_cost=False,
        fixed_trade_cost=None,
        perc_trade_cost=0.02,
        num_assets=num_assets,
        include_ti=use_indicators,
        indicator_list=indicators,
        indicator_args=indicator_args,
        include_news=False
    )

    rt = RoboTrader(exp_params, algo_params, train_params=tp_trn, test_params=tp_tst)
    rt.train()