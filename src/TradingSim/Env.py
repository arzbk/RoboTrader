import sys
import gymnasium as gym
import numpy as np
import random
from .Data import StockData
from .Graphics import StockChart
import logging

# Configure Numpy to throw divide by zero as exception and halt program
np.seterr(all='raise')

# Iterate through MatPlotLib backends

class StockMarket(gym.Env):

    def __init__(self,
                 cash=10000,
                 max_trade_perc=0.5,
                 max_drawdown=0.95,
                 short_selling=False,
                 rolling_window_size=30,
                 period_months=24,
                 lookback_steps=20,
                 fixed_start_date=None,
                 range_start_date=None,
                 range_end_date=None,
                 fixed_portfolio=False,
                 use_fixed_trade_cost=False,
                 fixed_trade_cost=None,
                 perc_trade_cost=None,
                 holding_cost=0.02, # 2% reduction from unrealized asset value every time step - compounding
                 num_assets=5,
                 include_ti=False,
                 indicator_list=None,
                 indicator_args={},
                 include_news=False
                ):

        super(StockMarket, self).__init__()

        # Dynamically build action space based on number of assets
        self.action_space = gym.spaces.Box(
            low=np.tile(np.array([0, 0]), num_assets),
            high=np.tile(np.array([3, 1]), num_assets),
            shape=(2 * num_assets,),
            dtype=np.float32
        )

        # Dynamically build state space based on parameters
        obs_dim_cnt = 1 + (num_assets * int(include_news)) + (num_assets * (2 + (int(include_ti)*len(indicator_list))))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim_cnt,), dtype=np.float32)

        self.render_mode = None

        # Intialize a live charting display for human feedback
        self.has_ui = False
        self.chart = StockChart(toolbox=False, include_table=True, table_cols=['Cash', 'Shares', 'Value', 'Net'])

        # Variables that define how the training/observations will work
        self.use_fixed_trade_cost = use_fixed_trade_cost
        self.trade_cost = fixed_trade_cost
        self.perc_trade_cost = perc_trade_cost
        self.total_cash = cash
        self.p_months = period_months
        self.lookback_steps = lookback_steps
        self.num_assets = num_assets or len(fixed_portfolio)
        self.include_ti = include_ti
        self.include_news = include_news
        self.max_trade_perc = max_trade_perc
        self.short_selling = short_selling
        self.rolling_window_size = rolling_window_size
        self.holding_cost = holding_cost

        # Instatiate StockData class
        self.data = StockData(
            filename="Stock Data/sp500_stocks.csv",
            num_assets=self.num_assets,
            period_months=period_months,
            fixed_start_date=fixed_start_date,
            range_start_date=range_start_date,
            range_end_date=range_end_date,
            fixed_portfolio=fixed_portfolio,
            include_ti=self.include_ti,
            indicator_list=indicator_list,
            indicator_args=indicator_args,
            rolling_window_size=rolling_window_size
        )

        # Cost and portfolio related vars
        self.action = {}
        self.assets = None
        self.last_close = None
        self.step_data = None
        self.remaining_cash = None
        self.net_worth = None
        self.current_reward = None
        self.cost_basis = None
        self.asset_value = None
        self.current_dd = None
        self.max_dd = max_drawdown
        self.hold_penalty = None
        self.shares_held = None
        self.current_price = None
        self.action_counts = None
        self.action_avgs = None


    # Resets env and state
    def reset(self, seed, **options):
        new_tickers = False
        new_dates = False
        has_ui = False
        if 'new_tickers' in list(options.keys()):
            new_tickers = options['new_tickers']
        elif 'new_dates' in list(options.keys()):
            new_dates = options['new_dates']
        elif 'has_ui' in list(options.keys()):
            has_ui = options['has_ui']

        # Set to True if we are rendering UI for this iteration
        self.has_ui = has_ui

        # Set random seed to be used system wide
        random.seed(a=seed)

        # Reset algorithm
        self.remaining_cash = self.total_cash
        self.current_reward = 0
        self.net_worth = self.total_cash
        self.current_dd = 0
        self.shares_held = {}
        self.cost_basis = {}
        self.asset_value = {}
        self.hold_penalty = {}
        self.current_price = {}
        self.action_counts = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        self.action_avgs = {'BUY': 0, 'HOLD': 0, 'SELL': 0}

        # Reset Stock Data
        self.assets = self.data.reset(seed, new_tickers=new_tickers, new_dates=new_dates)
        for asset in self.assets:
            self.shares_held[asset] = 0
            self.cost_basis[asset] = 0
            self.asset_value[asset] = 0
            self.hold_penalty[asset] = 0
            self.current_price[asset] = 0

        if self.has_ui:

            # Reset chart
            self.chart.reset(self.data.get_leading_data())
            self.chart.show()

        else:
            self.chart.hide()

        # Make first observation
        return self._next_observation(), {}


    # Used to prepare and package the next group of observations for the algo
    def _next_observation(self):

        """
         STATE SPACE = Rc, (So)*N, (Cp, As, (Ti)*n)*n, where...
         Rc = Remaining Cash (dim 1)
         So = Shares Owned (for each asset)
         (
            Cp = Close Price
            As = Average News Sentiment (NOT INCLUDED FOR NOW)
            Ti = Technical Indicator (for select group of 10)
        ) for each asset

        For 1 asset, and no sentiment score, this means a 13 dimensional vector
        """

        # Get next series of state data
        self.step_data = self.data.next()

        # Get random price factor and step data before proceeding
        perc_of_close = random.uniform(0.97, 1.03)

        # Initialize observation (state) array to be sent to the networks
        obs_arr = np.array([])

        # Remaining cash for network to use
        obs_arr = np.append(obs_arr, self.remaining_cash / self.net_worth)

        for asset in self.assets:

            # Set current price for asset for this next iteration
            self.current_price[asset] = self.step_data[asset]['Close'] * perc_of_close

            # Add price and shares held to the observation or state array
            total_possible_shares = (self.remaining_cash / self.current_price[asset]) + self.shares_held[asset]
            obs_arr = np.append(obs_arr, self.shares_held[asset] / total_possible_shares)

            # Build list of columns to use as features for state data
            col_postfix = "_scaled"
            col_list = ['Close' + col_postfix]
            if self.include_ti:
                col_list += [ind + col_postfix for ind in self.data.indicator_list]

            # Normalize and append features to observation array
            for col in col_list:
                col_val = self.step_data[asset][col]
                obs_arr = np.append(obs_arr, col_val)

        return obs_arr


    def _take_action(self, action):

        # Action needs to be reshaped from [A, B, A, B] -> [[A, B],[A, B]]
        action = action.reshape((-1, 2))

        # Build reward iteratively for portfolio on each step
        self.current_reward = 0

        """
        RULE: Process SELL actions first to free up cash for rest of portfolio, followed by BUY actions from
        largest to smallest...
        """
        # Pair actions with assets and split out as specified in rule
        buy_orders = []
        sell_orders = []
        pairs = zip(action, self.assets)
        ordered_pairs = sorted(pairs, key=lambda x: x[0][0])
        for pair in ordered_pairs:

            # Vars for action - asset pair
            action_tuple, asset = pair
            action, qty = action_tuple
            shares_held = self.shares_held[asset]
            current_price = self.current_price[asset]
            cost_basis = self.cost_basis[asset]
            prev_asset_value = self.asset_value[asset]
            prev_remaining_cash = self.remaining_cash
            prev_net_worth = self.net_worth

            # If sell action...
            if action <= 1.00 and shares_held > 0:

                self.action[asset] = "SELL"

                # Calculate how many shares to sell and update portfolio
                shares_sold_f = shares_held * self.max_trade_perc * qty
                shares_sold = int(shares_sold_f)

                # Calculate commission
                commission = None
                if self.use_fixed_trade_cost:
                    commission = self.trade_cost
                else:
                    commission = current_price * self.perc_trade_cost * shares_sold

                # Add funds to remaining cash
                share_value = shares_sold * current_price
                self.remaining_cash += (share_value - commission)

                # Update Share Count
                shares_held -= shares_sold
                assert shares_held >= 0, f"Cannot hold {shares_held} shares; value must be 0 or greater!"

                # Update Net Worth and Reward for timestep
                new_asset_value = shares_held * current_price
                self.net_worth += new_asset_value - prev_asset_value
                self.asset_value[asset] = new_asset_value
                self.current_reward += new_asset_value

                # Recalculate net worth
                self.net_worth += self.remaining_cash - prev_remaining_cash

                # Log key variables for debugging
                logging.debug(
                    f"[{asset}]: SELL ----- \n"
                    + f" - {shares_sold} SOLD @ ${current_price:.2f} == {(shares_sold * current_price):.2f}\n"
                    + f"- net worth = ${self.net_worth:.2f}\n"
                    + f"- previous asset value = ${prev_asset_value:.2f}\n"
                    + f"- new asset value = ${self.asset_value[asset]:.2f}\n"
                    + f"- current reward = {self.current_reward:.2f}\n"
                    + f"- new cash remaining = ${self.remaining_cash:.2f}\n"
                    + f"- new shares held = {shares_held}\n"
                    + f"- new shares sold = {shares_sold}\n"
                    + f"- new commission = {commission}\n"
                )

                # If all shares are sold, then cost basis is reset
                if shares_held == 0:
                    cost_basis = 0

            # If buy action...
            elif action >= 2.00 and self.remaining_cash > current_price and self.current_dd < self.max_dd:

                # Set current action
                self.action[asset] = "BUY"

                # Calculate remaining cash until max drawdown % reached
                remaining_dd = self.max_dd - self.current_dd
                fund_limit = self.net_worth * remaining_dd

                # Calculate available funds factoring in maximum trade percentage, and qty (% of max to trade)
                # then get min between product and fund limit (based on max drawdown)
                avail_funds = self.remaining_cash * self.max_trade_perc * qty
                avail_funds = min(avail_funds, fund_limit)

                # Lastly, calculate max number of shares that can be bought as well as commission payed
                shares_bought = None
                commission = None
                if self.use_fixed_trade_cost:
                    commission = self.trade_cost
                    shares_bought = int((avail_funds - commission) / current_price)
                else:
                    shares_bought = int(avail_funds / (current_price * (1 + self.perc_trade_cost)))
                    commission = shares_bought * (current_price * self.perc_trade_cost)

                prev_cost = None
                additional_cost = None

                if shares_bought > 0:

                    # Update cost basis (average cost per share)
                    prev_cost = cost_basis * shares_held
                    additional_cost = shares_bought * current_price
                    try:
                        cost_basis = ((prev_cost + additional_cost) / (shares_held + shares_bought))
                    except Exception:
                        print(f"PREV_COST {prev_cost}; ADDITIONAL {additional_cost}; shares_held {shares_held}; shares_bought {shares_bought}")
                        sys.exit()

                    # Update share count
                    shares_held += shares_bought

                    # Subtract costs from remaining cash and update net worth
                    self.remaining_cash -= (additional_cost + commission)

                    # Update Net Worth and Reward for timestep
                    new_asset_value = shares_held * current_price
                    self.net_worth += new_asset_value - prev_asset_value
                    self.asset_value[asset] = new_asset_value
                    self.current_reward += new_asset_value

                    # Recalculate net worth
                    self.net_worth += self.remaining_cash - prev_remaining_cash

                    # Log key variables for debugging
                    logging.debug(
                        f"[{asset}]: BUY ----- \n"
                        + f" - {shares_bought} BOUGHT @ ${current_price:.2f} == {(shares_bought * current_price):.2f}\n"
                        + f"- max_drawdown ({self.max_dd}) - current_drawdown ({self.current_dd}) = {remaining_dd}"
                        + f" (${fund_limit:.2f})\n"
                        + f"- Available funds (with max_trade_perc and drawdown limit = ${avail_funds:.2f}\n"
                        + f"- prev net worth = ${prev_net_worth:.2f}; net worth = ${self.net_worth:.2f}\n"
                        + f"- previous asset value = ${prev_asset_value:.2f}\n"
                        + f"- new asset value = ${self.asset_value[asset]:.2f}\n"
                        + f"- current reward = {self.current_reward:.2f}\n"
                        + f"- new cash remaining = ${self.remaining_cash:.2f}\n"
                        + f"- new shares held = {shares_held}\n"
                        + f"- new shares bought = {shares_bought}\n"
                        + f"- new commission = {commission}\n"
                    )

            # If hold, or failed sell/buy action...
            else:

                self.action[asset] = "HOLD"

                # Calculate current asset value
                new_asset_value = shares_held * current_price

                # Update current reward and net worth
                self.current_reward += new_asset_value
                self.net_worth += new_asset_value - prev_asset_value

                # Log key variables for debugging
                logging.debug(
                    f"[{asset}]: HOLD ----- \n"
                    + f"- current reward = {self.current_reward:.2f}\n"
                )

            # Recalculate drawdown
            self.current_dd = 1 - (self.remaining_cash / self.net_worth)

            # Update dictionaries
            self.shares_held[asset] = shares_held
            self.current_price[asset] = current_price
            self.cost_basis[asset] = cost_basis

            # Update stats on actions taken this episode
            self.action_counts[self.action[asset]] += 1
            self.action_avgs[self.action[asset]] = (((self.action_counts[self.action[asset]] - 1) * self.action_avgs[self.action[asset]]) + qty) / self.action_counts[self.action[asset]]

        # Update current reward
        self.current_reward += self.remaining_cash

        logging.debug(f"UPDATING CURRENT DRAWDOWN TO: {self.current_dd:.4f}")
        logging.debug(f"CURRENT REWARD FOR STEP: {self.current_reward:.2f}")

        return


    # Process a time step in the execution of trading simulation
    def step(self, action):

        # Get snapshots of reward and net worth
        prev_reward = self.current_reward
        prev_net = self.net_worth

        # Execute one time step within the environment
        self._take_action(action)

        # Calculate reward, and scale to between 0 and 100
        reward = 0
        if prev_reward and self.current_reward - prev_reward != 0:
            reward = ((self.current_reward - prev_reward) / prev_reward) * 100

        # Log reward calculation
        logging.debug(f"REWARD FOR STEP ==  {reward}\n")

        # Conditions for ending the training episode
        obs = None
        done = (self.data.current_step + self.data.start_index + 1) == self.data.max_steps

        # if ui enabled, slow down processing for human eyes
        """
        if self.has_ui:
            self.render(action=self.action)
            time.sleep(0.05)
        """

        # Get next observation
        if not done:
            obs = self._next_observation()

        # Add info for tensorboard and debugging
        info = {
            'action_counts': self.action_counts,
            'action_avgs': self.action_avgs,
            'hold_penalty': self.hold_penalty,
            'net_change': self.net_worth - prev_net,
            'net_worth': self.net_worth,
            'action': self.action
        }

        return obs, self.net_worth - prev_net, done, {}, info

    # Render the stock and trading decisions to the screen
    #TODO: Update charting to support portfolio of assets
    def render(self, action=None):

        if action == "BUY":
            self.chart.mark_action("BUY")

        elif action == "SELL" and self.current_price > self.cost_basis:
            self.chart.mark_action("SELL", "PROFIT")

        elif action == "SELL" and self.current_price < self.cost_basis:
            self.chart.mark_action("SELL", "LOSS")

        self.chart.add_step_data(self.step_data)
        self.chart.update_metrics([
            self.remaining_cash,
            self.shares_held,
            self.cost_basis,
            self.net_worth
        ])

