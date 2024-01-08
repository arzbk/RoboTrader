import sys
import gymnasium as gym
import numpy as np
import random
import time
from Data import StockData
from Graphics import StockChart
import pandas as pd

class StockMarket(gym.Env):

    def __init__(self,
                 cash=10000,
                 max_trade_perc=1.0,
                 max_drawdown=0.20,
                 short_selling=False,
                 rolling_window_size=60,
                 period_months=24,
                 lookback_steps=20,
                 fixed_start_date=None,
                 fixed_portfolio=False,
                 use_sp500=False,
                 log_transactions=False,
                 trade_cost=None,
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
        self.trade_cost = trade_cost if trade_cost else 0
        self.total_cash = cash
        self.p_months = period_months
        self.lookback_steps = lookback_steps
        self.num_assets = num_assets
        self.max_drawdown = max_drawdown
        self.include_ti = include_ti
        self.include_news = include_news
        self.action = None
        self.max_trade_perc = max_trade_perc
        self.short_selling = short_selling
        self.rolling_window_size = rolling_window_size
        self.log_transactions = log_transactions

        # Instatiate StockData class
        self.data = StockData(
            filename="Stock Data/sp500_stocks.csv",
            num_assets=num_assets,
            period_months=period_months,
            use_sp500=use_sp500,
            fixed_start_date=fixed_start_date,
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
        self.shares_held = None
        self.current_price = None
        self.action_counts = None
        self.action_avgs = None

        
    # Resets env and state    
    def reset(self, seed, **options):

        new_tickers = False
        new_dates = True
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
        self.shares_held = {}
        self.cost_basis = {}
        self.current_price = {}
        self.action_counts = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        self.action_avgs = {'BUY': 0, 'HOLD': 0, 'SELL': 0}

        # Reset Stock Data
        self.assets = self.data.reset(seed, new_tickers=new_tickers, new_dates=new_dates)
        for asset in self.assets:
            self.shares_held[asset] = 0
            self.cost_basis[asset] = 0
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

        # Get random price factor and step data before proceeding
        perc_of_close = random.uniform(0.97, 1.03)
        self.step_data = self.data[0]

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

            """ OLD WAY FOR SHARES HELD
            # For shares held, express as percentage of total possible shares at today's rate
            so = self.shares_held / (self.shares_held + (self.current_price / self.remaining_cash))
            """

            # Build list of columns to use as features for state data
            col_list = ['Close_norm']
            if self.include_ti:
                col_list += [ind + "_norm" for ind in self.data.indicator_list]

            # Normalize and append features to observation array
            for col in col_list:
                col_val = self.step_data[asset][col]
                obs_arr = np.append(obs_arr, col_val)

        return obs_arr


    def _take_action(self, action):

        # Action needs to be reshaped from [A, B, A, B] -> [[A, B],[A, B]]
        action = action.reshape((-1, 2))

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
            #print(asset, str(action_tuple))
            action, qty = action_tuple
            shares_held = self.shares_held[asset]
            current_price = self.current_price[asset]
            cost_basis = self.cost_basis[asset]

            # If sell action...
            if action <= 0.8 and shares_held > 0:

                self.action[asset] = "SELL"

                # Calculate how many shares to sell and update portfolio
                shares_sold = int(shares_held * qty)
                share_avg_cost = shares_sold * cost_basis
                share_value = shares_sold * current_price
                #self.current_reward += (share_value - share_avg_cost - self.trade_cost)
                self.remaining_cash += (share_value - self.trade_cost)
                shares_held -= shares_sold

                #print(f"[{asset}]: {shares_sold} @ ${current_price:.2f} == {(shares_sold * current_price):.2f}, leaving {self.remaining_cash:.2f}.")

                # If all shares are sold, then cost basis is reset
                if shares_held == 0:
                    cost_basis = 0

            # If buy action...
            elif action >= 2.2 and self.remaining_cash > current_price * 2:

                self.action[asset] = "BUY"

                # Calculate how many shares to buy
                capped_funds = self.max_drawdown * self.remaining_cash
                if self.num_assets > 1:
                    capped_funds = capped_funds * 0.75 # Only expose 75% of capped funds to allow remaining cash for other assets
                allocated_funds = int((capped_funds - self.trade_cost) / current_price)
                shares_bought = int(allocated_funds * qty)

                prev_cost = None
                additional_cost = None

                if shares_bought > 0:

                    # Calc and update average cost of position
                    prev_cost = cost_basis * shares_held
                    additional_cost = shares_bought * current_price
                    self.remaining_cash -= (additional_cost + self.trade_cost)

                    if (prev_cost + additional_cost) == 0 or (shares_held + shares_bought) == 0:
                        raise Exception(f"""Entered Error State: Invalid cost basis (division by zero). See below:
                                        Current Price: ${current_price:.2f}
                                        Previous Cost: ${prev_cost:.2f}
                                        Additional Cost: ${additional_cost:.2f}
                                        Shares Held: {shares_held}
                                        Shares Bought: {shares_bought}""")

                    # Update cost basis and share count
                    cost_basis = ((prev_cost + additional_cost) / (shares_held + shares_bought))
                    shares_held += shares_bought
                    #print(f"[{asset}]: {shares_bought} @ ${current_price:.2f} == {(shares_bought * current_price):.2f}, leaving {self.remaining_cash:.2f}.")

            # If hold, or failed sell/buy action...
            else:

                self.action[asset] = "HOLD"
                # print(f"[{asset}]: Hold")

            # Update dictionaries
            self.shares_held[asset] = shares_held
            self.current_price[asset] = current_price
            self.cost_basis[asset] = cost_basis

            # Update Net Worth for both actions above
            self.net_worth = self.remaining_cash + (shares_held * current_price)
            self.action_counts[self.action[asset]] += 1
            self.action_avgs[self.action[asset]] = (((self.action_counts[self.action[asset]] - 1) * self.action_avgs[self.action[asset]]) + qty) / self.action_counts[self.action[asset]]

        return


    def calculate_reward(self, prev_reward):

        # Calculate difference in net worth from t-1 to t
        net_reward = self.net_worth - prev_reward / prev_reward
        #print(f"Net: ${self.net_worth:.2f} vs. Prev Net: ${prev_net:.2f} == reward: {str(net_change)}")
        return net_reward
        
    
    # Process a time step in the execution of trading simulation
    def step(self, action):

        # Get net_worth before action as portfolio value @ t - 1
        prev_reward = self.net_worth

        # Execute one time step within the environment
        self._take_action(action)

        # Calculate reward for action
        reward = self.calculate_reward(prev_reward)
        self.current_reward = 0

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

            # Go to next step
            self.data.next()

            obs = self._next_observation()

        # Add info for tensorboard and debugging
        info = {
            'action_counts': self.action_counts,
            'action_avgs': self.action_avgs
        }

        return obs, reward, done, {}, info
            
            
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

