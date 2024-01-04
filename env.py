import sys
import gymnasium as gym
from gym import spaces
from sklearn.preprocessing import StandardScaler
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

        # Check assertions on params
        assert use_sp500 or num_assets > 1, "Assert Failed: Cannot trade only SPY and have multiple assets!"
        
        # Dynamically build action space based on number of assets
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_assets,))
        
        # Dynamically build state space based on parameters
        obs_dim_cnt = 1 + (num_assets * int(include_news)) + (num_assets * (2 + len(indicator_list)))
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim_cnt,))

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
            indicator_args=indicator_args
        )

        # Cost and portfolio related vars
        self.assets = None
        self.last_close = None
        self.step_data = None
        self.remaining_cash = None
        self.net_worth = None
        self.cost_basis = None
        self.shares_held = None
        self.current_price = None

        
    # Resets env and state    
    def reset(self, seed, new_tickers=False, new_dates=True, has_ui=False):

        # Set to True if we are rendering UI for this iteration
        self.has_ui = has_ui

        # Set random seed to be used system wide
        random.seed(a=seed)

        # Reset algorithm
        self.remaining_cash = self.total_cash
        self.net_worth = self.total_cash
        self.shares_held = {}
        self.cost_basis = {}
        self.current_price = {}

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
        return self._next_observation()

    
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
        obs_arr = np.array()

        # Remaining cash for network to use
        obs_arr = np.append(obs_arr, self.remaining_cash)

        for asset in self.assets:

            # Set current price for asset for this next iteration
            self.current_price[asset] = self.step_data[asset]['Close'] * perc_of_close

            # Add price and shares held to the observation or state array
            obs_arr = np.append(obs_arr, self.shares_held[asset])

            """ OLD WAY FOR SHARES HELD
            # For shares held, express as percentage of total possible shares at today's rate
            so = self.shares_held / (self.shares_held + (self.current_price / self.remaining_cash))
            """

            # Build list of columns to use as features for state data
            col_list = ['Close_delta']
            if self.include_ti:
                col_list += [ind + "_delta" for ind in self.data.indicator_list]

            # Normalize and append features to observation array
            for col in col_list:
                sc = StandardScaler()
                col_series = self.data[asset][-self.rolling_window_size:][col]
                shaped_series = col_series.to_numpy().reshape(-1, 1)
                scaled_series = sc.fit_transform(shaped_series)
                obs_arr = np.append(obs_arr, scaled_series[-1][0])

        return obs_arr


    def _take_action(self, action):

        """
        RULE: Process SELL actions first to free up cash for rest of portfolio, followed by BUY actions from
        largest to smallest...
        """
        # Pair actions with assets and split out as specified in rule
        buy_orders = []
        sell_orders = []
        pairs = zip(action, self.assets)
        ordered_pairs = sorted(pairs, key=lambda x: x[0])
        for pair in ordered_pairs:

            # Vars for action - asset pair
            qty, asset = pair
            qty = abs(qty)
            shares_held = self.shares_held[asset]
            current_price = self.current_price[asset]
            cost_basis = self.cost_basis[asset]

            # If sell action...
            if action < 0:

                self.action[asset] = "SELL"

                # Calculate how many shares to sell and update portfolio
                shares_sold = int(shares_held * qty)
                self.remaining_cash += ((shares_sold * current_price) - self.trade_cost)
                shares_held -= shares_sold

                # If all shares are sold, then cost basis is reset
                if shares_held == 0:
                    cost_basis = 0

            # If hold action...
            elif action == 0:
                self.action[asset] = "HOLD"

            # If buy action...
            else:

                self.action[asset] = "BUY"

                # Calculate how many shares to buy
                total_possible = int((self.remaining_cash - self.trade_cost) / current_price)
                shares_bought = int((total_possible * qty) * self.max_trade_perc)

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

            # Update Net Worth for both actions above
            self.net_worth = self.remaining_cash + (shares_held * current_price)

        return


    def calculate_reward(self, prev_net):

        # Calculate difference in net worth from t-1 to t
        net_change = self.net_worth - prev_net
        #print(f"Net: ${self.net_worth:.2f} vs. Prev Net: ${prev_net:.2f} == reward: {str(net_change)}")
        return net_change
        
    
    # Process a time step in the execution of trading simulation
    def step(self, action):

        # Get net_worth before action as portfolio value @ t - 1
        prev_net = self.net_worth

        # Execute one time step within the environment
        self._take_action(action)

        # Calculate reward for action
        reward = self.calculate_reward(prev_net)

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

        return obs, reward, done, {}
            
            
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

