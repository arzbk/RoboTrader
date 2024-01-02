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
                 render_mode="chart",
                 cash=10000,
                 max_trade_perc=1.0,
                 short_selling=True,
                 rolling_window_size=60,
                 period_months=6,
                 lookback_steps=14,
                 use_sp500=False,
                 num_assets=1,
                 include_ti=False,
                 indicator_list=None,
                 indicator_args={},
                 include_news=False
                ):

        super(StockMarket, self).__init__()
        
        # 2D Action Space - 1st D = action, 2nd D = Quantity
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float64)
        
        # Include last 5 observations of 5 diff values (OHLCV)
        obs_dim_cnt = 1 + (num_assets * int(include_news)) + (num_assets * (2 + len(indicator_list)))
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float64)

        self.render_mode = render_mode
        
        # Intialize a live charting display for human feedback
        self.ui = None
        if self.render_mode.upper() == "CHART":
            self.has_ui = True
            self.chart = StockChart(toolbox=False, include_table=True, table_cols=['Cash', 'Shares', 'Value', 'Net'])
        else:
            self.has_ui = False

        # Variables that define how the training/observations will work
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

        # Instatiate StockData class
        self.data = StockData(
            filename="Stock Data/sp500_stocks.csv",
            period_months=period_months,
            num_assets=num_assets,
            use_sp500=use_sp500,
            include_ti=self.include_ti,
            indicator_list=indicator_list,
            indicator_args=indicator_args
        )

        # Cost and portfolio related vars
        self.last_close = None
        self.step_data = None
        self.remaining_cash = None
        self.net_worth = None
        self.cost_basis = None
        self.shares_held = None
        self.current_price = None

        
    # Resets env and state    
    def reset(self, seed, new_ticker=True, new_dates=True):

        # Set random seed to be used system wide
        random.seed(a=seed)

        # Reset algorithm
        self.remaining_cash = self.total_cash
        self.net_worth = self.total_cash
        self.shares_held = 0
        self.cost_basis = 0

        # Reset Stock Data
        self.data.reset(seed, new_ticker=True, new_dates=True)

        if self.has_ui:

            # Reset chart
            self.chart.reset(self.data.get_leading_data())
            self.chart.show()

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

        # Set the current price to random price somewhere close to the close price
        perc_of_close = random.uniform(0.97, 1.03)
        self.step_data = self.data[0]
        self.current_price = self.step_data['Adj Close'] * perc_of_close

        # For remaining cash, express as percentage of total portfolio value
        rc = self.remaining_cash / self.net_worth

        # For shares held, express as percentage of total possible shares at today's rate
        so = self.shares_held / (self.shares_held + (self.current_price / self.remaining_cash))

        # Setup numpy array
        obs_arr = np.array([rc, so])

        # For close price and tech indicators, do a rolling z-score normalization
        col_list = ['Close']
        if self.include_ti:
            col_list += self.data.indicator_list

        for col in col_list:
            sc = StandardScaler()
            col_series = self.data[-self.rolling_window_size:][col]
            shaped_series = col_series.to_numpy().reshape(-1, 1)
            scaled_series = sc.fit_transform(shaped_series)
            obs_arr = np.append(obs_arr, scaled_series[-1][0])

        return obs_arr


    def _take_action(self, action):

        # Split action tuple into discrete categorical component and continuous component
        action_type, qty = action

        # Process BUY action (if cash available)
        if action_type <= 1 and self.net_worth >= self.current_price:

            self.action = "BUY"

            # Calculate how many shares to buy
            total_possible = int(self.remaining_cash / self.current_price)
            shares_bought = int((total_possible * qty) * self.max_trade_perc)

            # Calc and update average cost of position
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_price
            self.remaining_cash -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)

            # Update share count
            self.shares_held += shares_bought

        # Process SELL action (if stock is available)
        elif action_type <= 2:

            self.action = "SELL"

            # Calculate how many shares to sell
            shares_sold = int(self.shares_held * qty)
            
            # Update portfolio
            self.remaining_cash += shares_sold * self.current_price
            self.shares_held -= shares_sold

        else:

            self.action = "HOLD"

        # Update Net Worth for both actions above
        self.net_worth = self.remaining_cash + self.shares_held * self.current_price


        # If all shares are sold, then cost basis is reset
        if self.shares_held == 0:
            self.cost_basis = 0

        return
        
    
    # Process a time step in the execution of trading simulation
    def step(self, action):

        # if ui enabled, slow down processing for human eyes
        if self.has_ui:
            self.render(action=action)
            time.sleep(0.1)

        # Execute one time step within the environment
        self._take_action(action)

        # Calculate reward for action
        delay_modifier = (self.data.current_step / self.data.max_steps)
        reward = self.net_worth * delay_modifier

        # Conditions for ending the training episode
        obs = None
        done = (self.data.current_step + self.data.start_index + 1) == self.data.max_steps

        # Get next observation
        if not done:

            # Go to next step
            self.data.next()

            obs = self._next_observation()

        return obs, reward, done, {}
            
            
    # Render the stock and trading decisions to the screen
    def render(self, action=None):

        if action[0] <= 1:
            self.chart.mark_action("BUY")

        elif action[0] <= 2 and self.current_price > self.cost_basis:
            self.chart.mark_action("SELL", "PROFIT")
        elif action[0] <= 2 and self.current_price < self.cost_basis:
            self.chart.mark_action("SELL", "LOSS")

        self.chart.add_step_data(self.step_data)
        self.chart.update_metrics([
            self.remaining_cash,
            self.shares_held,
            self.cost_basis,
            self.net_worth
        ])

